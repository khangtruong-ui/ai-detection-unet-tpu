"""Dataset loading pipeline for SID-UNet JAX/TPU.

Supports real online Hugging Face datasets (e.g. KhangTruong/COCO-inpainted),
cached tensor datasets, and fast synthetic mock datasets with multi-worker parallelism.
"""

from __future__ import annotations

import collections.abc
import concurrent.futures
import itertools
import logging
import math
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from parquet_dataset_loader import (
    DataLoader,
    default_collate,
    numpy_collate,
    safe_dataloader_len,
)
from sid_unet.dataset.transforms import SpatialJointTransform
from sid_unet.dataset.mask_utils import process_mask

logger = logging.getLogger("sid_unet.dataset")


def resolve_num_workers(val: Any = None) -> int:
    """Resolve number of workers, defaulting to os.cpu_count().

    If val is None, negative, or 'auto', returns os.cpu_count() or 4.
    """
    if val is None:
        return max(1, os.cpu_count() or 4)
    if isinstance(val, str) and val.lower() in ("auto", "none", ""):
        return max(1, os.cpu_count() or 4)
    try:
        num = int(val)
        if num < 0:
            return max(1, os.cpu_count() or 4)
        return num
    except (ValueError, TypeError):
        return max(1, os.cpu_count() or 4)


def resolve_sample_limit(
    samples_val: Optional[Any],
    steps_val: Optional[Any],
    batch_size: int,
    default_samples: Optional[int] = None,
) -> Optional[int]:
    """Resolve sample count limit from samples_val or steps_val.

    If steps or samples <= 0 (e.g. -1), returns None (meaning run until dataset is depleted).
    """
    if steps_val is not None:
        try:
            steps_int = int(steps_val)
            if steps_int <= 0:
                return None
            return steps_int * batch_size
        except (ValueError, TypeError):
            pass

    if samples_val is not None:
        try:
            samples_int = int(samples_val)
            if samples_int <= 0:
                return None
            return samples_int
        except (ValueError, TypeError):
            pass

    return default_samples


def safe_dataloader_len(loader: Optional[Any]) -> Optional[int]:
    """Safely return DataLoader length, or None if dataset has no length (e.g. streaming)."""
    if loader is None:
        return None
    try:
        return len(loader)
    except (TypeError, NotImplementedError):
        return None


def process_raw_sample(
    sample: Dict[str, Any],
    target_image_size: Tuple[int, int] = (512, 512),
    transform: Optional[SpatialJointTransform] = None,
) -> Dict[str, np.ndarray]:
    """Process raw dataset sample (e.g. from KhangTruong/COCO-inpainted) into model arrays."""
    h, w = target_image_size

    raw_img = sample.get("image")
    if isinstance(raw_img, collections.abc.Mapping) and "bytes" in raw_img:
        raw_img = raw_img["bytes"]
    if isinstance(raw_img, (bytes, bytearray)):
        import io
        try:
            pil_img = Image.open(io.BytesIO(raw_img)).convert("RGB")
        except Exception:
            pil_img = Image.new("RGB", (w, h), color=(128, 128, 128))
    elif isinstance(raw_img, Image.Image):
        pil_img = raw_img.convert("RGB")
    elif isinstance(raw_img, np.ndarray):
        pil_img = Image.fromarray(raw_img).convert("RGB")
    else:
        pil_img = Image.new("RGB", (w, h), color=(128, 128, 128))

    if pil_img.size != (w, h):
        pil_img = pil_img.resize((w, h), Image.BILINEAR)

    img_arr = (np.array(pil_img, dtype=np.float32) / 127.5) - 1.0  # Normalized to [-1, 1]

    raw_mask = sample.get("mask")
    if isinstance(raw_mask, collections.abc.Mapping) and "bytes" in raw_mask:
        raw_mask = raw_mask["bytes"]
    if isinstance(raw_mask, (bytes, bytearray)):
        import io
        try:
            pil_mask = Image.open(io.BytesIO(raw_mask)).convert("L")
        except Exception:
            pil_mask = Image.new("L", (w, h), color=0)
    elif isinstance(raw_mask, Image.Image):
        pil_mask = raw_mask.convert("L")
    elif isinstance(raw_mask, np.ndarray):
        pil_mask = Image.fromarray(raw_mask).convert("L")
    else:
        pil_mask = Image.new("L", (w, h), color=0)

    if pil_mask.size != (w, h):
        pil_mask = pil_mask.resize((w, h), Image.NEAREST)

    mask_arr = (np.array(pil_mask, dtype=np.float32) >= 128.0).astype(np.float32)[:, :, None]

    # Infer label if not explicitly provided
    if "label" in sample and sample["label"] is not None:
        label = int(sample["label"])
    else:
        mask_sum = float(np.sum(mask_arr))
        total_px = float(mask_arr.size)
        if mask_sum == 0.0:
            label = 0  # Real / Untampered
        elif mask_sum >= total_px:
            label = 1  # Fully Synthetic
        else:
            label = 2  # Inpainted / Tampered

    if transform is not None:
        img_arr, mask_arr = transform(img_arr, mask_arr)

    return {
        "image": img_arr,
        "mask": mask_arr,
        "label": np.array([label], dtype=np.int32),
    }


class HFStreamingDataset:
    """Streaming dataset wrapper for Hugging Face datasets (e.g. KhangTruong/COCO-inpainted)."""

    def __init__(
        self,
        dataset_name: str = "KhangTruong/COCO-inpainted",
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        max_samples: Optional[int] = None,
        transform: Optional[SpatialJointTransform] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.max_samples = max_samples
        self.transform = transform
        self._load_dataset()

    def _load_dataset(self):
        from datasets import load_dataset
        self.hf_dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        count = 0
        limit = self.max_samples if (self.max_samples is not None and self.max_samples > 0) else None

        try:
            for item in self.hf_dataset:
                if limit is not None and count >= limit:
                    break
                try:
                    processed = process_raw_sample(
                        item,
                        target_image_size=self.image_size,
                        transform=self.transform,
                    )
                    yield processed
                    count += 1
                except Exception as e:
                    logger.debug(f"Skipping malformed dataset sample: {e}")
                    continue
        except Exception as e:
            logger.warning(f"⚠️ HF streaming interrupted: {e}. Gracefully yielding fallback samples to complete cycle.")
            needed = max(1, (limit - count) if limit else 4)
            mock_ds = MockDataset(num_samples=needed, image_size=self.image_size, transform=self.transform)
            for sample in mock_ds:
                yield sample
                count += 1
                if limit is not None and count >= limit:
                    break

    def __len__(self) -> int:
        if self.max_samples is not None and self.max_samples > 0:
            return self.max_samples
        if hasattr(self, "hf_dataset") and hasattr(self.hf_dataset, "info"):
            splits = getattr(self.hf_dataset.info, "splits", None)
            if splits and self.split in splits:
                num_ex = getattr(splits[self.split], "num_examples", None)
                if num_ex is not None and num_ex > 0:
                    return num_ex
        raise TypeError(f"'{type(self).__name__}' has no static length when max_samples is None")


class MockDataset:
    """Fast in-memory mock dataset for rapid testing, smoke checks, and benchmarking."""

    def __init__(
        self,
        num_samples: int = 64,
        image_size: Tuple[int, int] = (512, 512),
        is_cached: bool = False,
        latent_channels: int = 84,
        transform: Optional[SpatialJointTransform] = None,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.is_cached = is_cached
        self.latent_channels = latent_channels
        self.transform = transform

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        h, w = self.image_size
        rng = np.random.RandomState(idx)
        label = int(idx % 3)

        if label == 0:
            mask = np.zeros((h, w, 1), dtype=np.float32)
        elif label == 1:
            mask = np.ones((h, w, 1), dtype=np.float32)
        else:
            mask = np.zeros((h, w, 1), dtype=np.float32)
            mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, :] = 1.0

        if self.is_cached:
            lh, lw = h // 8, w // 8
            latent = rng.randn(lh, lw, self.latent_channels).astype(np.float32) * 0.1
            if self.transform is not None:
                latent, mask = self.transform(latent, mask)
            return {"latent": latent, "mask": mask, "label": np.array([label], dtype=np.int32)}

        image = (rng.rand(h, w, 3).astype(np.float32) * 2.0) - 1.0
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {"image": image, "mask": mask, "label": np.array([label], dtype=np.int32)}


class ParquetForensicsDataset(collections.abc.Sequence):
    """Forensics dataset adapter wrapping an IndexedParquetDataset.

    Applies `process_raw_sample` with spatial joint transforms and target sizing on the fly.
    Supports random access indexing, slicing, and length queries.
    """

    def __init__(
        self,
        parquet_dataset: Any,
        image_size: Tuple[int, int] = (512, 512),
        max_samples: Optional[int] = None,
        transform: Optional[SpatialJointTransform] = None,
    ):
        self.parquet_dataset = parquet_dataset
        self.image_size = image_size
        self.max_samples = max_samples
        self.transform = transform

    def __len__(self) -> int:
        total = len(self.parquet_dataset)
        if self.max_samples is not None and self.max_samples > 0:
            return min(total, self.max_samples)
        return total

    def __getitem__(self, idx: Union[int, slice, Sequence[int]]) -> Any:
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
            raw = self.parquet_dataset[idx]
            return process_raw_sample(
                raw,
                target_image_size=self.image_size,
                transform=self.transform,
            )
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(idx, (list, tuple)):
            return [self[i] for i in idx]
        else:
            raise TypeError(f"Invalid index type for {type(self).__name__}: {type(idx)}")


class DataLoaderIterator(DataLoader):
    """Batch generator over dataset with multi-worker prefetching and robust collation.

    Drop-in subclass of parquet_dataset_loader.DataLoader configured for JAX/Flax workloads.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: Optional[int] = None,
        prefetch_batches: int = 4,
        collate_fn: Optional[Callable[[Any], Any]] = None,
        timeout: float = 60.0,
        worker_type: str = "auto",
        pin_memory: bool = False,
        debug: bool = False,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        collator = collate_fn if collate_fn is not None else numpy_collate
        resolved_workers = resolve_num_workers(num_workers)
        prefetch_factor = max(1, prefetch_batches // max(1, resolved_workers)) if resolved_workers > 0 else 2
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=resolved_workers,
            collate_fn=collator,
            drop_last=drop_last,
            timeout=timeout,
            worker_type=worker_type,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            debug=debug,
            seed=seed,
        )

    def _collate(self, items: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Backward-compatibility collate method."""
        return self.collate_fn(items)


def create_dataloaders(
    config: Any,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create train, validation, and test dataloaders from configuration using parquet-dataset-loader."""
    data_cfg = getattr(config, "data", {}) if hasattr(config, "data") else config.get("data", {})
    img_size = tuple(data_cfg.get("image_size", [512, 512]))
    batch_size = int(data_cfg.get("batch_size", 8))
    dataset_name = str(data_cfg.get("dataset_name", "KhangTruong/COCO-inpainted"))
    prefetch_batches = int(data_cfg.get("prefetch_batches", 8))

    # Worker number defaults to os.cpu_count()
    raw_workers = data_cfg.get("num_workers", -1)
    num_workers = resolve_num_workers(raw_workers)
    logger.info(f"⚡ Data loading configured with {num_workers} workers (System CPU cores: {os.cpu_count()})")

    # Check if cached mode
    is_cached = bool(data_cfg.get("cached", False) or data_cfg.get("use_cache", False))

    augs = data_cfg.get("augmentations", {})
    train_transform = SpatialJointTransform(
        horizontal_flip=float(augs.get("horizontal_flip", 0.5)),
        vertical_flip=float(augs.get("vertical_flip", 0.0)),
        random_rotate90=float(augs.get("random_rotate90", 0.25)),
    )

    tr_cfg = getattr(config, "training", {}) if hasattr(config, "training") else (config.get("training", {}) if isinstance(config, dict) else {})
    steps_per_epoch_cfg = data_cfg.get("steps_per_epoch", tr_cfg.get("steps_per_epoch", None) if hasattr(tr_cfg, "get") else None)
    train_samples_cfg = data_cfg.get("train_samples_per_epoch", None)
    train_max_samples = resolve_sample_limit(
        samples_val=train_samples_cfg,
        steps_val=steps_per_epoch_cfg,
        batch_size=batch_size,
        default_samples=None,
    )

    val_steps_cfg = data_cfg.get("val_steps", tr_cfg.get("val_steps", None) if hasattr(tr_cfg, "get") else None)
    val_samples_cfg = data_cfg.get("val_samples", data_cfg.get("val_samples_per_epoch", None))
    val_max_samples = resolve_sample_limit(
        samples_val=val_samples_cfg,
        steps_val=val_steps_cfg,
        batch_size=batch_size,
        default_samples=None,
    )

    train_ds = None
    val_ds = None

    # Cached mode check
    if is_cached:
        cache_dir = data_cfg.get("cache_dir", None)
        if cache_dir and os.path.exists(cache_dir):
            import glob
            from sid_unet.cache.dataset import CachedShardDataset
            train_shards = glob.glob(os.path.join(cache_dir, "train", "*.npz"))
            val_shards = glob.glob(os.path.join(cache_dir, "validation", "*.npz"))
            if train_shards:
                train_ds = CachedShardDataset(train_shards, transform=train_transform)
            if val_shards:
                val_ds = CachedShardDataset(val_shards, transform=None)

    # Real dataset loading via parquet-dataset-loader
    if train_ds is None and dataset_name.lower() not in ("mock", "synthetic") and not is_cached:
        use_parquet = bool(data_cfg.get("use_parquet_loader", True))
        streaming = bool(data_cfg.get("streaming", True))
        save_to_disk = bool(data_cfg.get("save_to_disk", False))
        background_download = bool(data_cfg.get("background_download", False))
        columns = data_cfg.get("columns", None)
        token = data_cfg.get("token", None)
        train_split = str(data_cfg.get("train_split", "train"))
        val_split = str(data_cfg.get("val_split", "validation"))

        if use_parquet:
            try:
                import parquet_dataset_loader as pdl
                logger.info(f"🌐 Loading dataset '{dataset_name}' with parquet-dataset-loader (split='{train_split}')...")
                pdl_train = pdl.load_dataset(
                    path=dataset_name,
                    split=train_split,
                    streaming=streaming,
                    columns=columns,
                    token=token,
                    save_to_disk=save_to_disk,
                    background_download=background_download,
                )
                train_ds = ParquetForensicsDataset(
                    pdl_train,
                    image_size=img_size,
                    max_samples=train_max_samples,
                    transform=train_transform,
                )
                try:
                    pdl_val = pdl.load_dataset(
                        path=dataset_name,
                        split=val_split,
                        streaming=streaming,
                        columns=columns,
                        token=token,
                        save_to_disk=save_to_disk,
                        background_download=background_download,
                    )
                    val_ds = ParquetForensicsDataset(
                        pdl_val,
                        image_size=img_size,
                        max_samples=val_max_samples,
                        transform=None,
                    )
                except Exception as ve:
                    logger.debug(f"Could not load validation split '{val_split}': {ve}")
                    val_ds = None
            except Exception as e:
                logger.warning(f"⚠️ parquet-dataset-loader could not load '{dataset_name}': {e}. Trying fallback...")
                train_ds = None
                val_ds = None

        if train_ds is None:
            try:
                logger.info(f"🌐 Loading real dataset '{dataset_name}' from Hugging Face...")
                train_ds = HFStreamingDataset(
                    dataset_name=dataset_name,
                    split=data_cfg.get("train_split", "train"),
                    image_size=img_size,
                    max_samples=train_max_samples,
                    transform=train_transform,
                )
                val_ds = HFStreamingDataset(
                    dataset_name=dataset_name,
                    split=data_cfg.get("val_split", "validation"),
                    image_size=img_size,
                    max_samples=val_max_samples,
                    transform=None,
                )
            except Exception as e:
                logger.warning(f"⚠️ Could not stream dataset '{dataset_name}' ({e}). Falling back to fast MockDataset.")
                train_ds = None
                val_ds = None

    if train_ds is None:
        t_samples = train_max_samples if (train_max_samples is not None and train_max_samples > 0) else 64
        v_samples = val_max_samples if (val_max_samples is not None and val_max_samples > 0) else 32
        model_cfg = getattr(config, "model", {}) if hasattr(config, "model") else (config.get("model", {}) if isinstance(config, dict) else {})
        ts_dim = int(model_cfg.get("timestep_embed_dim", 32)) if hasattr(model_cfg, "get") else 32
        sig_dim = int(model_cfg.get("sigma_embed_dim", 32)) if hasattr(model_cfg, "get") else 32
        timesteps_len = len(model_cfg.get("timesteps", [250])) if hasattr(model_cfg, "get") else 1
        calc_latent_channels = 4 + timesteps_len * (16 + ts_dim + sig_dim)
        latent_channels = int(data_cfg.get("latent_channels", calc_latent_channels))

        train_ds = MockDataset(num_samples=t_samples, image_size=img_size, is_cached=is_cached, latent_channels=latent_channels, transform=train_transform)
        val_ds = MockDataset(num_samples=v_samples, image_size=img_size, is_cached=is_cached, latent_channels=latent_channels, transform=None)

    train_loader = DataLoaderIterator(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
        collate_fn=numpy_collate,
        pin_memory=bool(data_cfg.get("pin_memory", False)),
    )
    val_loader = (
        DataLoaderIterator(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
            collate_fn=numpy_collate,
            pin_memory=bool(data_cfg.get("pin_memory", False)),
        )
        if val_ds is not None
        else None
    )

    return train_loader, val_loader, None
