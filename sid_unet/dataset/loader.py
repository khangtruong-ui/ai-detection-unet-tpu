"""Dataset loading pipeline for SID-UNet JAX/TPU.

Supports real online Hugging Face datasets (e.g. KhangTruong/COCO-inpainted),
cached tensor datasets, and fast synthetic mock datasets with multi-worker parallelism.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import math
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

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


def process_raw_sample(
    sample: Dict[str, Any],
    target_image_size: Tuple[int, int] = (512, 512),
    transform: Optional[SpatialJointTransform] = None,
) -> Dict[str, np.ndarray]:
    """Process raw dataset sample (e.g. from KhangTruong/COCO-inpainted) into model arrays."""
    h, w = target_image_size

    raw_img = sample.get("image")
    if isinstance(raw_img, Image.Image):
        pil_img = raw_img.convert("RGB")
    elif isinstance(raw_img, np.ndarray):
        pil_img = Image.fromarray(raw_img).convert("RGB")
    else:
        pil_img = Image.new("RGB", (w, h), color=(128, 128, 128))

    if pil_img.size != (w, h):
        pil_img = pil_img.resize((w, h), Image.BILINEAR)

    img_arr = (np.array(pil_img, dtype=np.float32) / 127.5) - 1.0  # Normalized to [-1, 1]

    raw_mask = sample.get("mask")
    if isinstance(raw_mask, Image.Image):
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

    def __len__(self) -> int:
        if self.max_samples is not None and self.max_samples > 0:
            return self.max_samples
        return 2000


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


class DataLoaderIterator:
    """Batch generator over dataset with multi-worker prefetching and collation."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = resolve_num_workers(num_workers)
        self._is_map = hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__")

    def __len__(self) -> int:
        if self._is_map:
            total = len(self.dataset)
            if self.drop_last:
                return total // self.batch_size
            return math.ceil(total / self.batch_size)
        if hasattr(self.dataset, "__len__"):
            return math.ceil(len(self.dataset) / self.batch_size)
        return 100

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        if self._is_map:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                np.random.shuffle(indices)

            # When multi-worker is enabled and dataset is indexable, use parallel batch workers
            if self.num_workers > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    for start_idx in range(0, len(indices), self.batch_size):
                        batch_idxs = indices[start_idx : start_idx + self.batch_size]
                        if self.drop_last and len(batch_idxs) < self.batch_size:
                            break

                        items = list(executor.map(lambda i: self.dataset[i], batch_idxs))
                        yield self._collate(items)
            else:
                for start_idx in range(0, len(indices), self.batch_size):
                    batch_idxs = indices[start_idx : start_idx + self.batch_size]
                    if self.drop_last and len(batch_idxs) < self.batch_size:
                        break

                    items = [self.dataset[i] for i in batch_idxs]
                    yield self._collate(items)
        else:
            # Streaming iterable dataset
            batch = []
            for item in self.dataset:
                batch.append(item)
                if len(batch) == self.batch_size:
                    yield self._collate(batch)
                    batch = []
            if batch and not self.drop_last:
                yield self._collate(batch)

    def _collate(self, items: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        collated: Dict[str, np.ndarray] = {}
        for k in items[0].keys():
            collated[k] = np.stack([item[k] for item in items], axis=0)
        return collated


def create_dataloaders(
    config: Any,
) -> Tuple[DataLoaderIterator, Optional[DataLoaderIterator], Optional[DataLoaderIterator]]:
    """Create train, validation, and test dataloaders from configuration."""
    data_cfg = getattr(config, "data", {}) if hasattr(config, "data") else config.get("data", {})
    img_size = tuple(data_cfg.get("image_size", [512, 512]))
    batch_size = int(data_cfg.get("batch_size", 8))
    dataset_name = str(data_cfg.get("dataset_name", "KhangTruong/COCO-inpainted"))

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

    train_samples = int(data_cfg.get("train_samples_per_epoch", 64))
    val_samples = int(data_cfg.get("val_samples", 32))

    train_ds = None
    val_ds = None

    # Load real dataset if not 'mock' or 'synthetic'
    if dataset_name.lower() not in ("mock", "synthetic") and not is_cached:
        try:
            logger.info(f"🌐 Loading real dataset '{dataset_name}' from Hugging Face...")
            train_ds = HFStreamingDataset(
                dataset_name=dataset_name,
                split=data_cfg.get("train_split", "train"),
                image_size=img_size,
                max_samples=train_samples if train_samples > 0 else None,
                transform=train_transform,
            )
            val_ds = HFStreamingDataset(
                dataset_name=dataset_name,
                split=data_cfg.get("val_split", "validation"),
                image_size=img_size,
                max_samples=val_samples if val_samples > 0 else None,
                transform=None,
            )
        except Exception as e:
            logger.warning(f"⚠️ Could not stream dataset '{dataset_name}' ({e}). Falling back to fast MockDataset.")
            train_ds = None
            val_ds = None

    if train_ds is None:
        t_samples = train_samples if train_samples > 0 else 64
        v_samples = val_samples if val_samples > 0 else 32
        train_ds = MockDataset(num_samples=t_samples, image_size=img_size, is_cached=is_cached, transform=train_transform)
        val_ds = MockDataset(num_samples=v_samples, image_size=img_size, is_cached=is_cached, transform=None)

    train_loader = DataLoaderIterator(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoaderIterator(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    return train_loader, val_loader, None
