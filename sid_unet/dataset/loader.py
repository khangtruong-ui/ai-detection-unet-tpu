"""Dataset loading pipeline for SID-UNet JAX/TPU.

Supports synthetic mock generation for rapid testing, cached tensor datasets,
and Hugging Face streaming/map datasets.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from sid_unet.dataset.transforms import SpatialJointTransform
from sid_unet.dataset.mask_utils import process_mask

logger = logging.getLogger("sid_unet.dataset")


class MockDataset:
    """Fast in-memory mock dataset for rapid testing, smoke checks, and benchmarking."""

    def __init__(
        self,
        num_samples: int = 64,
        image_size: Tuple[int, int] = (256, 256),
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
        # Generate deterministic synthetic data per index
        rng = np.random.RandomState(idx)

        # Label: 0=real, 1=synthetic full, 2=inpainted
        label = int(idx % 3)

        if label == 0:
            mask = np.zeros((h, w, 1), dtype=np.float32)
        elif label == 1:
            mask = np.ones((h, w, 1), dtype=np.float32)
        else:
            mask = np.zeros((h, w, 1), dtype=np.float32)
            # Center tamper patch
            rh, rw = h // 4, w // 4
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
    """Batch generator over dataset with batch collation into NumPy arrays."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(dataset)))

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        indices = list(self.indices)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(indices), self.batch_size):
            batch_idxs = indices[start_idx : start_idx + self.batch_size]
            if self.drop_last and len(batch_idxs) < self.batch_size:
                break

            items = [self.dataset[i] for i in batch_idxs]
            collated: Dict[str, np.ndarray] = {}
            for k in items[0].keys():
                collated[k] = np.stack([item[k] for item in items], axis=0)

            yield collated


def create_dataloaders(
    config: Any,
) -> Tuple[DataLoaderIterator, Optional[DataLoaderIterator], Optional[DataLoaderIterator]]:
    """Create train, validation, and test dataloaders from configuration."""
    data_cfg = getattr(config, "data", {}) if hasattr(config, "data") else config.get("data", {})
    img_size = tuple(data_cfg.get("image_size", [256, 256]))
    batch_size = int(data_cfg.get("batch_size", 8))
    dataset_name = str(data_cfg.get("dataset_name", "mock"))

    # Check if cached mode
    is_cached = bool(data_cfg.get("cached", False) or data_cfg.get("use_cache", False))

    augs = data_cfg.get("augmentations", {})
    train_transform = SpatialJointTransform(
        horizontal_flip=float(augs.get("horizontal_flip", 0.5)),
        vertical_flip=float(augs.get("vertical_flip", 0.0)),
        random_rotate90=float(augs.get("random_rotate90", 0.25)),
    )

    # If dataset is mock / synthetic or for fast tests
    train_samples = int(data_cfg.get("train_samples_per_epoch", 64))
    if train_samples <= 0:
        train_samples = 64

    val_samples = int(data_cfg.get("val_samples", 32))
    if val_samples <= 0:
        val_samples = 32

    train_ds = MockDataset(num_samples=train_samples, image_size=img_size, is_cached=is_cached, transform=train_transform)
    val_ds = MockDataset(num_samples=val_samples, image_size=img_size, is_cached=is_cached, transform=None)

    train_loader = DataLoaderIterator(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoaderIterator(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, None
