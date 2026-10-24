"""Cached dataset loaders for high-dimensional forensics representations Z."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from sid_unet.dataset.transforms import SpatialJointTransform


class CachedShardDataset:
    """Dataset reading pre-extracted latent representation shards (.npz or .parquet)."""

    def __init__(
        self,
        shard_files: List[str],
        transform: Optional[SpatialJointTransform] = None,
    ):
        self.shard_files = sorted(shard_files)
        self.transform = transform
        self.samples: List[Dict[str, np.ndarray]] = []
        self._load_all_shards()

    def _load_all_shards(self):
        for f in self.shard_files:
            if f.endswith(".npz"):
                data = np.load(f)
                latents = data["latents"]
                masks = data["masks"]
                labels = data["labels"]
                for i in range(len(latents)):
                    self.samples.append({
                        "latent": latents[i],
                        "mask": masks[i],
                        "label": labels[i],
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        sample = self.samples[idx]
        latent = sample["latent"]
        mask = sample["mask"]
        label = sample["label"]

        if self.transform is not None:
            latent, mask = self.transform(latent, mask)

        return {"latent": latent, "mask": mask, "label": label}
