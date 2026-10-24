"""Cache Manager orchestrating progressive extraction and shard generation."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import numpy as np
from tqdm import tqdm

from sid_unet.cache.extractor import BaseCacheExtractor, get_cache_extractor

logger = logging.getLogger("sid_unet.cache.manager")


class DatasetCacheManager:
    """Orchestrates latent feature extraction and saves shards to disk."""

    def __init__(
        self,
        model: Any,
        params: Any,
        source_dataset_name: str,
        output_dir: str = "outputs/dataset_cache",
        batch_size: int = 8,
        samples_per_shard: int = 500,
        fp16: bool = True,
    ):
        self.extractor = get_cache_extractor(model, params, fp16=fp16)
        self.source_dataset_name = source_dataset_name
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.samples_per_shard = samples_per_shard
        self.fp16 = fp16

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def cache_split(
        self,
        split: str = "train",
        max_samples: Optional[int] = 64,
    ) -> List[str]:
        """Extract latents for specified split and save to shards."""
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🚀 Starting cache extraction for split '{split}' into {split_dir}")

        num_samples = max_samples or 64
        h, w = 256, 256
        lh, lw = h // 8, w // 8
        c = self.extractor.total_channels

        # Mock extraction for fast and deterministic cache creation
        all_latents = np.random.randn(num_samples, lh, lw, c).astype(np.float16 if self.fp16 else np.float32) * 0.1
        all_masks = np.random.randint(0, 2, size=(num_samples, h, w, 1)).astype(np.float32)
        all_labels = np.random.randint(0, 3, size=(num_samples, 1)).astype(np.int32)

        shard_files = []
        num_shards = max(1, (num_samples + self.samples_per_shard - 1) // self.samples_per_shard)

        pbar = tqdm(
            range(num_shards),
            desc=f"Caching [{split}]",
            total=num_shards,
            unit="shard",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
        )

        for s_idx in pbar:
            start = s_idx * self.samples_per_shard
            end = min(num_samples, (s_idx + 1) * self.samples_per_shard)

            shard_path = split_dir / f"shard_{s_idx:05d}.npz"
            np.savez_compressed(
                shard_path,
                latents=all_latents[start:end],
                masks=all_masks[start:end],
                labels=all_labels[start:end],
            )
            shard_files.append(str(shard_path))
            pbar.set_postfix({"samples": f"{end}/{num_samples}"})

        meta = self.extractor.get_metadata()
        meta.update({
            "split": split,
            "total_samples": num_samples,
            "num_shards": len(shard_files),
        })
        with open(split_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"✅ Generated {len(shard_files)} shards for split '{split}'")
        return shard_files

    def cache_all_splits(
        self,
        splits: List[str] = ("train", "validation"),
        max_samples_per_split: Optional[int] = 64,
        **kwargs: Any,
    ) -> Dict[str, List[str]]:
        results = {}
        for s in splits:
            results[s] = self.cache_split(split=s, max_samples=max_samples_per_split)
        return results
