"""CLI entrypoint for dataset tensor caching (sid-cache / sid-dataset-cache).

Defaults to detaching from terminal mode and logging output to '.logs.txt'.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional
import jax

from sid_unet.cache.manager import DatasetCacheManager
from sid_unet.models.unet import build_model
from sid_unet.utils.config import load_config
from sid_unet.utils.detach import maybe_detach_cli
from sid_unet.utils.logger import setup_logger

logger = logging.getLogger("sid_unet.cache")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sid-cache",
        description="Extract and cache high-dimensional diffusion forensics latents Z.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/diffusion_diff_minimized/default.yaml",
        help="Path to YAML model configuration file.",
    )
    parser.add_argument(
        "--dataset",
        "--dataset-name",
        type=str,
        default=None,
        dest="dataset_name",
        help="Source dataset name or path.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=str,
        default="outputs/dataset_cache",
        help="Local directory to store cached shards.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Splits to cache (e.g. train validation).",
    )
    parser.add_argument(
        "--max-samples",
        "--max_samples",
        type=int,
        default=64,
        help="Maximum samples to cache per split (for fast runs).",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Save cached tensors in fp16 precision (default: True).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Configuration overrides (key.nested=value).",
    )
    parser.add_argument(
        "--foreground",
        "--no-detach",
        action="store_true",
        dest="foreground",
        help="Execute synchronously in foreground without detaching from terminal session.",
    )
    return parser.parse_args(args)


def cli_main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint for sid-cache."""
    # Step 1: Handle default detachment from terminal mode into .logs.txt
    maybe_detach_cli(
        cli_name="sid-cache",
        module_name="sid_unet.cache.cli",
        argv=argv,
    )

    # Step 2: Initialize logger (logs directly to .logs.txt and console)
    setup_logger(name="sid_unet", log_file=".logs.txt")
    logger.info("⚡ Starting sid-cache session in JAX/TPU environment...")

    parsed_args = parse_args(argv)

    cfg = load_config(parsed_args.config, overrides=parsed_args.override)
    source_dataset = parsed_args.dataset_name or getattr(cfg.data, "dataset_name", "mock")

    logger.info(f"🏗️ Building model from config: {parsed_args.config}")
    model = build_model(cfg)

    # Dummy init for extractor parameters
    rng = jax.random.PRNGKey(0)
    init_x = jax.numpy.ones((1, 64, 64, 3))
    try:
        params = model.init(rng, init_x, rng=rng)
    except Exception:
        params = {}

    manager = DatasetCacheManager(
        model=model,
        params=params,
        source_dataset_name=source_dataset,
        output_dir=parsed_args.output_dir,
        batch_size=parsed_args.batch_size,
        fp16=parsed_args.fp16,
    )

    logger.info(
        f"🚀 Caching dataset: source='{source_dataset}', splits={parsed_args.splits}, "
        f"max_samples={parsed_args.max_samples}, output_dir='{parsed_args.output_dir}'"
    )

    all_shards = manager.cache_all_splits(
        splits=parsed_args.splits,
        max_samples_per_split=parsed_args.max_samples,
    )

    logger.info(f"🎉 Dataset caching finished! Total shards generated: {sum(len(v) for v in all_shards.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
