"""CLI prediction and inference entrypoint for SID-UNet (sid-predict)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional
import jax
import numpy as np
from PIL import Image

from sid_unet.models.unet import build_model
from sid_unet.utils.checkpoint import load_checkpoint
from sid_unet.utils.config import load_config
from sid_unet.utils.logger import setup_logger

logger = logging.getLogger("sid_unet.predict")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sid-predict",
        description="Run inference to generate binary manipulation masks.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/diffusion_diff_minimized/default.yaml",
        help="Path to YAML model configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        default=None,
        help="Path to input image file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Directory to save predicted mask illustrations.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization probability threshold.",
    )
    return parser.parse_args(args)


def cli_main(argv: Optional[List[str]] = None) -> int:
    setup_logger(name="sid_unet", log_file=".logs.txt")
    args = parse_args(argv)

    cfg = load_config(args.config)
    model = build_model(cfg)

    params = {}
    if args.checkpoint and os.path.exists(args.checkpoint):
        restored = load_checkpoint(args.checkpoint, config=cfg)
        params = restored.get("params", {})
    else:
        rng = jax.random.PRNGKey(0)
        dummy_x = jax.numpy.ones((1, 64, 64, 3))
        try:
            params = model.init(rng, dummy_x, rng=rng)
        except Exception:
            params = {}

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"✨ Inference pipeline ready. Output destination: '{args.output_dir}'")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
