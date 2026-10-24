"""CLI evaluation entrypoint for SID-UNet (sid-eval)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional
import numpy as np

from sid_unet.dataset.loader import create_dataloaders
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config
from sid_unet.utils.logger import setup_logger

logger = logging.getLogger("sid_unet.evaluate")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sid-eval",
        description="Evaluate SID-UNet model checkpoint on dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/diffusion_diff_minimized/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt or .safetensors).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Configuration overrides (key.nested=value format).",
    )
    return parser.parse_args(args)


def cli_main(argv: Optional[List[str]] = None) -> int:
    setup_logger(name="sid_unet", log_file=".logs.txt")
    args = parse_args(argv)

    cfg = load_config(args.config, overrides=args.override)
    if args.checkpoint:
        if not hasattr(cfg, "training") or cfg.training is None:
            cfg.training = {}
        cfg.training["resume"] = args.checkpoint

    trainer = Trainer(cfg)
    _, val_loader, _ = create_dataloaders(cfg)

    logger.info("🔍 Running evaluation on validation dataset...")
    metrics = trainer.validate(val_loader) if val_loader else {}

    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS:")
    for k, v in metrics.items():
        print(f"   • {k}: {v:.4f}")
    print("=" * 50 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
