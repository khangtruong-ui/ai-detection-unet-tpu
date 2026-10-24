"""CLI training entrypoint for SID-UNet JAX/TPU (sid-train).

Defaults to detaching from terminal mode and streaming execution logs to '.logs.txt'.
Supports interactive foreground mode via '--foreground' or '--no-detach'.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config
from sid_unet.utils.detach import maybe_detach_cli
from sid_unet.utils.logger import setup_logger
from sid_unet.utils.signals import shield_process_signals

logger = logging.getLogger("sid_unet.train")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sid-train",
        description="Train SID-UNet & Diffusion-Diff forensics segmentation models in JAX/TPU.",
    )
    parser.add_argument(
        "--config",
        "--configs",
        nargs="+",
        dest="config",
        default=["configs/experiments/diffusion_diff_minimized/default.yaml"],
        help="Path(s) to YAML configuration file(s).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Configuration overrides (key.nested=value format).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file or 'auto' to resume automatically.",
    )
    parser.add_argument(
        "--foreground",
        "--no-detach",
        action="store_true",
        dest="foreground",
        help="Execute synchronously in foreground without detaching from terminal session.",
    )
    return parser.parse_args(args)


def run_single_config(config_path: str, overrides: List[str], resume: Optional[str] = None):
    """Run training for a single YAML config."""
    cfg = load_config(config_path, overrides=overrides)
    if resume is not None:
        if not hasattr(cfg, "training") or cfg.training is None:
            cfg.training = {}
        cfg.training["resume"] = resume

    trainer = Trainer(cfg)
    return trainer.train()


def cli_main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint for sid-train."""
    # Step 1: Detach from controlling terminal by default unless foreground flag is present
    maybe_detach_cli(
        cli_name="sid-train",
        module_name="sid_unet.train",
        argv=argv,
    )

    # Step 2: In worker process: shield from SIGHUP / terminal drops & setup resilient logging
    shield_process_signals(detach_terminal=True, protect_streams=True)
    setup_logger(name="sid_unet", log_file=".logs.txt")
    logger.info("⚡ Starting sid-train session in JAX/TPU environment...")

    args = parse_args(argv)

    for cfg_path in args.config:
        logger.info(f"▶️ Launching training run for: {cfg_path}")
        run_single_config(cfg_path, overrides=args.override, resume=args.resume)

    logger.info("🏁 All training runs finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
