"""Training callbacks: CheckpointManager and EarlyStopping."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from sid_unet.utils.checkpoint import save_checkpoint

logger = logging.getLogger("sid_unet.training.callbacks")


class EarlyStopping:
    """Early stopping monitor based on validation metrics."""

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4):
        self.patience = max(1, patience)
        self.mode = mode.lower()
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > (self.best_score + self.min_delta)
        return score < (self.best_score - self.min_delta)

    def step(self, score: float) -> bool:
        if self.is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"🛑 Early stopping triggered after {self.counter} epochs without improvement.")
        return self.should_stop


class CheckpointManager:
    """Manages periodic, best, and latest checkpoint saves in PyTorch-compatible format."""

    def __init__(
        self,
        checkpoint_dir: str,
        metric_name: str = "val_iou",
        mode: str = "max",
        save_best: bool = True,
        save_latest: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.metric_name = metric_name
        self.mode = mode.lower()
        self.save_best = save_best
        self.save_latest = save_latest

        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def save(
        self,
        epoch: int,
        step: int,
        params: Any,
        metrics: Dict[str, float],
        config: Any,
        batch_stats: Optional[Any] = None,
        model_type: str = "diffusion_diff",
    ) -> Dict[str, str]:
        saved_paths = {}
        current_score = metrics.get(self.metric_name)

        is_best = False
        if current_score is not None and self.is_better(current_score):
            self.best_score = current_score
            self.best_epoch = epoch
            is_best = True

        if self.save_latest:
            latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
            save_checkpoint(
                params=params,
                path=latest_path,
                epoch=epoch,
                step=step,
                metrics=metrics,
                best_score=self.best_score,
                best_epoch=self.best_epoch,
                config=config,
                batch_stats=batch_stats,
                model_type=model_type,
            )
            saved_paths["latest"] = latest_path

        if is_best and self.save_best:
            best_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
            save_checkpoint(
                params=params,
                path=best_path,
                epoch=epoch,
                step=step,
                metrics=metrics,
                best_score=self.best_score,
                best_epoch=self.best_epoch,
                config=config,
                batch_stats=batch_stats,
                model_type=model_type,
            )
            saved_paths["best"] = best_path
            logger.info(f"🌟 Best model updated! {self.metric_name}: {current_score:.4f} at epoch {epoch}")

        return saved_paths
