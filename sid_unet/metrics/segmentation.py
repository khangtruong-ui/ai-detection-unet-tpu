"""Segmentation metrics: IoU, Dice / F1, Accuracy, Precision, Recall, and AUROC."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


def compute_binary_auroc(
    pred_probs: np.ndarray,
    target_mask: np.ndarray,
) -> float:
    """Compute continuous pixel AUROC curve area."""
    p = np.asarray(pred_probs, dtype=np.float64).flatten()
    t = (np.asarray(target_mask, dtype=np.float64) >= 0.5).flatten().astype(np.int32)

    n_pos = int(np.sum(t))
    n_neg = len(t) - n_pos

    if n_pos == 0:
        mean_p = float(np.mean(p))
        return 1.0 if mean_p < 0.5 else max(0.0, 1.0 - mean_p)

    if n_neg == 0:
        mean_p = float(np.mean(p))
        return 1.0 if mean_p >= 0.5 else max(0.0, mean_p)

    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(t, p))
    except Exception:
        return 0.5


def compute_segmentation_metrics(
    pred_probs: np.ndarray,
    target_mask: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive segmentation evaluation metrics for a single sample or batch."""
    p = np.asarray(pred_probs, dtype=np.float32)
    t = np.asarray(target_mask, dtype=np.float32)

    # Sigmoid if raw logits
    if np.any(p < 0.0) or np.any(p > 1.0):
        p = 1.0 / (1.0 + np.exp(-np.clip(p, -30.0, 30.0)))

    pred_bin = (p >= threshold).astype(np.int32)
    target_bin = (t >= 0.5).astype(np.int32)

    tp = float(np.sum((pred_bin == 1) & (target_bin == 1)))
    fp = float(np.sum((pred_bin == 1) & (target_bin == 0)))
    fn = float(np.sum((pred_bin == 0) & (target_bin == 1)))
    tn = float(np.sum((pred_bin == 0) & (target_bin == 0)))

    total = tp + fp + fn + tn
    smooth = 1e-7

    iou = (tp + smooth) / (tp + fp + fn + smooth)
    f1 = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    acc = (tp + tn) / max(total, 1.0)
    prec = (tp + smooth) / (tp + fp + smooth)
    rec = (tp + smooth) / (tp + fn + smooth)
    spec = (tn + smooth) / (tn + fp + smooth)

    return {
        "iou": float(iou),
        "f1": float(f1),
        "dice": float(f1),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
    }


class RunningSegmentationMetrics:
    """Accumulates confusion matrix totals across batches for dataset-level metrics."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.count = 0

    def update(self, pred_probs: np.ndarray, target_mask: np.ndarray):
        p = np.asarray(pred_probs, dtype=np.float32)
        t = np.asarray(target_mask, dtype=np.float32)

        if np.any(p < 0.0) or np.any(p > 1.0):
            p = 1.0 / (1.0 + np.exp(-np.clip(p, -30.0, 30.0)))

        pred_bin = (p >= self.threshold).astype(np.int32)
        target_bin = (t >= 0.5).astype(np.int32)

        self.tp += float(np.sum((pred_bin == 1) & (target_bin == 1)))
        self.fp += float(np.sum((pred_bin == 1) & (target_bin == 0)))
        self.fn += float(np.sum((pred_bin == 0) & (target_bin == 1)))
        self.tn += float(np.sum((pred_bin == 0) & (target_bin == 0)))
        self.count += 1

    def update_counts(
        self,
        tp: float | int,
        fp: float | int,
        fn: float | int,
        tn: float | int = 0.0,
    ) -> None:
        """Update confusion matrix statistics directly from scalar counts."""
        self.tp += float(tp)
        self.fp += float(fp)
        self.fn += float(fn)
        self.tn += float(tn)
        self.count += 1

    def compute(self) -> Dict[str, float]:
        smooth = 1e-7
        total = self.tp + self.fp + self.fn + self.tn
        iou = (self.tp + smooth) / (self.tp + self.fp + self.fn + smooth)
        f1 = (2.0 * self.tp + smooth) / (2.0 * self.tp + self.fp + self.fn + smooth)
        acc = (self.tp + self.tn) / max(total, 1.0)
        prec = (self.tp + smooth) / (self.tp + self.fp + smooth)
        rec = (self.tp + smooth) / (self.tp + self.fn + smooth)
        spec = (self.tn + smooth) / (self.tn + self.fp + smooth)

        return {
            "iou": float(iou),
            "f1": float(f1),
            "dice": float(f1),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(spec),
        }
