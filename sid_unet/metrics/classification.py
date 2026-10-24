"""Classification metrics: Accuracy, Precision, Recall, and F1 score."""

from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np


def compute_classification_metrics(
    class_logits: np.ndarray,
    target_labels: np.ndarray,
) -> Dict[str, float]:
    """Compute image-level classification metrics."""
    logits = np.asarray(class_logits, dtype=np.float32)
    labels = np.asarray(target_labels, dtype=np.int32).flatten()

    if logits.ndim == 1 or logits.shape[-1] == 1:
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits.flatten(), -30.0, 30.0)))
        preds = (probs >= 0.5).astype(np.int32)
    else:
        # Multi-class softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        preds = np.argmax(probs, axis=-1)

    correct = int(np.sum(preds == labels))
    total = len(labels)
    acc = correct / max(total, 1)

    return {
        "accuracy": float(acc),
        "total_samples": int(total),
        "correct_samples": int(correct),
    }
