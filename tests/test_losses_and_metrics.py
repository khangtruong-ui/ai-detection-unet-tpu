"""Unit tests for segmentation losses and metrics in JAX/NumPy."""

import pytest
import jax.numpy as jnp
import numpy as np

from sid_unet.losses.bce import BCELoss
from sid_unet.losses.dice import DiceLoss
from sid_unet.losses.focal import FocalLoss
from sid_unet.losses.auxiliary import AuxiliaryLoss
from sid_unet.losses.combined import CombinedMaskLoss
from sid_unet.metrics.segmentation import (
    compute_segmentation_metrics,
    compute_binary_auroc,
    RunningSegmentationMetrics,
)
from sid_unet.metrics.classification import compute_classification_metrics


@pytest.mark.timeout(60)
def test_bce_loss():
    bce = BCELoss(pos_weight=1.5)
    logits = jnp.array([[2.0, -2.0]])
    targets = jnp.array([[1.0, 0.0]])
    loss = bce(logits, targets)
    assert loss > 0.0
    assert not jnp.isnan(loss)


@pytest.mark.timeout(60)
def test_dice_loss():
    dice = DiceLoss()
    logits = jnp.array([[[[10.0]], [[-10.0]]]])
    targets = jnp.array([[[[1.0]], [[0.0]]]])
    loss = dice(logits, targets)
    # Perfect match: dice loss should be very close to 0.0
    assert loss < 0.1


@pytest.mark.timeout(60)
def test_focal_loss():
    focal = FocalLoss(alpha=0.5, gamma=2.0)
    logits = jnp.array([[[[0.0]], [[0.0]]]])
    targets = jnp.array([[[[1.0]], [[0.0]]]])
    loss = focal(logits, targets)
    assert loss > 0.0
    assert not jnp.isnan(loss)


@pytest.mark.timeout(60)
def test_combined_mask_loss():
    comb = CombinedMaskLoss(bce_weight=0.5, dice_weight=0.5, focal_weight=0.5)
    logits = jnp.zeros((2, 16, 16, 1))
    targets = jnp.ones((2, 16, 16, 1))

    total, metrics = comb(logits, targets)
    assert total > 0.0
    assert "bce_loss" in metrics
    assert "dice_loss" in metrics
    assert "focal_loss" in metrics


@pytest.mark.timeout(60)
def test_auxiliary_loss():
    aux = AuxiliaryLoss()
    logits = jnp.array([[2.0, -1.0, 0.5], [-1.0, 3.0, 0.0]])
    labels = jnp.array([0, 1])
    loss = aux(logits, labels)
    assert loss > 0.0


@pytest.mark.timeout(60)
def test_segmentation_metrics():
    preds = np.array([[[0.9, 0.1], [0.8, 0.2]]])
    targets = np.array([[[1.0, 0.0], [1.0, 0.0]]])

    metrics = compute_segmentation_metrics(preds, targets, threshold=0.5)
    assert metrics["iou"] > 0.9
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] > 0.9


@pytest.mark.timeout(60)
def test_running_segmentation_metrics():
    runner = RunningSegmentationMetrics(threshold=0.5)
    p1 = np.ones((4, 16, 16, 1), dtype=np.float32)
    t1 = np.ones((4, 16, 16, 1), dtype=np.float32)
    runner.update(p1, t1)

    p2 = np.zeros((4, 16, 16, 1), dtype=np.float32)
    t2 = np.zeros((4, 16, 16, 1), dtype=np.float32)
    runner.update(p2, t2)

    res = runner.compute()
    assert res["iou"] > 0.9
    assert res["accuracy"] == 1.0


@pytest.mark.timeout(60)
def test_classification_metrics():
    logits = np.array([[2.0, -1.0], [-1.0, 2.0], [3.0, 0.0]])
    targets = np.array([0, 1, 0])
    res = compute_classification_metrics(logits, targets)
    assert res["accuracy"] == 1.0
    assert res["total_samples"] == 3
