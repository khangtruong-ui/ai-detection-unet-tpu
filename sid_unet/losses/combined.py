"""Combined Mask Loss combining BCE, Dice, and Focal Loss in JAX."""

from __future__ import annotations

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp

from sid_unet.losses.bce import BCELoss, binary_cross_entropy_with_logits
from sid_unet.losses.dice import DiceLoss, dice_loss
from sid_unet.losses.focal import FocalLoss, focal_loss


class CombinedMaskLoss:
    """Weighted combination of BCE Loss, Soft Dice Loss, and Focal Loss."""

    def __init__(
        self,
        loss_type: str = "combined",
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        pos_weight: float = 1.0,
    ):
        self.loss_type = loss_type.lower()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.pos_weight = float(pos_weight)

        self.bce = BCELoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def __call__(
        self,
        logits: jax.Array,
        targets: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Compute mask loss and loss component metrics."""
        metrics: Dict[str, jax.Array] = {}

        if self.loss_type == "bce":
            loss = self.bce(logits, targets)
            metrics["bce_loss"] = loss
            return loss, metrics

        if self.loss_type == "dice":
            loss = self.dice(logits, targets)
            metrics["dice_loss"] = loss
            return loss, metrics

        if self.loss_type == "focal":
            loss = self.focal(logits, targets)
            metrics["focal_loss"] = loss
            return loss, metrics

        # Combined loss
        b_loss = self.bce(logits, targets)
        d_loss = self.dice(logits, targets)
        f_loss = self.focal(logits, targets)

        total_loss = (
            self.bce_weight * b_loss
            + self.dice_weight * d_loss
            + self.focal_weight * f_loss
        )

        metrics["bce_loss"] = b_loss
        metrics["dice_loss"] = d_loss
        metrics["focal_loss"] = f_loss
        metrics["total_mask_loss"] = total_loss

        return total_loss, metrics
