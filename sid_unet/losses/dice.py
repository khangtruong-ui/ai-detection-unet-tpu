"""Soft Dice Loss in JAX for binary mask segmentation."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def dice_loss(
    logits: jax.Array,
    targets: jax.Array,
    smooth: float = 1.0,
    eps: float = 1e-7,
) -> jax.Array:
    """Soft Dice Loss: 1.0 - (2 * intersection + smooth) / (cardinality + smooth + eps).

    Args:
        logits: Mask logits of shape (B, H, W, 1) or (B, H, W).
        targets: Ground truth masks of shape (B, H, W, 1) or (B, H, W).
    """
    probs = jax.nn.sigmoid(logits)
    targets = targets.astype(probs.dtype)

    # Flatten spatial and channel dimensions per batch item: (B, N)
    b_sz = probs.shape[0]
    p_flat = probs.reshape((b_sz, -1))
    t_flat = targets.reshape((b_sz, -1))

    intersection = jnp.sum(p_flat * t_flat, axis=-1)
    cardinality = jnp.sum(p_flat, axis=-1) + jnp.sum(t_flat, axis=-1)

    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth + eps)
    return jnp.mean(1.0 - dice_score)


class DiceLoss:
    """DiceLoss callable object matching PyTorch API."""

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        self.smooth = smooth
        self.eps = eps

    def __call__(self, logits: jax.Array, targets: jax.Array) -> jax.Array:
        return dice_loss(logits, targets, smooth=self.smooth, eps=self.eps)
