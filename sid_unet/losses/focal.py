"""Binary Focal Loss in JAX for hard-example mining in segmentation masks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def focal_loss(
    logits: jax.Array,
    targets: jax.Array,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> jax.Array:
    """Binary Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""
    logits = jnp.clip(logits, -30.0, 30.0)
    targets = targets.astype(logits.dtype)

    bce = optax.sigmoid_binary_cross_entropy(logits, targets)
    probs = jax.nn.sigmoid(logits)

    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    p_t = jnp.clip(p_t, 1e-7, 1.0 - 1e-7)

    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal_weight = alpha_t * jnp.power(jnp.clip(1.0 - p_t, 0.0, 1.0), gamma)

    return jnp.mean(focal_weight * bce)


class FocalLoss:
    """FocalLoss callable object matching PyTorch API."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def __call__(self, logits: jax.Array, targets: jax.Array) -> jax.Array:
        return focal_loss(logits, targets, alpha=self.alpha, gamma=self.gamma)
