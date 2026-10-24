"""Binary Cross Entropy Loss in JAX/Flax."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def binary_cross_entropy_with_logits(
    logits: jax.Array,
    targets: jax.Array,
    pos_weight: float = 1.0,
) -> jax.Array:
    """Compute binary cross entropy loss with numerical stability."""
    logits = jnp.clip(logits, -30.0, 30.0)
    targets = targets.astype(logits.dtype)

    # Standard optax sigmoid binary cross entropy
    loss = optax.sigmoid_binary_cross_entropy(logits, targets)
    if pos_weight != 1.0:
        weight_mask = targets * (pos_weight - 1.0) + 1.0
        loss = loss * weight_mask

    return jnp.mean(loss)


class BCELoss:
    """BCELoss callable object matching PyTorch API."""

    def __init__(self, pos_weight: float = 1.0):
        self.pos_weight = float(pos_weight)

    def __call__(self, logits: jax.Array, targets: jax.Array) -> jax.Array:
        return binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
