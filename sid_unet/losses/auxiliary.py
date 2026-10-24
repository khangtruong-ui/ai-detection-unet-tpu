"""Auxiliary classification loss for image-level AI vs Real detection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def auxiliary_loss(
    logits: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    """Cross-entropy loss for auxiliary image-level classification."""
    labels = labels.astype(jnp.int32).squeeze()
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))


class AuxiliaryLoss:
    """Auxiliary classification loss callable."""

    def __call__(self, logits: jax.Array, labels: jax.Array) -> jax.Array:
        return auxiliary_loss(logits, labels)
