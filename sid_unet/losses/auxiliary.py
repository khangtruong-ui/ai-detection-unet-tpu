"""Auxiliary classification loss for image-level AI vs Real detection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def auxiliary_loss(
    logits: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    """Cross-entropy loss for auxiliary image-level classification.

    Supports arbitrary batch dimensions including single-sample batches (batch size 1)
    and standard multi-sample batches (e.g. logits of shape ``(B, num_classes)`` with
    labels of shape ``(B,)`` or ``(B, 1)``).

    Args:
        logits: Unnormalized log probabilities with shape ``(*batch_dims, num_classes)``.
        labels: Integer class indices with shape matching or reshapable to ``batch_dims``.

    Returns:
        Scalar mean cross-entropy loss across all batch elements.
    """
    labels = labels.astype(jnp.int32)
    target_shape = logits.shape[:-1]
    if labels.shape != target_shape:
        labels = labels.reshape(target_shape)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))


class AuxiliaryLoss:
    """Auxiliary classification loss callable."""

    def __call__(self, logits: jax.Array, labels: jax.Array) -> jax.Array:
        return auxiliary_loss(logits, labels)
