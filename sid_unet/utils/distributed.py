"""Device and distributed scaling utilities for JAX/Flax on GPU and TPU (v5e-1, v5e-8).

Handles topology detection, single vs multi-device scaling, parameter replication,
and batch sharding across TPU cores and GPUs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union
import jax
import jax.numpy as jnp
import numpy as np

try:
    from jax_model_zoo.core.distributed import (
        get_device_count as jmz_get_device_count,
        get_devices as jmz_get_devices,
        replicate as jmz_replicate,
        shard_batch as jmz_shard_batch,
        unreplicate as jmz_unreplicate,
        unshard_batch as jmz_unshard_batch,
    )
    _HAS_JMZ_DIST = True
except ImportError:
    _HAS_JMZ_DIST = False

logger = logging.getLogger("sid_unet.distributed")


def get_backend() -> str:
    """Get active JAX default backend ('tpu', 'gpu', or 'cpu')."""
    try:
        return jax.default_backend()
    except Exception:
        return "cpu"


def is_tpu() -> bool:
    """Check if execution environment is running on Google Cloud TPU."""
    return get_backend() == "tpu"


def is_gpu() -> bool:
    """Check if execution environment is running on NVIDIA GPU."""
    return get_backend() == "gpu"


def get_device_count() -> int:
    """Return number of available local devices/cores."""
    try:
        return jax.local_device_count()
    except Exception:
        return 1


def is_multi_device() -> bool:
    """Check if multiple GPU or TPU cores (such as TPU v5e-8) are available."""
    return get_device_count() > 1


def get_devices() -> Sequence[jax.Device]:
    """Get list of local JAX devices."""
    try:
        return jax.local_devices()
    except Exception:
        return []


def get_device_info() -> Dict[str, Any]:
    """Return dictionary summarizing current hardware acceleration setup."""
    backend = get_backend()
    dev_count = get_device_count()
    devices = get_devices()
    dev_kinds = [d.device_kind for d in devices] if devices else [backend]

    is_tpu_v5e = any("v5e" in k.lower() or "tpu v5" in k.lower() for k in dev_kinds)
    tpu_variant = None
    if is_tpu() or is_tpu_v5e:
        tpu_variant = f"v5e-{dev_count}" if dev_count in (1, 4, 8) else f"TPU ({dev_count} cores)"

    return {
        "backend": backend,
        "device_count": dev_count,
        "devices": [str(d) for d in devices],
        "device_kinds": dev_kinds,
        "is_tpu": is_tpu() or is_tpu_v5e,
        "is_gpu": is_gpu(),
        "is_multi_device": dev_count > 1,
        "tpu_variant": tpu_variant,
    }


def replicate_params(pytree: Any) -> Any:
    """Replicate parameter PyTree across all local devices for pmap data parallelism."""
    if _HAS_JMZ_DIST:
        return jmz_replicate(pytree)
    import flax.jax_utils as jutils
    return jutils.replicate(pytree)


def unreplicate_params(pytree: Any) -> Any:
    """Retrieve non-replicated copy of parameter PyTree from the first device."""
    if _HAS_JMZ_DIST:
        return jmz_unreplicate(pytree)
    import flax.jax_utils as jutils
    return jutils.unreplicate(pytree)


def shard_batch(
    batch: Union[jax.Array, np.ndarray, Dict[str, Any]],
    num_devices: Optional[int] = None,
) -> Any:
    """Shard batch across devices by leading dimension: [B, ...] -> [num_devices, B // num_devices, ...]."""
    if _HAS_JMZ_DIST and num_devices is None:
        return jmz_shard_batch(batch)

    n_dev = num_devices if num_devices is not None else get_device_count()
    if n_dev <= 1:
        return batch

    def _shard(x: Any) -> Any:
        if isinstance(x, (np.ndarray, jax.Array)):
            b_sz = x.shape[0]
            if b_sz % n_dev != 0:
                raise ValueError(f"Batch size {b_sz} must be divisible by device count {n_dev}")
            per_dev = b_sz // n_dev
            return x.reshape((n_dev, per_dev, *x.shape[1:]))
        return x

    return jax.tree_util.tree_map(_shard, batch)


def unshard_batch(sharded_batch: Any) -> Any:
    """Recombine sharded batch back to original dimension: [num_devices, per_dev, ...] -> [B, ...]."""
    if _HAS_JMZ_DIST:
        return jmz_unshard_batch(sharded_batch)

    def _unshard(x: Any) -> Any:
        if isinstance(x, (np.ndarray, jax.Array)) and x.ndim >= 2:
            return x.reshape((x.shape[0] * x.shape[1], *x.shape[2:]))
        return x

    return jax.tree_util.tree_map(_unshard, sharded_batch)
