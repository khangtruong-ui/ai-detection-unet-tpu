"""Unit tests for GPU and TPU scaling, detection, and batch sharding utilities."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from sid_unet.utils.distributed import (
    get_backend,
    get_device_count,
    get_device_info,
    is_gpu,
    is_tpu,
    is_multi_device,
    shard_batch,
    unshard_batch,
    replicate_params,
    unreplicate_params,
)


@pytest.mark.timeout(60)
def test_device_info_structure():
    info = get_device_info()
    assert "backend" in info
    assert "device_count" in info
    assert "is_tpu" in info
    assert "is_gpu" in info
    assert info["device_count"] >= 1


@pytest.mark.timeout(60)
def test_shard_and_unshard_batch():
    batch = {
        "x": np.ones((8, 16, 16, 3), dtype=np.float32),
        "y": np.zeros((8, 1), dtype=np.int32),
    }

    # Shard across 2 devices
    sharded = shard_batch(batch, num_devices=2)
    assert sharded["x"].shape == (2, 4, 16, 16, 3)
    assert sharded["y"].shape == (2, 4, 1)

    # Recombine
    unsharded = unshard_batch(sharded)
    assert unsharded["x"].shape == (8, 16, 16, 3)
    assert unsharded["y"].shape == (8, 1)


@pytest.mark.timeout(60)
def test_replicate_and_unreplicate_params():
    tree = {"w": jnp.array([1.0, 2.0, 3.0])}
    rep = replicate_params(tree)
    unrep = unreplicate_params(rep)
    np.testing.assert_allclose(np.asarray(unrep["w"]), np.asarray(tree["w"]))
