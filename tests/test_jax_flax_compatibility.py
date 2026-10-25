"""Unit tests for JAX and Flax tracer compatibility and replication shims."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

import sid_unet
from sid_unet.utils.distributed import replicate_params, unreplicate_params


class DummyForensicsBlock(nn.Module):
    channels: int = 16

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        h = nn.Conv(features=self.channels, kernel_size=(3, 3), padding="SAME")(x)
        h = nn.BatchNorm(use_running_average=not train)(h)
        h = nn.relu(h)
        return h


@pytest.mark.timeout(60)
def test_opaque_trace_state_shim_exists():
    """Verify that jax.core.get_opaque_trace_state is accessible for Flax tracers."""
    assert hasattr(jax.core, "get_opaque_trace_state")
    assert callable(getattr(jax.core, "get_opaque_trace_state"))


@pytest.mark.timeout(60)
def test_flax_linen_apply_tracing():
    """Verify that Linen model apply does not raise AttributeError with JAX >= 0.11."""
    model = DummyForensicsBlock(channels=8)
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((2, 16, 16, 3), dtype=jnp.float32)

    variables = model.init(rng, dummy_input, train=True)
    assert "params" in variables
    assert "batch_stats" in variables

    out, mutated = model.apply(
        variables,
        dummy_input,
        train=True,
        mutable=["batch_stats"],
    )
    assert out.shape == (2, 16, 16, 8)
    assert "batch_stats" in mutated


@pytest.mark.timeout(60)
def test_device_put_replicated_shim():
    """Verify replicate and unreplicate functionality with JAX >= 0.11 sharding."""
    params = {"kernel": jnp.ones((3, 3, 3, 8), dtype=jnp.float32)}
    replicated = replicate_params(params)
    unreplicated = unreplicate_params(replicated)

    np.testing.assert_allclose(
        np.asarray(unreplicated["kernel"]),
        np.asarray(params["kernel"]),
    )
