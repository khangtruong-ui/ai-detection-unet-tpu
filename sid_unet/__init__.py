"""SID-UNet TPU / JAX: AI-generated synthetic image forensics detection and segmentation."""

__version__ = "0.1.0"

# Compatibility shims for JAX >= 0.11 with Flax < 0.12
try:
    import jax
    import jax.core

    if not hasattr(jax.core, "get_opaque_trace_state"):
        try:
            import jax.extend.core as jex_core

            if hasattr(jex_core, "get_opaque_trace_state"):
                jax.core.get_opaque_trace_state = jex_core.get_opaque_trace_state
        except ImportError:
            pass

    if not hasattr(jax, "device_put_replicated"):

        def _device_put_replicated(tree, devices=None):
            import numpy as np
            import jax.numpy as jnp

            devices = devices or jax.local_devices()
            mesh = jax.sharding.Mesh(np.array(devices), ("_device_put_sharded",))
            sharding = jax.NamedSharding(mesh, jax.P("_device_put_sharded"))

            def _rep(x):
                if isinstance(x, jax.Array):
                    return jax.device_put(jnp.stack([x] * len(devices)), sharding)
                return jax.device_put(np.stack([x] * len(devices)), sharding)

            return jax.tree_util.tree_map(_rep, tree)

        jax.device_put_replicated = _device_put_replicated
except ImportError:
    pass
