"""Flax Linen implementation of Diffusion-Diff-V2 with Perpendicular Skip Connections."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp

from jax_model_zoo.core.config import DiffusionDiffConfig
from jax_model_zoo.models.diffusion_diff.diffusion_diff_v2 import (
    DiffusionDiffV2 as JMZDiffusionDiffV2,
    TrainableLatentDecoderV2,
)


class DiffusionDiffV2(JMZDiffusionDiffV2):
    """Flax Linen DiffusionDiffV2 model with cache tensor extraction support."""

    @property
    def total_channels(self) -> int:
        return self.total_z_channels

    def extract_cache_tensors(
        self,
        x: jax.Array,
        rng: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Extract high-dimensional representation Z from input images for offline caching."""
        b, h, w, c = x.shape
        posterior = self.vae.encode(x)
        z0 = posterior.mode() * self.config.scaling_factor
        z0 = jnp.clip(z0, -15.0, 15.0)
        _, h_z, w_z, _ = z0.shape

        z_components: List[jax.Array] = []
        if self.config.include_z0:
            z_components.append(z0)

        curr_rng = rng if rng is not None else jax.random.PRNGKey(42)

        from jax_model_zoo.models.diffusion_diff.blocks import expand_to_spatial, sinusoidal_embedding

        for t_val in self.config.timesteps:
            t_clamped = max(0, min(t_val, len(self.alphas_cumprod) - 1))
            alpha_bar = self.alphas_cumprod[t_clamped]
            sigma_val = jnp.sqrt(jnp.maximum(1.0 - alpha_bar, 1e-8))
            sqrt_alpha_bar = jnp.sqrt(alpha_bar)

            curr_rng, subkey = jax.random.split(curr_rng)
            eps_k = jax.random.normal(subkey, z0.shape, dtype=z0.dtype)
            z_tk = sqrt_alpha_bar.astype(z0.dtype) * z0 + sigma_val.astype(z0.dtype) * eps_k

            t_tensor = jnp.full((b,), t_clamped, dtype=jnp.int32)
            cross_dim = self.diffuser.config.cross_attention_dim
            uncond_emb = jnp.zeros((b, 1, cross_dim), dtype=z_tk.dtype)

            pred_eps = self.diffuser(z_tk, t_tensor, encoder_hidden_states=uncond_emb)

            t_emb = sinusoidal_embedding(t_tensor.astype(jnp.float32), dim=self.config.timestep_embed_dim)
            t_spatial = expand_to_spatial(t_emb, h_z, w_z).astype(z0.dtype)

            sigma_tensor = jnp.broadcast_to(sigma_val[None], (b,))
            sigma_emb = sinusoidal_embedding(sigma_tensor, dim=self.config.sigma_embed_dim)
            sigma_spatial = expand_to_spatial(sigma_emb, h_z, w_z).astype(z0.dtype)

            if self.config.include_noisy_latents:
                z_components.append(z_tk)
            if self.config.include_added_noise:
                z_components.append(eps_k)
            if self.config.include_predicted_noise:
                z_components.append(pred_eps)
            if self.config.include_noise_diff:
                z_components.append(pred_eps - eps_k)
            if self.config.timestep_embed_dim > 0:
                z_components.append(t_spatial)
            if self.config.sigma_embed_dim > 0:
                z_components.append(sigma_spatial)

        return jnp.concatenate(z_components, axis=-1)


# Aliases matching ai-detection-unet-gpu
DiffusionDiffV2Model = DiffusionDiffV2
