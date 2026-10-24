"""Flax Linen implementation of Diffusion-Diff-Minimized.

Strictly adheres to 2 lines of computation:
1. Real image clean latent z0 (4 channels)
2. Single chosen noisy latent from the diffusion model (default t=250, 80 channels)
Total Z = 84 channels with parallel frozen VAE decoder perpendicular skips.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp

from jax_model_zoo.core.config import DiffusionDiffConfig
from jax_model_zoo.models.diffusion_diff.blocks import (
    expand_to_spatial,
    get_valid_group_norm_groups,
    sinusoidal_embedding,
)
from jax_model_zoo.models.diffusion_diff.diffusion_diff_minimized import (
    DiffusionDiffMinimized as JMZDiffusionDiffMinimized,
)


class DiffusionDiffMinimized(JMZDiffusionDiffMinimized):
    """DiffusionDiffMinimized model supporting cache extraction and fast cached forward."""

    @property
    def total_channels(self) -> int:
        return self.total_z_channels

    def extract_cache_tensors(
        self,
        x: jax.Array,
        rng: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Extract high-dimensional representation Z from input images for offline caching.

        Returns:
            Cached latent array Z of shape (B, H/8, W/8, total_z_channels).
        """
        b, h, w, c = x.shape
        posterior = self.vae.encode(x)
        z0 = posterior.mode() * self.config.scaling_factor
        z0 = jnp.clip(z0, -15.0, 15.0)
        _, h_z, w_z, _ = z0.shape

        z_components: List[jax.Array] = []
        if self.config.include_z0:
            z_components.append(z0)

        curr_rng = rng if rng is not None else jax.random.PRNGKey(42)

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

    @nn.compact
    def forward_cached(
        self,
        z_high_dim: jax.Array,
        target_size: Optional[Tuple[int, int]] = None,
        deterministic: bool = True,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Fast forward evaluation directly from pre-computed cached latent representation Z.

        Bypasses the frozen diffuser UNet, saving computation and memory.
        """
        b, h_z, w_z, _ = z_high_dim.shape

        # Extract z0 (first 4 channels) to generate parallel decoder perpendicular skips
        z0 = z_high_dim[..., :4]
        z0_for_decoder = z0 / jnp.array(self.config.scaling_factor, dtype=z0.dtype)
        _, parallel_decoder_features = self.vae.decode(z0_for_decoder, return_features=True)
        perp_skips = [p.astype(z_high_dim.dtype) for p in parallel_decoder_features[: len(self.config.decoder_channels) - 1]]

        z_groups = get_valid_group_norm_groups(z_high_dim.shape[-1])
        z_normed = nn.GroupNorm(num_groups=z_groups, name="z_norm")(z_high_dim)

        class_logits = None
        if self.config.aux_classifier:
            class_logits = self.classifier_head(z_normed, deterministic=deterministic)

        mask_logits = self.decoder(z_normed, perp_skips=perp_skips, deterministic=deterministic)

        if target_size is not None and mask_logits.shape[1:3] != target_size:
            h, w = target_size
            mask_logits = jax.image.resize(mask_logits, (b, h, w, mask_logits.shape[-1]), method="bilinear")

        if self.config.aux_classifier:
            return mask_logits, class_logits
        return mask_logits


# Aliases matching ai-detection-unet-gpu
DiffusionDiffMinimizedModel = DiffusionDiffMinimized
