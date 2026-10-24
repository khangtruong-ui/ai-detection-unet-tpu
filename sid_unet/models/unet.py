"""Standard Flax Linen UNet model and unified build_model factory function."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp

from jax_model_zoo.core.config import DiffusionDiffConfig
from sid_unet.models.diffusion_diff import DiffusionDiff
from sid_unet.models.diffusion_diff_v2 import DiffusionDiffV2
from sid_unet.models.diffusion_diff_minimized import DiffusionDiffMinimized


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2 block."""

    out_channels: int
    mid_channels: Optional[int] = None

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        mid_ch = self.mid_channels or self.out_channels
        x = nn.Conv(mid_ch, kernel_size=(3, 3), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=deterministic)(x)
        x = nn.relu(x)

        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=deterministic)(x)
        x = nn.relu(x)
        return x


class UNet(nn.Module):
    """Standard 4-stage UNet architecture implemented in Flax Linen."""

    out_channels: int = 1
    features: Sequence[int] = (64, 128, 256, 512)
    bilinear: bool = True
    dropout: float = 0.0
    aux_classifier: bool = False
    num_classes: int = 2

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> Union[jax.Array, Tuple[jax.Array, Optional[jax.Array]]]:
        b, h, w, _ = x.shape

        # Encoder path
        skips = []
        curr = x
        for i, feat in enumerate(self.features):
            curr = DoubleConv(out_channels=feat, name=f"enc_{i}")(curr, deterministic=deterministic)
            skips.append(curr)
            if i < len(self.features) - 1:
                curr = nn.max_pool(curr, window_shape=(2, 2), strides=(2, 2))

        # Bottleneck
        bottleneck = curr

        # Optional classification head
        class_logits = None
        if self.aux_classifier:
            pooled = jnp.mean(bottleneck, axis=(1, 2))
            class_logits = nn.Dense(self.num_classes, name="class_head")(pooled)

        # Decoder path
        curr_dec = bottleneck
        rev_features = list(reversed(self.features[:-1]))
        rev_skips = list(reversed(skips[:-1]))

        for i, feat in enumerate(rev_features):
            curr_h, curr_w = curr_dec.shape[1], curr_dec.shape[2]
            curr_dec = jax.image.resize(curr_dec, (b, curr_h * 2, curr_w * 2, curr_dec.shape[-1]), method="bilinear")
            curr_dec = nn.Conv(feat, kernel_size=(3, 3), padding="SAME", name=f"dec_up_{i}")(curr_dec)

            skip = rev_skips[i]
            # Match spatial dimensions if necessary
            if curr_dec.shape[1:3] != skip.shape[1:3]:
                curr_dec = jax.image.resize(curr_dec, (b, skip.shape[1], skip.shape[2], curr_dec.shape[-1]), method="bilinear")

            curr_dec = jnp.concatenate([skip, curr_dec], axis=-1)
            curr_dec = DoubleConv(out_channels=feat, name=f"dec_{i}")(curr_dec, deterministic=deterministic)

        mask_logits = nn.Conv(self.out_channels, kernel_size=(1, 1), name="out_conv")(curr_dec)

        if mask_logits.shape[1:3] != (h, w):
            mask_logits = jax.image.resize(mask_logits, (b, h, w, mask_logits.shape[-1]), method="bilinear")

        if self.aux_classifier:
            return mask_logits, class_logits
        return mask_logits


def build_model(config: Any) -> nn.Module:
    """Build model module from configuration dictionary.

    Supports:
      - 'diffusion_diff'
      - 'diffusion_diff_v2'
      - 'diffusion_diff_minimized'
      - 'unet'
    """
    model_cfg = getattr(config, "model", {}) if hasattr(config, "model") else config.get("model", {})
    name = (getattr(model_cfg, "name", "") or model_cfg.get("name", "diffusion_diff_minimized")).lower().replace("-", "_")

    dec = model_cfg.get("decoder", {})
    dec_channels = dec.get("channels", [256, 128, 64, 32]) if hasattr(dec, "get") else [256, 128, 64, 32]
    dec_norm = dec.get("norm_layer", "batchnorm") if hasattr(dec, "get") else "batchnorm"
    dec_act = dec.get("activation", "silu") if hasattr(dec, "get") else "silu"
    dec_drop = float(dec.get("dropout", 0.1)) if hasattr(dec, "get") else 0.1
    dec_res = int(dec.get("num_res_blocks", 1)) if hasattr(dec, "get") else 1
    use_fpn = bool(model_cfg.get("use_fpn", False))
    fpn_channels = int(model_cfg.get("fpn_channels", 128))
    aux_classifier = bool(model_cfg.get("aux_classifier", False))
    num_classes = int(model_cfg.get("num_classes", 3))
    scaling_factor = float(model_cfg.get("scaling_factor", 0.18215))
    is_eff = bool(model_cfg.get("is_efficient_backbone", True))

    if name in ("diffusion_diff_minimized", "diffusiondiffminimized"):
        timesteps = model_cfg.get("timesteps", [250])
        ts_dim = int(model_cfg.get("timestep_embed_dim", 32))
        sig_dim = int(model_cfg.get("sigma_embed_dim", 32))

        jmz_cfg = DiffusionDiffConfig(
            version="minimized",
            timesteps=list(timesteps),
            timestep_embed_dim=ts_dim,
            sigma_embed_dim=sig_dim,
            include_z0=bool(model_cfg.get("include_z0", True)),
            include_noisy_latents=bool(model_cfg.get("include_noisy_latents", True)),
            include_added_noise=bool(model_cfg.get("include_added_noise", True)),
            include_predicted_noise=bool(model_cfg.get("include_predicted_noise", True)),
            include_noise_diff=bool(model_cfg.get("include_noise_diff", True)),
            scaling_factor=scaling_factor,
            use_skip_connections=bool(model_cfg.get("use_perpendicular_skips", True) or model_cfg.get("use_skip_connections", True)),
            decoder_channels=list(dec_channels),
            decoder_norm=dec_norm,
            decoder_act=dec_act,
            decoder_dropout=dec_drop,
            decoder_res_blocks=dec_res,
            use_fpn=use_fpn,
            fpn_channels=fpn_channels,
            aux_classifier=aux_classifier,
            num_classes=num_classes,
            is_efficient_backbone=is_eff,
        )
        return DiffusionDiffMinimized(config=jmz_cfg)

    elif name in ("diffusion_diff_v2", "diffusiondiffv2"):
        timesteps = model_cfg.get("timesteps", [100, 250, 500])
        ts_dim = int(model_cfg.get("timestep_embed_dim", 32))
        sig_dim = int(model_cfg.get("sigma_embed_dim", 32))

        jmz_cfg = DiffusionDiffConfig(
            version="v2",
            timesteps=list(timesteps),
            timestep_embed_dim=ts_dim,
            sigma_embed_dim=sig_dim,
            include_z0=bool(model_cfg.get("include_z0", True)),
            include_noisy_latents=bool(model_cfg.get("include_noisy_latents", True)),
            include_added_noise=bool(model_cfg.get("include_added_noise", True)),
            include_predicted_noise=bool(model_cfg.get("include_predicted_noise", True)),
            include_noise_diff=bool(model_cfg.get("include_noise_diff", True)),
            scaling_factor=scaling_factor,
            use_skip_connections=bool(model_cfg.get("use_perpendicular_skips", True) or model_cfg.get("use_skip_connections", True)),
            decoder_channels=list(dec_channels),
            decoder_norm=dec_norm,
            decoder_act=dec_act,
            decoder_dropout=dec_drop,
            decoder_res_blocks=dec_res,
            use_fpn=use_fpn,
            fpn_channels=fpn_channels,
            aux_classifier=aux_classifier,
            num_classes=num_classes,
            is_efficient_backbone=is_eff,
        )
        return DiffusionDiffV2(config=jmz_cfg)

    elif name in ("diffusion_diff", "diffusiondiff"):
        timesteps = model_cfg.get("timesteps", [100, 250, 500])
        ts_dim = int(model_cfg.get("timestep_embed_dim", 32))
        sig_dim = int(model_cfg.get("sigma_embed_dim", 32))

        jmz_cfg = DiffusionDiffConfig(
            version="v1",
            timesteps=list(timesteps),
            timestep_embed_dim=ts_dim,
            sigma_embed_dim=sig_dim,
            include_z0=bool(model_cfg.get("include_z0", True)),
            include_noisy_latents=bool(model_cfg.get("include_noisy_latents", True)),
            include_added_noise=bool(model_cfg.get("include_added_noise", True)),
            include_predicted_noise=bool(model_cfg.get("include_predicted_noise", True)),
            include_noise_diff=bool(model_cfg.get("include_noise_diff", True)),
            scaling_factor=scaling_factor,
            use_skip_connections=bool(model_cfg.get("use_skip_connections", True)),
            decoder_channels=list(dec_channels),
            decoder_norm=dec_norm,
            decoder_act=dec_act,
            decoder_dropout=dec_drop,
            decoder_res_blocks=dec_res,
            use_fpn=use_fpn,
            fpn_channels=fpn_channels,
            aux_classifier=aux_classifier,
            num_classes=num_classes,
            is_efficient_backbone=is_eff,
        )
        return DiffusionDiff(config=jmz_cfg)

    elif name in ("unet", "standard_unet"):
        features = model_cfg.get("features", [64, 128, 256, 512])
        out_channels = int(model_cfg.get("out_channels", 1))
        bilinear = bool(model_cfg.get("bilinear", True))
        dropout = float(model_cfg.get("dropout", 0.0))
        return UNet(
            out_channels=out_channels,
            features=features,
            bilinear=bilinear,
            dropout=dropout,
            aux_classifier=aux_classifier,
            num_classes=num_classes,
        )

    else:
        raise ValueError(f"Unknown model name '{name}'. Supported: diffusion_diff, diffusion_diff_v2, diffusion_diff_minimized, unet.")
