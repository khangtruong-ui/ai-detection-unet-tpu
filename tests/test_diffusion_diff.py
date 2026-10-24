"""Unit tests for DiffusionDiff models family (DiffusionDiff, DiffusionDiffV2, DiffusionDiffMinimized, UNet)."""

import pytest
import jax
import jax.numpy as jnp

from sid_unet.models.unet import build_model, UNet
from sid_unet.models.diffusion_diff import DiffusionDiff
from sid_unet.models.diffusion_diff_v2 import DiffusionDiffV2
from sid_unet.models.diffusion_diff_minimized import DiffusionDiffMinimized
from sid_unet.utils.config import ConfigDict


@pytest.mark.timeout(60)
def test_unet_forward():
    model = UNet(out_channels=1, features=[8, 16, 32], aux_classifier=True, num_classes=3)
    x = jnp.ones((2, 64, 64, 3))
    rng = jax.random.PRNGKey(0)

    params = model.init(rng, x, deterministic=True)
    mask_logits, class_logits = model.apply(params, x, deterministic=True)

    assert mask_logits.shape == (2, 64, 64, 1)
    assert class_logits.shape == (2, 3)


@pytest.mark.timeout(60)
def test_diffusion_diff_minimized_build():
    cfg = ConfigDict({
        "model": {
            "name": "diffusion_diff_minimized",
            "timesteps": [250],
            "timestep_embed_dim": 16,
            "sigma_embed_dim": 16,
            "decoder": {"channels": [32, 16, 8]},
            "aux_classifier": True,
            "num_classes": 3,
            "is_efficient_backbone": True,
        }
    })
    model = build_model(cfg)
    assert isinstance(model, DiffusionDiffMinimized)
    assert model.total_channels == 52  # 4 + 4 + 4 + 4 + 4 + 16 + 16


@pytest.mark.timeout(60)
def test_diffusion_diff_v2_build():
    cfg = ConfigDict({
        "model": {
            "name": "diffusion_diff_v2",
            "timesteps": [100, 500],
            "timestep_embed_dim": 16,
            "sigma_embed_dim": 16,
            "decoder": {"channels": [32, 16, 8]},
            "aux_classifier": False,
            "is_efficient_backbone": True,
        }
    })
    model = build_model(cfg)
    assert isinstance(model, DiffusionDiffV2)


@pytest.mark.timeout(60)
def test_diffusion_diff_build():
    cfg = ConfigDict({
        "model": {
            "name": "diffusion_diff",
            "timesteps": [100, 500],
            "timestep_embed_dim": 16,
            "sigma_embed_dim": 16,
            "decoder": {"channels": [32, 16, 8]},
            "aux_classifier": False,
            "is_efficient_backbone": True,
        }
    })
    model = build_model(cfg)
    assert isinstance(model, DiffusionDiff)


@pytest.mark.timeout(60)
def test_diffusion_diff_forward_and_cache_extraction():
    cfg = ConfigDict({
        "model": {
            "name": "diffusion_diff_minimized",
            "timesteps": [250],
            "timestep_embed_dim": 16,
            "sigma_embed_dim": 16,
            "decoder": {"channels": [32, 16, 8]},
            "aux_classifier": True,
            "num_classes": 2,
            "is_efficient_backbone": True,
        }
    })
    model = build_model(cfg)
    x = jnp.ones((1, 64, 64, 3))
    rng = jax.random.PRNGKey(42)

    params = model.init(rng, x, rng=rng)
    mask_logits, class_logits = model.apply(params, x, rng=rng, deterministic=True)

    assert mask_logits.shape == (1, 64, 64, 1)
    assert class_logits.shape == (1, 2)

    # Test cache extraction
    z = model.apply(params, x, rng=rng, method=model.extract_cache_tensors)
    assert z.shape == (1, 8, 8, model.total_channels)

    # Test fast cached forward
    mask_cached, class_cached = model.apply(
        params, z, target_size=(64, 64), deterministic=True, method=model.forward_cached
    )
    assert mask_cached.shape == (1, 64, 64, 1)
    assert class_cached.shape == (1, 2)
