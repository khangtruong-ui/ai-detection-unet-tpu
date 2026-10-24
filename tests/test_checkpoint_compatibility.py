"""Tests for PyTorch state dict checkpoint compatibility with ai-detection-unet-gpu."""

import os
import pytest
import numpy as np
import torch
import jax
import jax.numpy as jnp

from sid_unet.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_auto_resume_checkpoint,
)
from sid_unet.utils.config import ConfigDict
from jax_model_zoo.core.config import DiffusionDiffConfig
from jax_model_zoo.converters.jax_to_torch import convert_jax_diffusion_diff_to_torch
from jax_model_zoo.converters.torch_to_jax import convert_torch_diffusion_diff_to_jax


@pytest.mark.timeout(60)
def test_diffusion_diff_pytorch_checkpoint_roundtrip(tmp_path):
    """Verify bidirectional PyTorch state_dict format conversion and save/load."""
    torch_state = {
        "z_norm.weight": torch.ones(84),
        "z_norm.bias": torch.zeros(84),
        "decoder.init_conv.0.weight": torch.randn(32, 84, 3, 3),
        "decoder.init_conv.1.weight": torch.ones(32),
        "decoder.init_conv.1.bias": torch.zeros(32),
        "decoder.stages.0.up.1.weight": torch.randn(16, 32, 3, 3),
        "decoder.stages.0.norm.weight": torch.ones(16),
        "decoder.stages.0.norm.bias": torch.zeros(16),
        "decoder.out_conv.weight": torch.randn(1, 16, 3, 3),
        "decoder.out_conv.bias": torch.zeros(1),
        "classifier_head.fc1.weight": torch.randn(128, 84),
        "classifier_head.fc1.bias": torch.zeros(128),
        "classifier_head.fc2.weight": torch.randn(2, 128),
        "classifier_head.fc2.bias": torch.zeros(2),
    }

    cfg = DiffusionDiffConfig(decoder_channels=[32, 16], aux_classifier=True)
    jax_params = convert_torch_diffusion_diff_to_jax(torch_state, cfg)

    assert "params" in jax_params
    assert "decoder" in jax_params["params"]
    assert "z_norm" in jax_params["params"]

    # Save checkpoint using SID-UNet save_checkpoint
    ckpt_path = tmp_path / "checkpoint_test.pt"
    save_checkpoint(
        params=jax_params["params"],
        path=ckpt_path,
        epoch=3,
        step=150,
        metrics={"val_iou": 0.85},
        best_score=0.85,
        config=cfg,
        save_torch_format=True,
    )

    assert ckpt_path.exists()

    # Load back using PyTorch directly to inspect keys as ai-detection-unet-gpu would
    raw_loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in raw_loaded
    sd = raw_loaded["model_state_dict"]
    assert "z_norm.weight" in sd
    assert "decoder.init_conv.0.weight" in sd
    assert "decoder.out_conv.weight" in sd

    # Load back through SID-UNet load_checkpoint
    restored = load_checkpoint(ckpt_path, config=cfg)
    assert "params" in restored
    assert restored["epoch"] == 3
    assert restored["step"] == 150
    assert restored["best_score"] == 0.85


@pytest.mark.timeout(60)
def test_find_auto_resume_checkpoint(tmp_path):
    assert find_auto_resume_checkpoint(tmp_path) is None

    latest_file = tmp_path / "checkpoint_latest.pt"
    latest_file.touch()

    best_file = tmp_path / "checkpoint_best.pt"
    best_file.touch()

    # Should prioritize checkpoint_latest.pt
    found = find_auto_resume_checkpoint(tmp_path)
    assert found is not None
    assert "checkpoint_latest.pt" in found
