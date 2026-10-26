"""Tests for PyTorch state dict checkpoint compatibility with ai-detection-unet-gpu."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import torch
import jax
import jax.numpy as jnp

from sid_unet.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_auto_resume_checkpoint,
    is_hf_repo_id,
    parse_hf_repo_uri,
    upload_hf_checkpoint,
    download_hf_checkpoint,
)
from sid_unet.utils.config import ConfigDict
from sid_unet.utils.distributed import is_tpu
from sid_unet.training.trainer import Trainer
from sid_unet.training.callbacks import CheckpointManager
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


@pytest.mark.timeout(120)
@pytest.mark.skipif(is_tpu(), reason="Skip PyTorch GPU checkpoint verification on TPU mode")
def test_load_ai_detection_unet_gpu_checkpoint(tmp_path):
    """Verify loading checkpoints originating from the khangtruong-ui/ai-detection-unet-gpu repository.

    This test verifies that PyTorch state dict checkpoints saved by the GPU repo
    (e.g., DiffusionDiffMinimized with auxiliary classifier head and decoder stages)
    can be properly loaded into JAX/Flax parameter trees with correct shapes, metadata,
    and parameter naming mappings.
    Skipped on TPU mode as required.
    """
    gpu_repo_dir = Path("/tmp/ai-detection-unet-gpu")
    if not gpu_repo_dir.exists() or not (gpu_repo_dir / "sid_unet").exists():
        # Clone repo if not cached
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/khangtruong-ui/ai-detection-unet-gpu.git", str(gpu_repo_dir)],
            check=True,
            capture_output=True,
            timeout=60,
        )

    ckpt_path = tmp_path / "checkpoint_from_gpu_repo.pt"

    # Generate checkpoint in isolated process using ai-detection-unet-gpu model
    generator_script = f"""
import sys
sys.path.insert(0, "{gpu_repo_dir}")
from sid_unet.models.diffusion_diff_minimized import DiffusionDiffMinimized
import torch

m = DiffusionDiffMinimized(use_dummy=True, decoder_channels=[32, 16], aux_classifier=True)
ckpt = {{
    "epoch": 4,
    "step": 320,
    "model_state_dict": m.state_dict(),
    "metrics": {{"val_loss": 0.185, "val_iou": 0.792}},
    "best_score": 0.792,
    "best_epoch": 4,
}}
torch.save(ckpt, "{ckpt_path}")
"""
    result = subprocess.run(
        [sys.executable, "-c", generator_script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"Failed generating GPU checkpoint: {result.stderr}"
    assert ckpt_path.exists()

    cfg = DiffusionDiffConfig(decoder_channels=[32, 16], aux_classifier=True)
    restored = load_checkpoint(ckpt_path, config=cfg)

    # Verify restored metadata
    assert restored["epoch"] == 4
    assert restored["step"] == 320
    assert restored["best_score"] == pytest.approx(0.792)
    assert restored["best_epoch"] == 4
    assert "val_iou" in restored["metrics"]
    assert restored["metrics"]["val_iou"] == pytest.approx(0.792)

    # Verify restored Flax parameters
    assert "params" in restored
    params = restored["params"]
    assert "z_norm" in params
    assert "decoder" in params

    # Verify z_norm parameters
    assert "scale" in params["z_norm"]
    assert params["z_norm"]["scale"].shape == (84,)

    # Verify decoder parameters
    assert "init_conv_0" in params["decoder"]
    assert "out_conv" in params["decoder"]


@pytest.mark.timeout(30)
def test_checkpoint_period_parsing():
    """Verify Trainer checkpoint_period string and numeric parsing."""
    assert Trainer._parse_period(None) == 0.0
    assert Trainer._parse_period("") == 0.0
    assert Trainer._parse_period(False) == 0.0
    assert Trainer._parse_period(180) == 180.0
    assert Trainer._parse_period(180.5) == 180.5
    assert Trainer._parse_period("30s") == 30.0
    assert Trainer._parse_period("2m") == 120.0
    assert Trainer._parse_period("1h") == 3600.0
    assert Trainer._parse_period("1.5h") == 5400.0
    assert Trainer._parse_period("non-numeric") == 0.0


@pytest.mark.timeout(30)
def test_checkpoint_manager_periodic_save(tmp_path):
    """Verify CheckpointManager saves checkpoint_periodic.pt and updates checkpoint_latest.pt."""
    mgr = CheckpointManager(
        checkpoint_dir=str(tmp_path),
        save_best=True,
        save_latest=True,
        mode="max",
    )

    dummy_params = {"dense": {"kernel": np.ones((4, 4), dtype=np.float32)}}
    periodic_path = mgr.save_periodic(
        epoch=2,
        step=50,
        params=dummy_params,
        metrics={"val_iou": 0.75},
        config={"decoder_channels": [32, 16]},
    )

    assert os.path.exists(periodic_path)
    assert "checkpoint_periodic.pt" in periodic_path

    latest_path = tmp_path / "checkpoint_latest.pt"
    assert latest_path.exists()

    # Load back using load_checkpoint
    restored = load_checkpoint(periodic_path)
    assert restored["epoch"] == 2
    assert restored["step"] == 50
    assert restored["metrics"]["val_iou"] == 0.75


@pytest.mark.timeout(30)
def test_hf_repo_identification_and_parsing():
    """Verify Hugging Face repo ID parsing and identification helpers."""
    assert is_hf_repo_id("KhangTruong/diffusion-diff-minimized-512") is True
    assert is_hf_repo_id("hf://KhangTruong/diffusion-diff-minimized-512") is True
    assert is_hf_repo_id("https://huggingface.co/KhangTruong/diffusion-diff-minimized-512") is True
    assert is_hf_repo_id("") is False
    assert is_hf_repo_id(None) is False

    repo_id, fname = parse_hf_repo_uri("KhangTruong/diffusion-diff-minimized-512:checkpoint_latest.pt")
    assert repo_id == "KhangTruong/diffusion-diff-minimized-512"
    assert fname == "checkpoint_latest.pt"

    repo_id, fname = parse_hf_repo_uri("hf://KhangTruong/diffusion-diff-minimized-512/checkpoints/checkpoint_best.pt")
    assert repo_id == "KhangTruong/diffusion-diff-minimized-512"
    assert fname == "checkpoints/checkpoint_best.pt"


@pytest.mark.timeout(30)
def test_hf_checkpoint_upload_and_download_mocked(tmp_path):
    """Verify Hugging Face checkpoint upload and download utilities with mocked HfApi."""
    ckpt_file = tmp_path / "test_ckpt.pt"
    ckpt_file.write_text("dummy checkpoint content")

    with patch("huggingface_hub.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.upload_file.return_value = "https://huggingface.co/KhangTruong/test/blob/main/checkpoints/test_ckpt.pt"

        res = upload_hf_checkpoint(
            file_path=ckpt_file,
            repo_id="KhangTruong/diffusion-diff-minimized-512",
        )
        assert res is not None
        assert "huggingface.co" in res
        mock_api.upload_file.assert_called_once()

    with patch("huggingface_hub.hf_hub_download") as mock_download:
        mock_download.return_value = str(ckpt_file)

        downloaded = download_hf_checkpoint(
            repo_id_or_uri="KhangTruong/diffusion-diff-minimized-512",
            filename="checkpoint_latest.pt",
        )
        assert downloaded == str(ckpt_file)
        mock_download.assert_called_once()


@pytest.mark.timeout(30)
def test_periodic_checkpoint_timer_starts_after_first_batch(tmp_path):
    """Verify Trainer initializes last_periodic_ckpt_time to None and starts timer only after batch 1."""
    from sid_unet.utils.config import load_config
    out_dir = tmp_path / "periodic_test"
    cfg = load_config(
        "configs/test_smoke.yaml",
        overrides=[
            f"project.output_dir={out_dir}",
            "training.checkpoint_period=10s",
            "data.dataset_name=mock",
        ],
    )
    trainer = Trainer(cfg)
    # Timer MUST NOT start at __init__ before first batch finishes
    assert trainer.checkpoint_period == 10.0
    assert trainer.last_periodic_ckpt_time is None


@pytest.mark.timeout(30)
def test_cli_hf_repo_flag_parsing():
    """Verify sid-train parse_args handles --hf-repo and --hub-repo flags."""
    from sid_unet.train import parse_args
    args = parse_args(["--config", "configs/test_smoke.yaml", "--hf-repo", "KhangTruong/diffusion-diff-minimized-512-fpn"])
    assert args.hf_repo == "KhangTruong/diffusion-diff-minimized-512-fpn"

    args2 = parse_args(["--config", "configs/test_smoke.yaml", "--hub-repo", "KhangTruong/test-model"])
    assert args2.hf_repo == "KhangTruong/test-model"


