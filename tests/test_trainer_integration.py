"""Integration tests for end-to-end Trainer execution, checkpointing, and resume."""

import os
import pytest
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config


@pytest.mark.timeout(120)
def test_trainer_smoke_run(tmp_path):
    out_dir = tmp_path / "smoke_run"
    cfg = load_config(
        "configs/test_smoke.yaml",
        overrides=[
            f"project.output_dir={out_dir}",
            "data.dataset_name=mock",
            "training.epochs=2",
            "data.train_samples_per_epoch=16",
            "data.val_samples=8",
            "data.batch_size=4",
        ],
    )

    trainer = Trainer(cfg)
    results = trainer.train()

    assert results["total_epochs"] == 2
    assert "best_val_iou" in results

    ckpt_dir = out_dir / "checkpoints"
    latest_ckpt = ckpt_dir / "checkpoint_latest.pt"
    assert latest_ckpt.exists()

    # Test resume from saved checkpoint
    cfg_resume = load_config(
        "configs/test_smoke.yaml",
        overrides=[
            f"project.output_dir={out_dir}",
            f"training.resume={latest_ckpt}",
            "data.dataset_name=mock",
            "training.epochs=3",
            "data.train_samples_per_epoch=16",
            "data.val_samples=8",
            "data.batch_size=4",
        ],
    )
    trainer_resume = Trainer(cfg_resume)
    assert trainer_resume.start_epoch == 3
    results_resume = trainer_resume.train()
    assert results_resume["total_epochs"] == 3


@pytest.mark.timeout(120)
def test_trainer_diffusion_diff_minimized_smoke_run(tmp_path):
    out_dir = tmp_path / "diff_smoke"
    cfg = load_config(
        "configs/test_smoke.yaml",
        overrides=[
            f"project.output_dir={out_dir}",
            "model.name=diffusion_diff_minimized",
            "model.decoder.channels=[16, 8]",
            "model.timestep_embed_dim=16",
            "model.sigma_embed_dim=16",
            "model.is_efficient_backbone=true",
            "data.cached=true",
            "data.batch_size=2",
            "data.image_size=[64, 64]",
            "data.train_samples_per_epoch=4",
            "data.val_samples=2",
            "training.epochs=1",
        ],
    )
    trainer = Trainer(cfg)
    results = trainer.train()
    assert results["total_epochs"] == 1
    assert "best_val_iou" in results
    assert trainer.state.batch_stats is not None
