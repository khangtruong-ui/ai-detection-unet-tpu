"""Tests for configuration files, default schema validation, and model building."""

import glob
import os
import pytest
from pathlib import Path

from sid_unet.utils.config import DEFAULT_CONFIG, load_config, copy_config, deep_merge
from sid_unet.models.unet import build_model, UNet
from sid_unet.models.diffusion_diff import DiffusionDiff
from sid_unet.models.diffusion_diff_v2 import DiffusionDiffV2
from sid_unet.models.diffusion_diff_minimized import DiffusionDiffMinimized


@pytest.mark.timeout(60)
def test_default_config_schema():
    assert DEFAULT_CONFIG["data"]["dataset_name"] == "KhangTruong/COCO-inpainted"
    assert DEFAULT_CONFIG["data"]["image_size"] == [512, 512]
    assert DEFAULT_CONFIG["data"]["num_workers"] == -1
    assert "project" in DEFAULT_CONFIG
    assert "model" in DEFAULT_CONFIG
    assert "loss" in DEFAULT_CONFIG
    assert "training" in DEFAULT_CONFIG
    assert "bootstrapping" in DEFAULT_CONFIG
    assert "hard_mining" in DEFAULT_CONFIG
    assert "logging" in DEFAULT_CONFIG
    assert "post_processing" in DEFAULT_CONFIG


@pytest.mark.timeout(60)
def test_deep_merge():
    base = {"a": 1, "sub": {"b": 2, "c": 3}}
    update = {"sub": {"b": 20, "d": 40}, "e": 5}
    merged = deep_merge(base, update)
    assert merged == {"a": 1, "sub": {"b": 20, "c": 3, "d": 40}, "e": 5}
    # Check base wasn't mutated
    assert base["sub"]["b"] == 2


@pytest.mark.timeout(120)
def test_all_repo_configs_load_and_build_models():
    repo_root = Path(__file__).resolve().parent.parent
    config_files = sorted(glob.glob(str(repo_root / "configs" / "**" / "*.yaml"), recursive=True))

    assert len(config_files) >= 28, f"Expected at least 28 configs, found {len(config_files)}"

    for cfg_file in config_files:
        rel_path = os.path.relpath(cfg_file, repo_root)
        cfg = load_config(cfg_file)

        # Verify baseline sections exist without collisions
        assert hasattr(cfg, "project"), f"Missing 'project' in {rel_path}"
        assert hasattr(cfg, "data"), f"Missing 'data' in {rel_path}"
        assert hasattr(cfg, "model"), f"Missing 'model' in {rel_path}"
        assert hasattr(cfg, "loss"), f"Missing 'loss' in {rel_path}"
        assert hasattr(cfg, "training"), f"Missing 'training' in {rel_path}"

        # Verify model builds successfully
        model = build_model(cfg)
        assert isinstance(
            model,
            (UNet, DiffusionDiff, DiffusionDiffV2, DiffusionDiffMinimized),
        ), f"Unexpected model type for {rel_path}: {type(model)}"


@pytest.mark.timeout(60)
def test_copy_config_utility(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    source = repo_root / "configs" / "default.yaml"
    dest = tmp_path / "copied.yaml"

    copied = copy_config(
        source,
        dest_path=dest,
        overrides=["data.batch_size=32", "training.epochs=20"],
    )

    assert dest.exists()
    assert copied["data"]["batch_size"] == 32
    assert copied["training"]["epochs"] == 20
