"""Configuration loader and management utilities for SID-UNet JAX/TPU.

Supports YAML serialization, nested dot-accessible dictionaries, recursive deep merging,
CLI overrides, and full compatibility with ai-detection-unet-gpu configurations.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import yaml

logger = logging.getLogger("sid_unet.config")


DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "sid_unet_tpu",
        "seed": 42,
        "device": "auto",  # 'auto', 'cuda', 'tpu', or 'cpu'
        "output_dir": "outputs",
        "hub_repo": None,
        "resume_repo": None,
    },
    "data": {
        "dataset_name": "KhangTruong/COCO-inpainted",
        "streaming": True,
        "image_size": [512, 512],
        "batch_size": 16,
        "num_workers": -1,  # -1 automatically uses all available CPU cores (os.cpu_count())
        "pin_memory": True,
        "shuffle_buffer_size": 1000,
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "eval_split": "test",
        "train_samples_per_epoch": -1,  # Set to -1 to run until dataset is depleted
        "val_samples_per_epoch": -1,
        "val_samples": -1,
        "test_samples": -1,
        "evaluate_on_test": True,
        "use_parquet_loader": True,
        "save_to_disk": False,
        "background_download": False,
        "columns": None,
        "token": None,
        "augmentations": {
            "horizontal_flip": 0.5,
            "vertical_flip": 0.2,
            "random_rotate90": 0.5,
            "color_jitter": 0.0,
        },
    },
    "model": {
        "name": "unet",
        "in_channels": 3,
        "out_channels": 1,
        "features": [64, 128, 256, 512],
        "bilinear": True,
        "dropout": 0.1,
        "aux_classifier": False,
        "num_classes": 3,
        "gradient_checkpointing": False,
    },
    "loss": {
        "mask_loss_type": "combined",
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "focal_weight": 0.5,
        "focal_gamma": 2.0,
        "focal_alpha": 0.25,
        "aux_loss_type": "cross_entropy",
        "aux_weight": 0.2,
    },
    "training": {
        "epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "use_8bit_optimizer": False,
        "check_8bit_compatibility": True,
        "fallback_to_16bit": True,
        "fallback_on_unsupported_8bit": True,
        "scheduler": "cosine",
        "resume_lr_mode": "auto",
        "resume_lr": None,
        "warmup_epochs": 1,
        "warmup_steps": 100,
        "total_steps": 10000,
        "min_lr": 1e-6,
        "grad_clip_norm": 1.0,
        "amp": True,
        "gradient_accumulation_steps": 1,
        "data_parallel": True,
        "use_hard_mining": False,
        "auto_batch_size": True,
        "empty_cache_per_epoch": True,
        "save_best": True,
        "save_latest": False,
        "checkpoint_period": 3600,
        "eval_interval": 1,
        "early_stopping_patience": 5,
        "early_stopping_metric": "val_iou",
        "early_stopping_mode": "max",
        "push_to_hub": False,
        "hub_repo": None,
        "hub_version": "v1",
    },
    "hard_mining": {
        "enabled": False,
        "metric": "median",
        "reset_epochs": 5,
    },
    "bootstrapping": {
        "enabled": False,
        "run_bootstrap": False,
        "epochs": 5,
        "num_samples": 512,
        "batch_size": None,
        "learning_rate": None,
        "freeze_strategy": "channel_stream",
        "stream_ratio": 0.5,
        "release_mode": "restore",
        "freeze_modules": [],
        "initialization": "kaiming_normal",
        "target_score": 0.50,
        "min_loss_drop": 0.15,
        "early_stopping": False,
        "patience": 3,
    },
    "logging": {
        "log_interval": 20,
        "log_memory": True,
        "measure_network": True,
    },
    "post_processing": {
        "enabled": True,
        "min_area": 64,
        "fill_holes": True,
        "max_hole_size": 256,
        "morphology": "open_close",
        "morph_kernel_size": 3,
    },
}


class ConfigDict(dict):
    """A dictionary subclass that allows attribute-style dot access to nested keys."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, val: Any) -> Any:
        if isinstance(val, dict) and not isinstance(val, cls):
            return cls(val)
        if isinstance(val, list):
            return [cls._wrap(item) for item in val]
        return val

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to standard nested python dictionary."""
        d = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                d[k] = v.to_dict()
            elif isinstance(v, list):
                d[k] = [item.to_dict() if isinstance(item, ConfigDict) else item for item in v]
            else:
                d[k] = v
        return d


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionary `update` into `base`."""
    result = copy.deepcopy(base)
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def parse_override_value(val_str: str) -> Any:
    """Parse string override value into appropriate primitive type or collection."""
    val_str = val_str.strip()
    if val_str.lower() in ("true", "yes", "on"):
        return True
    if val_str.lower() in ("false", "no", "off"):
        return False
    if val_str.lower() in ("none", "null"):
        return None

    try:
        if "." in val_str or "e" in val_str.lower():
            return float(val_str)
        return int(val_str)
    except ValueError:
        pass

    # Try YAML evaluation for lists and dicts
    if (val_str.startswith("[") and val_str.endswith("]")) or (val_str.startswith("{") and val_str.endswith("}")):
        try:
            return yaml.safe_load(val_str)
        except Exception:
            pass

    return val_str


def apply_overrides(config: Union[ConfigDict, Dict[str, Any]], overrides: Sequence[str]) -> ConfigDict:
    """Apply list of 'key.nested=value' overrides to config in place."""
    for override in overrides:
        if "=" not in override:
            logger.warning(f"Skipping invalid override expression '{override}': must contain '='")
            continue

        key_path, val_str = override.split("=", 1)
        parsed_val = parse_override_value(val_str)

        keys = key_path.strip().split(".")
        curr = config
        for k in keys[:-1]:
            if k not in curr or not isinstance(curr[k], (dict, ConfigDict)):
                curr[k] = ConfigDict()
            curr = curr[k]

        curr[keys[-1]] = parsed_val
        logger.info(f"🔧 Config override applied: {key_path.strip()} = {parsed_val}")

    return ConfigDict(config) if not isinstance(config, ConfigDict) else config


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Sequence[str]] = None,
) -> ConfigDict:
    """Load configuration from YAML file merged over DEFAULT_CONFIG and apply optional overrides."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    if config_path is not None:
        resolved_path = Path(config_path)
        if not resolved_path.exists():
            repo_root = Path(__file__).resolve().parent.parent.parent
            candidate = repo_root / config_path
            if candidate.exists():
                resolved_path = candidate
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(resolved_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        if user_config and isinstance(user_config, dict):
            config = deep_merge(config, user_config)

    cfg = ConfigDict(config)

    if overrides:
        apply_overrides(cfg, overrides)

    return cfg


def save_config(
    config: Union[ConfigDict, Dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """Save configuration dictionary to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, ConfigDict):
        data = config.to_dict()
    else:
        data = copy.deepcopy(config)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    return output_path


def copy_config(
    source_path: Union[str, Path],
    dest_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Sequence[str]] = None,
    merge_defaults: bool = False,
) -> Dict[str, Any]:
    """Duplicate a YAML configuration file, apply key.nested=value overrides, and save."""
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {source_path}")

    if merge_defaults:
        cfg = load_config(source_path, overrides=overrides)
        config_dict = cfg.to_dict()
    else:
        with open(source_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}
        if not isinstance(raw_dict, dict):
            raise ValueError(f"Invalid YAML content in '{source_path}'")
        if overrides:
            raw_dict = apply_overrides(raw_dict, overrides)
        config_dict = raw_dict

    if dest_path is not None:
        save_config(config_dict, dest_path)

    return config_dict
