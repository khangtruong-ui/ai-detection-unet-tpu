"""Configuration loader and management utilities for SID-UNet JAX/TPU.

Supports YAML serialization, nested dot-accessible dictionaries, and CLI overrides.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

logger = logging.getLogger("sid_unet.config")


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

    # Try YAML/JSON evaluation for lists and dicts
    if (val_str.startswith("[") and val_str.endswith("]")) or (val_str.startswith("{") and val_str.endswith("}")):
        try:
            return yaml.safe_load(val_str)
        except Exception:
            pass

    return val_str


def apply_overrides(config: ConfigDict, overrides: List[str]) -> ConfigDict:
    """Apply list of 'key.nested=value' overrides to ConfigDict in place."""
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

    return config


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[List[str]] = None,
) -> ConfigDict:
    """Load configuration from YAML file and apply optional overrides."""
    resolved_path = Path(config_path)
    if not resolved_path.exists():
        # Check relative to package/repo root
        repo_root = Path(__file__).resolve().parent.parent.parent
        candidate = repo_root / config_path
        if candidate.exists():
            resolved_path = candidate
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cfg = ConfigDict(data)

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
