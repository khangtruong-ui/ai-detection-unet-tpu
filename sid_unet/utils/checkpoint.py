"""Checkpoint save/load utilities with bi-directional ai-detection-unet-gpu interoperability.

Ensures that JAX/Flax Diffusion-Diff models default to saving and loading weights in
the exact PyTorch state dict format compatible with ai-detection-unet-gpu.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

try:
    from jax_model_zoo.converters.jax_to_torch import convert_jax_diffusion_diff_to_torch
    from jax_model_zoo.converters.torch_to_jax import convert_torch_diffusion_diff_to_jax
    from jax_model_zoo.core.checkpoint import (
        load_checkpoint as jmz_load_checkpoint,
        save_checkpoint as jmz_save_checkpoint,
    )
    _HAS_JMZ = True
except ImportError:
    _HAS_JMZ = False

logger = logging.getLogger("sid_unet.checkpoint")

LOCAL_CHECKPOINT_PRIORITY = [
    "checkpoint_latest.pt",
    "checkpoint_periodic.pt",
    "checkpoint_best.pt",
    "checkpoint_latest.safetensors",
    "checkpoint_best.safetensors",
]

HF_CHECKPOINT_CANDIDATES = [
    "checkpoints/checkpoint_latest.pt",
    "checkpoint_latest.pt",
    "checkpoints/checkpoint_best.pt",
    "checkpoint_best.pt",
    "checkpoints/checkpoint_periodic.pt",
    "checkpoint_periodic.pt",
    "model.pt",
    "pytorch_model.bin",
]


def is_hf_repo_id(identifier: str) -> bool:
    """Determine if a given string represents a Hugging Face repository identifier."""
    if not identifier or not isinstance(identifier, str):
        return False

    raw = identifier.strip()
    if raw.startswith("hf://") or raw.startswith("https://huggingface.co/"):
        return True

    # If the path exists as a local file or directory, prioritize local path
    if os.path.exists(raw):
        return False

    if raw.startswith("/") or raw.startswith("./") or raw.startswith("../") or "\\" in raw:
        return False

    for prefix in ["checkpoints/", "outputs/", "weights/", "runs/"]:
        if raw.lower().startswith(prefix):
            return False

    repo_part = raw.split(":", 1)[0]
    parts = repo_part.split("/")
    if len(parts) >= 2:
        owner, repo = parts[0], parts[1]
        if re.match(r"^[a-zA-Z0-9_\-\.]+$", owner) and re.match(r"^[a-zA-Z0-9_\-\.]+$", repo):
            if len(parts) == 2:
                ext = os.path.splitext(repo)[1].lower()
                if ext in [".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".yaml", ".yml", ".json"]:
                    return False
            return True

    return False


def parse_hf_repo_uri(uri: str) -> Tuple[str, Optional[str]]:
    """Parse a Hugging Face URI/identifier into (repo_id, optional_filename)."""
    cleaned = uri.strip()
    if cleaned.startswith("hf://"):
        cleaned = cleaned[5:]
    elif cleaned.startswith("https://huggingface.co/"):
        cleaned = cleaned[23:]

    filename = None
    if ":" in cleaned:
        repo_id, filename = cleaned.split(":", 1)
    else:
        parts = cleaned.split("/")
        if len(parts) > 2:
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[2:])
        else:
            repo_id = cleaned

    return repo_id.strip("/"), filename.strip("/") if filename else None


def upload_hf_checkpoint(
    file_path: Union[str, Path],
    repo_id: str,
    path_in_repo: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> Optional[str]:
    """Upload a checkpoint file to a Hugging Face Hub model repository."""
    file_path = Path(file_path)
    if not file_path.is_file():
        logger.warning(f"Cannot upload non-existent checkpoint: {file_path}")
        return None

    target_name = path_in_repo or f"checkpoints/{file_path.name}"
    msg = commit_message or f"Upload checkpoint {file_path.name}"

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        url = api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=target_name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=msg,
        )
        logger.info(f"🚀 Checkpoint successfully uploaded to Hugging Face: {repo_id}/{target_name}")
        return str(url)
    except Exception as e:
        logger.warning(f"⚠️ Failed to upload checkpoint to Hugging Face repo '{repo_id}': {e}")
        return None


def download_hf_checkpoint(
    repo_id_or_uri: str,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Download checkpoint from Hugging Face Hub, trying common checkpoint filenames."""
    from huggingface_hub import hf_hub_download

    repo_id, explicit_file = parse_hf_repo_uri(repo_id_or_uri)
    target_file = filename or explicit_file

    candidates = [target_file] if target_file else HF_CHECKPOINT_CANDIDATES
    last_err = None

    for cand in candidates:
        try:
            logger.info(f"Checking Hugging Face repo '{repo_id}' for file '{cand}'...")
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=cand,
                cache_dir=cache_dir,
            )
            logger.info(f"✅ Checkpoint downloaded from Hugging Face: {downloaded}")
            return downloaded
        except Exception as e:
            last_err = e
            continue

    raise FileNotFoundError(
        f"Could not find any valid checkpoint in Hugging Face repo '{repo_id}'. Last error: {last_err}"
    )


def find_auto_resume_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[str]:
    """Search for the most recent valid resume checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    for candidate in LOCAL_CHECKPOINT_PRIORITY:
        p = os.path.join(checkpoint_dir, candidate)
        if os.path.isfile(p):
            return p

    # Fallback to any .pt or .safetensors files
    pts = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")), key=os.path.getmtime, reverse=True)
    if pts:
        return pts[0]

    safes = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")), key=os.path.getmtime, reverse=True)
    if safes:
        return safes[0]

    return None


def save_checkpoint(
    params: Dict[str, Any],
    path: Union[str, Path],
    epoch: int = 1,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    best_score: Optional[float] = None,
    best_epoch: Optional[int] = None,
    config: Optional[Any] = None,
    model_type: str = "diffusion_diff",
    batch_stats: Optional[Dict[str, Any]] = None,
    save_torch_format: bool = True,
    hf_repo_id: Optional[str] = None,
    push_to_hub: bool = False,
) -> str:
    """Save model checkpoint.

    Defaults to PyTorch state dict format compatible with ai-detection-unet-gpu,
    storing 'model_state_dict', epoch, step, and evaluation metrics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert Flax params to PyTorch state dict for full interoperability
    if save_torch_format and _HAS_JMZ:
        try:
            torch_state = convert_jax_diffusion_diff_to_torch(params, config=config)
        except Exception as e:
            logger.warning(f"Could not convert JAX parameters to PyTorch state dict: {e}")
            torch_state = None
    else:
        torch_state = None

    cfg_dict = config.to_dict() if hasattr(config, "to_dict") else (dict(config) if config else {})

    if str(path).endswith(".safetensors"):
        if _HAS_JMZ:
            meta = {
                "epoch": str(epoch),
                "step": str(step),
                "best_score": str(best_score or 0.0),
                "model_type": model_type,
            }
            jmz_save_checkpoint(params, path, format="safetensors", metadata=meta)
            if push_to_hub and hf_repo_id:
                upload_hf_checkpoint(path, repo_id=hf_repo_id)
            return str(path)

    # Standard .pt / .pth checkpoint containing model_state_dict
    ckpt_dict: Dict[str, Any] = {
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "best_score": best_score,
        "best_epoch": best_epoch,
        "config": cfg_dict,
        "model_type": model_type,
    }

    if torch_state is not None:
        ckpt_dict["model_state_dict"] = torch_state

    # Also store raw leaves converted to host numpy for native fast JAX reload
    def _to_host_np(x: Any) -> Any:
        if isinstance(x, (np.ndarray, list, int, float)):
            return x
        if hasattr(x, "device"):
            return np.asarray(x)
        return x

    import jax
    ckpt_dict["jax_params"] = jax.tree_util.tree_map(_to_host_np, params)
    if batch_stats is not None:
        ckpt_dict["jax_batch_stats"] = jax.tree_util.tree_map(_to_host_np, batch_stats)

    torch.save(ckpt_dict, str(path))
    logger.info(f"💾 Checkpoint successfully saved to {path} (Epoch {epoch}, Step {step})")

    if push_to_hub and hf_repo_id:
        upload_hf_checkpoint(path, repo_id=hf_repo_id)

    return str(path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    config: Optional[Any] = None,
    model_type: str = "diffusion_diff",
) -> Dict[str, Any]:
    """Load checkpoint file and restore Flax parameters.

    Seamlessly supports:
    1. PyTorch checkpoints (.pt / .pth) from ai-detection-unet-gpu containing 'model_state_dict'.
    2. JAX safetensors / msgpack checkpoints.
    3. Combined hybrid checkpoints.
    4. Hugging Face Hub model repositories (e.g. 'KhangTruong/diffusion-diff-minimized-512').
    """
    path_str = str(checkpoint_path)
    if not os.path.exists(path_str):
        if is_hf_repo_id(path_str):
            logger.info(f"Resolving checkpoint from Hugging Face repository: {path_str}")
            path_str = download_hf_checkpoint(path_str)
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {path_str}")

    logger.info(f"Loading checkpoint from: {path_str}")

    # Case 1: Safetensors file
    if path_str.endswith(".safetensors") or path_str.endswith(".msgpack"):
        if _HAS_JMZ:
            loaded_tree = jmz_load_checkpoint(path_str)
            params = loaded_tree.get("params", loaded_tree)
            return {
                "params": params,
                "batch_stats": loaded_tree.get("batch_stats", {}),
                "epoch": 1,
                "step": 0,
                "metrics": {},
                "best_score": None,
            }

    # Case 2: PyTorch .pt / .pth checkpoint
    loaded = torch.load(path_str, map_location="cpu", weights_only=False)

    # 2a. Check if direct JAX params were saved
    if isinstance(loaded, dict) and "jax_params" in loaded:
        import jax.numpy as jnp
        import jax
        params = jax.tree_util.tree_map(jnp.asarray, loaded["jax_params"])
        batch_stats = jax.tree_util.tree_map(jnp.asarray, loaded.get("jax_batch_stats", {}))
        return {
            "params": params,
            "batch_stats": batch_stats,
            "epoch": loaded.get("epoch", 1),
            "step": loaded.get("step", 0),
            "metrics": loaded.get("metrics", {}),
            "best_score": loaded.get("best_score"),
            "best_epoch": loaded.get("best_epoch"),
            "config": loaded.get("config", {}),
        }

    # 2b. Extract model_state_dict from PyTorch / ai-detection-unet-gpu checkpoint
    torch_state = None
    if isinstance(loaded, dict):
        if "model_state_dict" in loaded:
            torch_state = loaded["model_state_dict"]
        elif "state_dict" in loaded:
            torch_state = loaded["state_dict"]
        elif any(".weight" in k for k in loaded.keys()):
            torch_state = loaded

    if torch_state is not None:
        if _HAS_JMZ:
            normalized_state = {}
            for k, v in torch_state.items():
                norm_k = k
                if "classifier_head.fc.1." in norm_k:
                    norm_k = norm_k.replace("classifier_head.fc.1.", "classifier_head.fc1.")
                elif "classifier_head.fc.4." in norm_k:
                    norm_k = norm_k.replace("classifier_head.fc.4.", "classifier_head.fc2.")
                normalized_state[norm_k] = v

            converted = convert_torch_diffusion_diff_to_jax(normalized_state, config=config)
            params = converted.get("params", converted)
            batch_stats = converted.get("batch_stats", {})
            return {
                "params": params,
                "batch_stats": batch_stats,
                "epoch": loaded.get("epoch", 1) if isinstance(loaded, dict) else 1,
                "step": loaded.get("step", 0) if isinstance(loaded, dict) else 0,
                "metrics": loaded.get("metrics", {}) if isinstance(loaded, dict) else {},
                "best_score": loaded.get("best_score") if isinstance(loaded, dict) else None,
                "best_epoch": loaded.get("best_epoch") if isinstance(loaded, dict) else None,
                "config": loaded.get("config", {}) if isinstance(loaded, dict) else {},
            }
        else:
            raise RuntimeError("jax-model-zoo is required for PyTorch state dict conversion.")

    raise ValueError(f"Could not parse checkpoint format at {path_str}")
