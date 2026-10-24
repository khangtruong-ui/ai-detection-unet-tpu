"""Model Cache Extractors for High-Dimensional Forensics Representations in JAX/Flax."""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Optional
import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger("sid_unet.cache.extractor")


class BaseCacheExtractor(abc.ABC):
    """Abstract base class for model cache feature extractors."""

    def __init__(self, model: Any, params: Any, fp16: bool = True):
        self.model = model
        self.params = params
        self.fp16 = fp16

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def total_channels(self) -> int:
        pass

    @abc.abstractmethod
    def extract_batch(self, images: np.ndarray, rng: Optional[jax.Array] = None) -> np.ndarray:
        """Extract high-dimensional tensor representation from a batch of RGB images.

        Args:
            images: Array of shape (B, H, W, 3).
            rng: Optional PRNGKey.

        Returns:
            Cached latent array of shape (B, H/8, W/8, total_channels).
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_channels": self.total_channels,
            "fp16": self.fp16,
        }


class DiffusionDiffMinimizedExtractor(BaseCacheExtractor):
    """Cache extractor for DiffusionDiffMinimized."""

    @property
    def model_name(self) -> str:
        return "diffusion_diff_minimized"

    @property
    def total_channels(self) -> int:
        return getattr(self.model, "total_channels", 84)

    def extract_batch(self, images: np.ndarray, rng: Optional[jax.Array] = None) -> np.ndarray:
        x = jnp.asarray(images)
        if hasattr(self.model, "extract_cache_tensors"):
            # Bind parameters
            z = self.model.apply(self.params, x, rng=rng, method=self.model.extract_cache_tensors)
        else:
            # Fallback mock extraction for tests
            b, h, w, _ = x.shape
            z = jnp.zeros((b, h // 8, w // 8, self.total_channels), dtype=jnp.float32)

        z_np = np.asarray(z)
        if self.fp16 and z_np.dtype != np.float16:
            z_np = z_np.astype(np.float16)
        return z_np


class DiffusionDiffV2Extractor(BaseCacheExtractor):
    """Cache extractor for DiffusionDiffV2."""

    @property
    def model_name(self) -> str:
        return "diffusion_diff_v2"

    @property
    def total_channels(self) -> int:
        return getattr(self.model, "total_channels", 84)

    def extract_batch(self, images: np.ndarray, rng: Optional[jax.Array] = None) -> np.ndarray:
        x = jnp.asarray(images)
        if hasattr(self.model, "extract_cache_tensors"):
            z = self.model.apply(self.params, x, rng=rng, method=self.model.extract_cache_tensors)
        else:
            b, h, w, _ = x.shape
            z = jnp.zeros((b, h // 8, w // 8, self.total_channels), dtype=jnp.float32)

        z_np = np.asarray(z)
        if self.fp16 and z_np.dtype != np.float16:
            z_np = z_np.astype(np.float16)
        return z_np


class DiffusionDiffExtractor(BaseCacheExtractor):
    """Cache extractor for standard DiffusionDiff."""

    @property
    def model_name(self) -> str:
        return "diffusion_diff"

    @property
    def total_channels(self) -> int:
        return getattr(self.model, "total_channels", 84)

    def extract_batch(self, images: np.ndarray, rng: Optional[jax.Array] = None) -> np.ndarray:
        x = jnp.asarray(images)
        if hasattr(self.model, "extract_cache_tensors"):
            z = self.model.apply(self.params, x, rng=rng, method=self.model.extract_cache_tensors)
        else:
            b, h, w, _ = x.shape
            z = jnp.zeros((b, h // 8, w // 8, self.total_channels), dtype=jnp.float32)

        z_np = np.asarray(z)
        if self.fp16 and z_np.dtype != np.float16:
            z_np = z_np.astype(np.float16)
        return z_np


def get_cache_extractor(model: Any, params: Any, fp16: bool = True) -> BaseCacheExtractor:
    """Factory creating corresponding cache extractor for given model."""
    cls_name = model.__class__.__name__.lower()
    if "minimized" in cls_name:
        return DiffusionDiffMinimizedExtractor(model, params, fp16=fp16)
    elif "v2" in cls_name:
        return DiffusionDiffV2Extractor(model, params, fp16=fp16)
    else:
        return DiffusionDiffExtractor(model, params, fp16=fp16)
