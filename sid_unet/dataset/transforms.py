"""Spatial transformations for images, masks, and high-dimensional cached latents."""

from __future__ import annotations

import random
from typing import Tuple
import numpy as np


class SpatialJointTransform:
    """Applies synchronous spatial 2D transformations (flips, rotations)

    to both input images/latents and ground-truth masks.
    """

    def __init__(
        self,
        horizontal_flip: float = 0.5,
        vertical_flip: float = 0.0,
        random_rotate90: float = 0.25,
    ):
        self.hflip = float(horizontal_flip)
        self.vflip = float(vertical_flip)
        self.rot90 = float(random_rotate90)

    def __call__(
        self,
        image_or_latent: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply transforms to image/latent array and mask array."""
        x = np.asarray(image_or_latent)
        m = np.asarray(mask)

        # Horizontal flip (axis 1: width in H, W, C or axis -2)
        if self.hflip > 0 and random.random() < self.hflip:
            x = np.flip(x, axis=1)
            m = np.flip(m, axis=1)

        # Vertical flip (axis 0: height)
        if self.vflip > 0 and random.random() < self.vflip:
            x = np.flip(x, axis=0)
            m = np.flip(m, axis=0)

        # 90-degree rotation
        if self.rot90 > 0 and random.random() < self.rot90:
            k = random.choice([1, 2, 3])
            x = np.rot90(x, k=k, axes=(0, 1))
            m = np.rot90(m, k=k, axes=(0, 1))

        return np.ascontiguousarray(x), np.ascontiguousarray(m)
