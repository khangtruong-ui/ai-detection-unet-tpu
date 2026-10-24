"""Mask preprocessing and conversion utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image


def process_mask(
    mask_input: Any,
    target_size: Optional[Tuple[int, int]] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Standardize mask input into float32 array in [0, 1] range of shape (H, W, 1)."""
    if isinstance(mask_input, Image.Image):
        if target_size is not None and mask_input.size != (target_size[1], target_size[0]):
            mask_input = mask_input.resize((target_size[1], target_size[0]), Image.NEAREST)
        arr = np.array(mask_input.convert("L"), dtype=np.float32) / 255.0
    else:
        arr = np.asarray(mask_input, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

    if arr.ndim == 2:
        arr = arr[:, :, None]

    if threshold is not None:
        arr = (arr >= threshold).astype(np.float32)

    return arr
