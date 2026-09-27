"""Grids are native 3-D: shape (Z, H, W), coordinates written (x, y, z).

A 2-D grid is the depth-1 slice; promotion/demotion happens at API
boundaries so ARC grids and voxel tasks share one code path.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def to_volume(grid: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Promote to (Z, H, W); returns (volume, was_2d)."""
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim == 2:
        return arr[None, :, :], True
    if arr.ndim == 3:
        return arr, False
    raise ValueError(f"grid must be 2-D or 3-D, got ndim={arr.ndim}")


def from_volume(vol: np.ndarray, was_2d: bool) -> np.ndarray:
    if was_2d and vol.ndim == 3 and vol.shape[0] == 1:
        return vol[0]
    return vol
