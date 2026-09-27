"""Symmetry operators on 2-D grids and (Z, H, W) volumes.

D4     planar dihedral group acting on the (H, W) plane (slice-wise).
SYM3D  proper 3-D operators: rot90 about each principal axis, depth flip,
       depth/width swap (offered only when depth > 1).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from hrm.perception.grid.volume import from_volume, to_volume


def _planar(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def op(g: np.ndarray) -> np.ndarray:
        vol, was_2d = to_volume(g)
        return from_volume(fn(vol), was_2d)
    return op


D4: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "e":    _planar(lambda v: v.copy()),
    "r90":  _planar(lambda v: np.rot90(v, 1, axes=(1, 2))),
    "r180": _planar(lambda v: np.rot90(v, 2, axes=(1, 2))),
    "r270": _planar(lambda v: np.rot90(v, 3, axes=(1, 2))),
    "fx":   _planar(lambda v: np.flip(v, axis=1)),
    "fy":   _planar(lambda v: np.flip(v, axis=2)),
    "fd":   _planar(lambda v: np.swapaxes(v, 1, 2).copy()),
    "fa":   _planar(lambda v: np.swapaxes(np.rot90(v, 2, axes=(1, 2)), 1, 2).copy()),
}

D4_INVERSE: Dict[str, str] = {"e": "e", "r90": "r270", "r180": "r180", "r270": "r90",
                              "fx": "fx", "fy": "fy", "fd": "fd", "fa": "fa"}

SYM3D: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "rz90": lambda v: np.rot90(v, 1, axes=(1, 2)),
    "rx90": lambda v: np.rot90(v, 1, axes=(0, 1)),
    "ry90": lambda v: np.rot90(v, 1, axes=(0, 2)),
    "fz":   lambda v: np.flip(v, axis=0),
    "fzx":  lambda v: np.swapaxes(v, 0, 2).copy(),
}


def apply(name: str, grid: np.ndarray) -> np.ndarray:
    if name in D4:
        return D4[name](grid)
    vol, was_2d = to_volume(grid)
    return from_volume(SYM3D[name](vol), was_2d)


def color_permutation(grid: np.ndarray, perm: Dict[int, int]) -> np.ndarray:
    g = np.asarray(grid)
    out = g.copy()
    for src, dst in perm.items():
        out[g == src] = dst
    return out
