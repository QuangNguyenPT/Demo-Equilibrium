"""Segmentation hypotheses over voxel grids (flood fills in pure numpy).

conn4 (face-adjacent, conn6 in 3-D), conn8 (vertex-adjacent, conn26 in 3-D),
same-color-anywhere, and flat (no objects). A wrong hypothesis fails to
converge and the controller swaps it on restart: segmentation errors surface
as residual, never as silent corruption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from hrm.perception.grid.volume import to_volume

BACKGROUND = 0
Coord = Tuple[int, int, int]          # (x, y, z)

_N6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
_N26 = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)]


@dataclass
class GridObject:
    cells: List[Coord]
    colors: Dict[Coord, int]
    layer: int                         # discovery order: Z_OBJ slot

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def centroid(self) -> Tuple[float, float, float]:
        n = len(self.cells)
        return (sum(c[0] for c in self.cells) / n, sum(c[1] for c in self.cells) / n,
                sum(c[2] for c in self.cells) / n)

    @property
    def dominant_color(self) -> int:
        vals = list(self.colors.values())
        return max(set(vals), key=vals.count)


@dataclass
class Segmentation:
    hypothesis: str
    objects: List[GridObject] = field(default_factory=list)


class AlgorithmicSegmenter:
    HYPOTHESES = ("conn4_color", "conn8_color", "same_color_global", "flat")

    def __init__(self, k_max: int = 16):
        self.k_max = k_max

    def segment(self, grid: np.ndarray, hypothesis: str) -> Segmentation:
        vol, _ = to_volume(grid)
        if hypothesis == "conn4_color":
            return Segmentation(hypothesis, self._components(vol, _N6))
        if hypothesis == "conn8_color":
            return Segmentation(hypothesis, self._components(vol, _N26))
        if hypothesis == "same_color_global":
            return Segmentation(hypothesis, self._by_color(vol))
        if hypothesis == "flat":
            return Segmentation(hypothesis, [])
        raise ValueError(f"unknown segmentation hypothesis '{hypothesis}'")

    def _components(self, vol: np.ndarray, nbrs) -> List[GridObject]:
        depth, h, w = vol.shape
        seen = np.zeros(vol.shape, dtype=bool)
        objs: List[GridObject] = []
        for z in range(depth):
            for y in range(h):
                for x in range(w):
                    if seen[z, y, x] or vol[z, y, x] == BACKGROUND:
                        continue
                    color = int(vol[z, y, x])
                    stack, cells = [(x, y, z)], []
                    seen[z, y, x] = True
                    while stack:
                        cx, cy, cz = stack.pop()
                        cells.append((cx, cy, cz))
                        for dx, dy, dz in nbrs:
                            nx, ny, nz = cx + dx, cy + dy, cz + dz
                            if (0 <= nx < w and 0 <= ny < h and 0 <= nz < depth
                                    and not seen[nz, ny, nx] and vol[nz, ny, nx] == color):
                                seen[nz, ny, nx] = True
                                stack.append((nx, ny, nz))
                    objs.append(GridObject(cells, {c: color for c in cells}, len(objs) + 1))
                    if len(objs) >= self.k_max:
                        return objs
        return objs

    def _by_color(self, vol: np.ndarray) -> List[GridObject]:
        objs: List[GridObject] = []
        for color in sorted(set(vol.flatten().tolist()) - {BACKGROUND}):
            zs, ys, xs = np.where(vol == color)
            cells = list(zip(xs.tolist(), ys.tolist(), zs.tolist()))
            objs.append(GridObject(cells, {c: int(color) for c in cells}, len(objs) + 1))
            if len(objs) >= self.k_max:
                break
        return objs
