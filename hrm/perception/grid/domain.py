"""Grid domain: ARC-style 2-D grids and voxel volumes, colors 0..9 (0 = background).

Encodings:
    cell    X^x Y^y W^z C^c
    object  Pi_T(sum of cells) bound with its attributes, tagged Z_OBJ^layer
    scene   bundle of objects + the flat cell checksum
Translations and color cycles act on these encodings exactly as their program
signatures say, so a task's pair vectors point straight at its program.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from hrm.config import PerceptionConfig
from hrm.perception.base import Domain, Example, Program, int_kwargs
from hrm.perception.grid.programs import (N_NONZERO_COLORS, ColorCycle, Dihedral, GridProgram,
                                          Identity, Ortho3D, RecolorMap, Translate)
from hrm.perception.grid.segmentation import AlgorithmicSegmenter, GridObject
from hrm.perception.grid.symmetry import D4, SYM3D
from hrm.perception.grid.volume import to_volume
from hrm.substrate import Codebook, PhasorVector, bundle


class GridDomain(Domain):
    name = "grid"
    PARSES = AlgorithmicSegmenter.HYPOTHESES
    n_parses = len(PARSES)
    closed_families = ("translate", "color_cycle", "recolor", "dihedral", "ortho3d")
    format_example = (((0, 3, 0), (0, 3, 0)), "translate(dx=1, dy=0) . color_cycle(shift=1)")

    def __init__(self, cb: Codebook, cfg: PerceptionConfig):
        super().__init__(cb)
        self.max_shift = cfg.grid_max_shift
        self.segmenter = AlgorithmicSegmenter(cfg.grid_max_objects)

    def accepts(self, demos: Sequence[Example]) -> bool:
        return len(demos) > 0 and all(_is_grid(x) for pair in demos for x in pair)

    # --- encoding --------------------------------------------------------------
    def encode_cell(self, x: float, y: float, color: int, z: float = 0) -> PhasorVector:
        v = self.cb.power("grid.X", x) * self.cb.power("grid.Y", y)
        if z:
            v = v * self.cb.power("grid.W", z)
        return v * self.cb.power("grid.C", color, period=N_NONZERO_COLORS)

    def encode_flat(self, grid: Any) -> PhasorVector:
        """Superposition of every non-background cell: a checksum, never decoded cell-wise."""
        vol, _ = to_volume(grid)
        zs, ys, xs = np.nonzero(vol)
        if len(xs) == 0:
            return PhasorVector.identity(self.cb.dim, self.cb.dtype)
        return bundle(self.encode_cell(int(x), int(y), int(vol[z, y, x]), int(z))
                      for x, y, z in zip(xs, ys, zs))

    def encode_object(self, obj: GridObject) -> PhasorVector:
        body = bundle(self.encode_cell(x, y, obj.colors[(x, y, z)], z) for x, y, z in obj.cells)
        cx, cy, cz = obj.centroid
        attrs = (self.cb.power("grid.SIZE", obj.size) * self.cb.power("grid.X", cx)
                 * self.cb.power("grid.Y", cy) * self.cb.power("grid.W", cz)
                 * self.cb.power("grid.C", obj.dominant_color, period=N_NONZERO_COLORS))
        return bundle([body, attrs]) * self.cb.power("grid.Z_OBJ", obj.layer)

    def encode(self, x: Any, parse: int = 0) -> PhasorVector:
        seg = self.segmenter.segment(x, self.PARSES[parse])
        checksum = self.encode_flat(x)
        if not seg.objects:
            return checksum
        return bundle([self.encode_object(o) for o in seg.objects] + [checksum])

    # --- hypotheses -----------------------------------------------------------------
    def candidates(self, demos: Sequence[Example]) -> List[Program]:
        volumetric = any(to_volume(i)[0].shape[0] > 1 for i, _ in demos)
        s = self.max_shift
        dzs = range(-2, 3) if volumetric else (0,)
        cands: List[Program] = [GridProgram((Identity(),))]
        cands += [GridProgram((Translate(dx, dy, dz),))
                  for dx in range(-s, s + 1) for dy in range(-s, s + 1) for dz in dzs
                  if (dx, dy, dz) != (0, 0, 0)]
        cands += [GridProgram((ColorCycle(k),)) for k in range(1, N_NONZERO_COLORS)]
        cands += [GridProgram((Dihedral(e),)) for e in D4 if e != "e"]
        if volumetric:
            cands += [GridProgram((Ortho3D(e),)) for e in SYM3D]
        recolors = self._observed_recolors(demos)
        cands += recolors
        shifts = [Translate(dx, dy) for dx in (-2, -1, 1, 2) for dy in (-2, -1, 1, 2)]
        cands += [GridProgram((t, *r.primitives)) for r in recolors for t in shifts]
        return cands

    def _observed_recolors(self, demos: Sequence[Example]) -> List[GridProgram]:
        """A recolor table consistent with every demo, if the data implies one."""
        mapping: Dict[int, set] = {}
        for inp, out in demos:
            vin, _ = to_volume(inp)
            vout, _ = to_volume(out)
            if vin.shape != vout.shape:
                return []
            for src in set(vin.flatten().tolist()) - {0}:
                dsts = set(vout[vin == src].flatten().tolist()) - {0}
                if len(dsts) == 1:
                    mapping.setdefault(int(src), set()).add(next(iter(dsts)))
        proper = {s: next(iter(d)) for s, d in mapping.items() if len(d) == 1}
        proper = {s: d for s, d in proper.items() if s != d}
        return [GridProgram((RecolorMap(tuple(sorted(proper.items()))),))] if proper else []

    def compose(self, first: Program, then: Program) -> Program:
        if not (isinstance(first, GridProgram) and isinstance(then, GridProgram)):
            raise TypeError("grid domain composes GridPrograms only")
        return GridProgram(first.primitives + then.primitives, provenance="synthesized")

    # --- judging ----------------------------------------------------------------------
    def equal(self, pred: Any, target: Any) -> bool:
        return bool(np.array_equal(np.asarray(pred), np.asarray(target)))

    def similarity(self, pred: Any, target: Any) -> float:
        """Partial credit: exact cells, nonzero-mask IoU (structure), color histogram
        (palette). Decomposed credit gives search a gradient where the exact check is flat."""
        p, t = np.asarray(pred), np.asarray(target)
        if p.shape != t.shape:
            return 0.0
        if np.array_equal(p, t):
            return 1.0
        pm, tm = p != 0, t != 0
        union = int((pm | tm).sum())
        iou = (pm & tm).sum() / union if union else 1.0
        n_colors = int(max(p.max(initial=0), t.max(initial=0))) + 1
        hp = np.bincount(p[pm].ravel(), minlength=n_colors)
        ht = np.bincount(t[tm].ravel(), minlength=n_colors)
        hist = 1.0 - np.abs(hp - ht).sum() / max(int(hp.sum() + ht.sum()), 1)
        return float(0.5 * (p == t).mean() + 0.3 * iou + 0.2 * hist)

    def size(self, x: Any) -> int:
        return int(np.asarray(x).size)

    def program_from_payload(self, payload: Dict[str, Any]) -> Program:
        return GridProgram.from_payload(payload)

    def vocabulary(self) -> str:
        return ("translate(dx=INT, dy=INT) moves everything by dx columns and dy rows; "
                "color_cycle(shift=INT) adds shift to every nonzero color (1-9 wrap around); "
                "recolor({SRC: DST, ...}) replaces colors; dihedral(E) with E one of "
                "r90, r180, r270, fx, fy, fd, fa rotates or mirrors the grid")

    def program_from_terms(self, terms: List[Tuple[str, Any]]) -> Program:
        prims = []
        for name, args in terms:
            if name == "identity" and args is None:
                prims.append(Identity())
            elif name == "translate" and int_kwargs(args, {"dx", "dy", "dz"}, 30):
                prims.append(Translate(**args))
            elif name == "color_cycle" and int_kwargs(args, {"shift"}, 9):
                prims.append(ColorCycle(**args))
            elif (name == "recolor" and isinstance(args, dict) and args
                  and all(_color(k) and _color(v) for k, v in args.items())):
                prims.append(RecolorMap(tuple(sorted(args.items()))))
            elif name == "dihedral" and args in D4:
                prims.append(Dihedral(args))
            elif name == "ortho3d" and args in SYM3D:
                prims.append(Ortho3D(args))
            else:
                raise ValueError(f"not a grid operation: {name}({args})")
        if not prims:
            raise ValueError("empty program")
        return GridProgram(tuple(prims), provenance="language")


def _color(v: Any) -> bool:
    return type(v) is int and 0 <= v <= N_NONZERO_COLORS


def _is_grid(x: Any) -> bool:
    try:
        arr = np.asarray(x)
    except (ValueError, TypeError):      # ragged nested lists
        return False
    if arr.ndim not in (2, 3) or arr.size == 0 or arr.dtype.kind not in "iu":
        return False
    return bool(arr.min() >= 0 and arr.max() <= N_NONZERO_COLORS)
