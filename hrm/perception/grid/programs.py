"""Grid programs: compositions of primitives with an executable and an algebraic face.

Signatures follow the encoding algebra where one exists: a translation by
(dx, dy, dz) IS binding X^dx Y^dy W^dz, and a color cycle by k IS C^k on the
period-9 circular color base (nonzero colors cycle 1..9). Operators without
such a relation (dihedral, 3-D, recolor tables) get their own named symbols
so they never alias scene features.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, Tuple

import numpy as np

from hrm.perception.base import Program, ProgramError
from hrm.perception.grid import symmetry
from hrm.perception.grid.volume import from_volume, to_volume
from hrm.substrate import Codebook, PhasorVector

N_NONZERO_COLORS = 9      # colors 1..9; the circular base "grid.C" has this period


class Primitive(ABC):
    name: ClassVar[str] = "primitive"
    identity: ClassVar[bool] = False

    @abstractmethod
    def apply_volume(self, vol: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def signature(self, cb: Codebook) -> PhasorVector:
        ...

    @abstractmethod
    def describe(self) -> str:
        ...


@dataclass(frozen=True)
class Identity(Primitive):
    name: ClassVar[str] = "identity"
    identity: ClassVar[bool] = True

    def apply_volume(self, vol: np.ndarray) -> np.ndarray:
        return vol.copy()

    def signature(self, cb: Codebook) -> PhasorVector:
        return PhasorVector.identity(cb.dim, cb.dtype)

    def describe(self) -> str:
        return "identity"


@dataclass(frozen=True)
class Translate(Primitive):
    dx: int = 0
    dy: int = 0
    dz: int = 0
    name: ClassVar[str] = "translate"

    def apply_volume(self, vol: np.ndarray) -> np.ndarray:
        out = np.zeros_like(vol)
        depth, h, w = vol.shape
        zs, ys, xs = np.nonzero(vol)
        nx, ny, nz = xs + self.dx, ys + self.dy, zs + self.dz
        keep = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h) & (nz >= 0) & (nz < depth)
        out[nz[keep], ny[keep], nx[keep]] = vol[zs[keep], ys[keep], xs[keep]]
        return out

    def signature(self, cb: Codebook) -> PhasorVector:
        sig = cb.power("grid.X", self.dx) * cb.power("grid.Y", self.dy)
        return sig * cb.power("grid.W", self.dz) if self.dz else sig

    def describe(self) -> str:
        base = f"translate(dx={self.dx}, dy={self.dy}"
        return base + (f", dz={self.dz})" if self.dz else ")")


@dataclass(frozen=True)
class ColorCycle(Primitive):
    shift: int = 1
    name: ClassVar[str] = "color_cycle"

    def apply_volume(self, vol: np.ndarray) -> np.ndarray:
        out = vol.copy()
        mask = out != 0
        out[mask] = (out[mask] - 1 + self.shift) % N_NONZERO_COLORS + 1
        return out

    def signature(self, cb: Codebook) -> PhasorVector:
        return cb.power("grid.C", self.shift, period=N_NONZERO_COLORS)

    def describe(self) -> str:
        return f"color_cycle(shift={self.shift})"


@dataclass(frozen=True)
class RecolorMap(Primitive):
    mapping: Tuple[Tuple[int, int], ...] = ()
    name: ClassVar[str] = "recolor"

    def apply_volume(self, vol: np.ndarray) -> np.ndarray:
        return symmetry.color_permutation(vol, dict(self.mapping))

    def signature(self, cb: Codebook) -> PhasorVector:
        sig = PhasorVector.identity(cb.dim, cb.dtype)
        for src, dst in self.mapping:
            sig = sig * cb.power("grid.C", dst - src, period=N_NONZERO_COLORS)
        return sig

    def describe(self) -> str:
        return f"recolor({dict(self.mapping)})"


@dataclass(frozen=True)
class Dihedral(Primitive):
    """Planar D4 element applied slice-wise."""
    element: str = "e"
    name: ClassVar[str] = "dihedral"

    def apply_volume(self, vol: np.ndarray) -> np.ndarray:
        return symmetry.D4[self.element](vol)

    def signature(self, cb: Codebook) -> PhasorVector:
        return cb.symbol(f"grid.D4.{self.element}")

    def describe(self) -> str:
        return f"dihedral({self.element})"


@dataclass(frozen=True)
class Ortho3D(Primitive):
    """Proper 3-D operator; meaningful only when depth > 1."""
    element: str = "rz90"
    name: ClassVar[str] = "ortho3d"

    def apply_volume(self, vol: np.ndarray) -> np.ndarray:
        return symmetry.SYM3D[self.element](vol)

    def signature(self, cb: Codebook) -> PhasorVector:
        return cb.symbol(f"grid.O3.{self.element}")

    def describe(self) -> str:
        return f"ortho3d({self.element})"


PRIMITIVES: Dict[str, type] = {cls.name: cls for cls in
                               (Identity, Translate, ColorCycle, RecolorMap, Dihedral, Ortho3D)}


@dataclass
class GridProgram(Program):
    primitives: Tuple[Primitive, ...] = field(default_factory=tuple)
    provenance: str = "search"

    def apply(self, x: Any) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim not in (2, 3):
            raise ProgramError(f"grid program needs a 2-D or 3-D grid, got ndim={arr.ndim}")
        vol, was_2d = to_volume(arr)
        for p in self.primitives:
            vol = p.apply_volume(vol)
        return from_volume(vol, was_2d)

    def describe(self) -> str:
        return " . ".join(p.describe() for p in self.primitives) or "identity"

    def signature(self, cb: Codebook) -> PhasorVector:
        sig = PhasorVector.identity(cb.dim, cb.dtype)
        for p in self.primitives:
            sig = sig * p.signature(cb)
        return sig

    @property
    def complexity(self) -> int:
        return sum(1 for p in self.primitives if not p.identity)

    @property
    def family(self) -> str:
        return "+".join(p.name for p in self.primitives if not p.identity) or "identity"

    def to_payload(self) -> Dict[str, Any]:
        return {"primitives": [{"op": p.name, **asdict(p)} for p in self.primitives]}

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], provenance: str = "memory") -> "GridProgram":
        prims = []
        for spec in payload["primitives"]:
            spec = dict(spec)
            kind = PRIMITIVES[spec.pop("op")]
            if kind is RecolorMap:
                spec["mapping"] = tuple(tuple(m) for m in spec["mapping"])
            prims.append(kind(**spec))
        return cls(tuple(prims), provenance)
