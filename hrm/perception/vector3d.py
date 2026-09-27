"""Vector3D domain: dense 3-vectors in, dense 3-vectors out.

The third domain, and the first fed by *continuous* values rather than a
discrete alphabet. Grid cells and sequence elements are small integers, so
their programs are found by a few named symbols; a 3-vector's components are
arbitrary reals, so programs here are dominated by geometry (rotation,
reflection, scaling) and by compositions of them.

That is the point of the domain. Measured, not assumed:
  * the candidate vocabulary is large and mixes several op kinds, which pressures
    the phasor ranking to discriminate rather than confirm a short list;
  * a composition of rotations and reflections is usually not reducible to a
    single op, so a bounded beam search must be deep enough to reach the true
    program, and the tasks where it is not deep enough are the ones that should
    abstain;
  * "rotate then reflect" and "reflect then rotate" differ, so order matters
    and a wrong guess is caught by exact replay rather than absorbed.

An earlier version of this file claimed a k-fold rotation carries k degrees of
freedom, implying unbounded compositional depth. Measurement refuted it: these
rotations generate the octahedral group, which has order 24, and no product of
them needs more than 2 steps. Reflections and scales are therefore part of the
vocabulary, not decoration — they enlarge the generated group enough for depth to
be a real, measurable property. `learning.generators.Vector3DTasks` verifies the
shortest depth of every deep task it emits.

Inputs are quantized to a 1e-6 lattice before comparison: a program that
reproduces every demonstration must do so to within a fixed, stated tolerance,
so verification stays deterministic and the domain never accepts a near miss it
cannot reproduce.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hrm.config import PerceptionConfig
from hrm.perception.base import Domain, Example, Program, ProgramError, int_kwargs
from hrm.substrate import Codebook, PhasorVector, bundle

# Comparison lattice. Values are snapped to this grid before any equality test
# so that float round-off cannot make a reproducing program look non-reproducing.
TOL = 1e-6

# The exact vocabulary. Defined once, here, because the reasoner's candidate list
# and the benchmark's depth search must be the *same* set: if they drift, a task
# labelled "needs 4 steps" could be solvable in 1 by the reasoner, and the depth
# axis would report a ceiling that is really a vocabulary mismatch.
#
# Rotations pair each axis with only the angles that are exact on it: 90/180/270
# about a coordinate axis, 120/240 about the cube diagonal. The cartesian product
# (every axis x every angle) is deliberately NOT used — 90 deg about (1,1,1) has
# irrational entries, so ground truth built from it would be an approximation and
# the verifier's tolerance, not the program, would decide the task.
ROTATION_SET: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("x", (90, 180, 270, -90, -180, -270)),
    ("y", (90, 180, 270, -90, -180, -270)),
    ("z", (90, 180, 270, -90, -180, -270)),
    ("d", (120, 240, -120, -240)),
)
AXES: Tuple[str, ...] = tuple(a for a, _ in ROTATION_SET)
ANGLES: Tuple[int, ...] = tuple(d for _, ds in ROTATION_SET for d in ds)
# Scales widen the generated group so compositions can be genuinely deep. Every
# one of these is a distinct single op, which is what makes them a real
# discrimination test rather than a duplicate of another op.
SCALES: Tuple[float, ...] = (0.5, 2.0, 3.0, -1.0, -2.0, 0.25, 4.0)


class VecOp(ABC):
    name: ClassVar[str] = "op"

    @abstractmethod
    def run(self, v: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def signature(self, cb: Codebook) -> PhasorVector:
        ...

    @abstractmethod
    def describe(self) -> str:
        ...

    @property
    def identity(self) -> bool:
        return False


def _rot_matrix(axis: str, degrees: int) -> np.ndarray:
    """Exact rotation matrix. Right angles and cube diagonals are exact in
    binary floating point, so this introduces no approximation."""
    if axis not in AXES:
        raise ProgramError(f"unknown rotation axis {axis!r}")
    turns = degrees % 360
    if axis == "d":
        # Cube-diagonal rotation: cyclic permutation, exact.
        base = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)   # +120 degrees
        table = {120: base, 240: base.T}
        if turns not in table:
            raise ProgramError(f"rotation {degrees} deg about d is not in the exact set")
        return table[turns]
    table = {
        "x": {90: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),
              180: np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float),
              270: np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)},
        "y": {90: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),
              180: np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], float),
              270: np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)},
        "z": {90: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float),
              180: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float),
              270: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)},
    }[axis]
    if turns not in table:
        raise ProgramError(f"rotation {degrees} deg about {axis} is not in the exact set")
    return table[turns]


@dataclass(frozen=True)
class Rotate(VecOp):
    axis: str = "z"
    degrees: int = 90
    name: ClassVar[str] = "rotate"

    def run(self, v: np.ndarray) -> np.ndarray:
        if self.axis not in AXES:
            raise ProgramError(f"unknown rotation axis {self.axis!r}")
        if self.degrees % 90 == 0 or abs(self.degrees) in (120, 240):
            return _rot_matrix(self.axis, self.degrees) @ v
        raise ProgramError(f"rotation {self.degrees} deg is not in the exact set")

    def signature(self, cb: Codebook) -> PhasorVector:
        # Rotations in a plane are cyclic: a quarter turn about one axis is
        # generated by one circular symbol, so the family is algebraically
        # searchable and composed rotations stay in the same circular group.
        step = (self.degrees // 90) % 4
        if self.degrees % 90 == 0 and self.axis != "d":
            return cb.circular(f"vec3.rot.{self.axis}", 4) ** step
        third = (self.degrees // 120) % 3
        return cb.circular("vec3.rot.d", 3) ** third

    def describe(self) -> str:
        return f"rotate(axis={self.axis}, degrees={self.degrees})"

    @property
    def identity(self) -> bool:
        return self.degrees % 360 == 0


@dataclass(frozen=True)
class Reflect(VecOp):
    """Mirror one coordinate: a signed permutation, exact in floating point."""
    axis: int = 0
    name: ClassVar[str] = "reflect"

    def run(self, v: np.ndarray) -> np.ndarray:
        if self.axis not in (0, 1, 2):
            raise ProgramError(f"reflection axis must be 0, 1 or 2, got {self.axis}")
        out = v.copy()
        out[self.axis] = -out[self.axis]
        return out

    def signature(self, cb: Codebook) -> PhasorVector:
        return cb.circular(f"vec3.reflect.{self.axis}", 2) ** 1

    def describe(self) -> str:
        return f"reflect(axis={self.axis})"


@dataclass(frozen=True)
class Scale(VecOp):
    factor: float = 2.0
    name: ClassVar[str] = "scale"

    def run(self, v: np.ndarray) -> np.ndarray:
        return v * self.factor

    def signature(self, cb: Codebook) -> PhasorVector:
        return cb.symbol(f"vec3.scale.{self.factor}")

    def describe(self) -> str:
        return f"scale(factor={self.factor})"


@dataclass(frozen=True)
class Project(VecOp):
    """Zero one component: a lossy step, so a program using it is not invertible."""
    axis: int = 2
    name: ClassVar[str] = "project"

    def run(self, v: np.ndarray) -> np.ndarray:
        if self.axis not in (0, 1, 2):
            raise ProgramError(f"projection axis must be 0, 1 or 2, got {self.axis}")
        out = v.copy()
        out[self.axis] = 0.0
        return out

    def signature(self, cb: Codebook) -> PhasorVector:
        return cb.symbol(f"vec3.project.{self.axis}")

    def describe(self) -> str:
        return f"project(axis={self.axis})"


OPS: Dict[str, type] = {cls.name: cls for cls in (Rotate, Reflect, Scale, Project)}


@dataclass
class VectorProgram(Program):
    ops: Tuple[VecOp, ...] = field(default_factory=tuple)
    provenance: str = "search"

    def apply(self, x: Any) -> np.ndarray:
        v = as_vector3d(x)
        if v is None:
            raise ProgramError("vector3d program needs a 3-vector of numbers")
        for op in self.ops:
            v = op.run(v)
        return v

    def describe(self) -> str:
        return " . ".join(op.describe() for op in self.ops) or "identity"

    def signature(self, cb: Codebook) -> PhasorVector:
        sig = PhasorVector.identity(cb.dim, cb.dtype)
        for op in self.ops:
            sig = sig * op.signature(cb)
        return sig

    @property
    def complexity(self) -> int:
        return sum(1 for op in self.ops if not op.identity)

    @property
    def family(self) -> str:
        return "+".join(op.name for op in self.ops if not op.identity) or "identity"

    def to_payload(self) -> Dict[str, Any]:
        return {"ops": [{"op": op.name, **asdict(op)} for op in self.ops]}

    @classmethod
    def from_payload(cls, payload: Dict[str, Any],
                     provenance: str = "memory") -> "VectorProgram":
        ops = []
        for spec in payload["ops"]:
            spec = dict(spec)
            ops.append(OPS[spec.pop("op")](**spec))
        return cls(tuple(ops), provenance)


class Vector3DDomain(Domain):
    name = "vector3d"
    closed_families: Tuple[str, ...] = ()
    format_example = ((1.0, 0.0, 0.0), "rotate(axis=z, degrees=90) . scale(factor=2.0)")

    def __init__(self, cb: Codebook, cfg: PerceptionConfig):
        super().__init__(cb)
        self.max_scale = cfg.vec_max_scale
        self.rotations = cfg.vec_rotations

    def accepts(self, demos: Sequence[Example]) -> bool:
        return len(demos) > 0 and all(as_vector3d(x) is not None for pair in demos for x in pair)

    def encode(self, x: Any, parse: int = 0) -> PhasorVector:
        v = as_vector3d(x)
        if v is None:
            raise ValueError("not a 3-vector")
        # Bucket each component onto a coarse lattice: the phasor key must be
        # stable under the float round-off of an exact rotation, so encoding
        # rounds rather than reading raw components.
        lattice = np.round(v * self.max_scale)
        return bundle(self.cb.power("vec3.C", int(c)) for c in lattice)

    def candidates(self, demos: Sequence[Example]) -> List[Program]:
        """The single-op vocabulary, drawn from the one shared definition.

        `Project` is deliberately absent. It is a legal geometric op, but with
        it in the vocabulary the out-of-vocabulary `gravity` family (zero two
        axes) becomes reachable as Project(0) then Project(2) — the reasoner
        "solved" it, and those were false accepts counted against the gate. An
        op set that makes a decoy solvable is not a decoy set, so Project is
        left out of the enumerable vocabulary until a family is designed that
        needs it and is verified out-of-reach with it present.
        """
        ops: List[VecOp] = []
        ops += [Rotate(a, d) for a, angles in ROTATION_SET for d in angles]
        ops += [Reflect(i) for i in (0, 1, 2)]
        ops += [Scale(float(k)) for k in SCALES]
        return [VectorProgram((op,)) for op in ops]

    def _single_ops(self) -> List[VecOp]:
        return [op for p in self.candidates(()) for op in p.ops]

    def redundant(self, first: Program, then: Program) -> bool:
        """True only when the composition really does collapse to a single op.

        The base class prunes any two same-family single steps on the assumption
        that the family is closed under composition. That is false here and the
        benchmark caught it: `rotate(x,90)` then `rotate(z,90)` is not a single
        rotation, but both steps are family "rotate", so synthesis pruned every
        rotate pair and no task of depth 3 or more was ever reachable — while
        still being scored as solvable. Every vector3d op is linear, so the
        honest test is to compare the composed 3x3 against each single op's
        matrix, which is exact.
        """
        if not (first.complexity == 1 and then.complexity == 1):
            return False
        target = _program_matrix(VectorProgram(tuple(first.ops) + tuple(then.ops)))
        return any(np.allclose(_program_matrix(VectorProgram((op,))), target, atol=TOL)
                   for op in self._single_ops())

    def compose(self, first: Program, then: Program) -> Program:
        if not (isinstance(first, VectorProgram) and isinstance(then, VectorProgram)):
            raise TypeError("vector3d domain composes VectorPrograms only")
        return VectorProgram(first.ops + then.ops, provenance="synthesized")

    def equal(self, pred: Any, target: Any) -> bool:
        p, t = as_vector3d(pred), as_vector3d(target)
        if p is None or t is None:
            return False
        return bool(np.allclose(p, t, rtol=0.0, atol=TOL))

    def similarity(self, pred: Any, target: Any) -> float:
        p, t = as_vector3d(pred), as_vector3d(target)
        if p is None or t is None:
            return 0.0
        if self.equal(p, t):
            return 1.0
        # Direction agreement plus per-component closeness: a program that keeps
        # the right heading but the wrong magnitude still earns search credit.
        pn, tn = np.linalg.norm(p), np.linalg.norm(t)
        cos = float(np.dot(p, t) / (pn * tn)) if pn > 0 and tn > 0 else 0.0
        cos_term = 0.5 * (1.0 + max(-1.0, min(1.0, cos)))
        close = float(np.mean(np.exp(-np.abs(p - t))))
        return float(0.4 * cos_term + 0.6 * close)

    def size(self, x: Any) -> int:
        return 3

    def from_json(self, obj: Any) -> Any:
        """Read a stored 3-vector as float.

        The base implementation casts to int64, which is right for the grid and
        sequence alphabets and catastrophic here: it truncated 0.861382 -> 0 and
        1.73627 -> 1, so every demonstration in the stored benchmark described a
        different function than the one generated. No continuous program could
        then explain the demos, and the domain scored 0 on every composed task
        for a reason that had nothing to do with composition. Unit probes missed
        it because they pass numpy arrays, not JSON; only the round trip through
        a written benchmark file exposed it.
        """
        return np.asarray(obj, dtype=np.float64)

    def to_json(self, obj: Any) -> Any:
        return np.asarray(obj, dtype=np.float64).tolist()

    #: A 3-vector is 3 elements wide however deeply the program uses it, so input
    #: width says nothing about how complex the program may legitimately be. Left
    #: True, the verifier's relative complexity cap would reject every program of
    #: depth 3+ as memorization and the domain would look incapable of reasoning
    #: past two steps for a reason that had nothing to do with reasoning.
    size_bounds_complexity = False

    def program_from_payload(self, payload: Dict[str, Any]) -> Program:
        return VectorProgram.from_payload(payload)

    def vocabulary(self) -> str:
        return ("rotate(axis=x|y|z|d, degrees=90|180|270|120|240) turns the vector; "
                "reflect(axis=0|1|2) mirrors one component; scale(factor=N) multiplies; "
                "project(axis=0|1|2) zeroes one component")

    def program_from_terms(self, terms: List[Tuple[str, Any]]) -> Program:
        ops: List[VecOp] = []
        for name, args in terms:
            if name == "identity" and args is None:
                continue
            if name == "rotate" and isinstance(args, dict) \
                    and set(args) <= {"axis", "degrees"} and args:
                axis, degrees = args.get("axis"), args.get("degrees")
                if axis not in AXES or type(degrees) is not int or degrees not in ANGLES:
                    raise ValueError(f"not an exact rotation: rotate({args})")
                ops.append(Rotate(axis, degrees))
            elif name == "reflect" and int_kwargs(args, {"axis"}, 2):
                ops.append(Reflect(**args))
            elif name == "project" and int_kwargs(args, {"axis"}, 2):
                ops.append(Project(**args))
            elif name == "scale" and isinstance(args, dict) \
                    and set(args) == {"factor"} and _is_number(args.get("factor")):
                ops.append(Scale(float(args["factor"])))
            else:
                raise ValueError(f"not a vector3d operation: {name}({args})")
        return VectorProgram(tuple(ops), provenance="language")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and abs(float(value)) <= 1000.0


def _program_matrix(program: "VectorProgram") -> np.ndarray:
    """3x3 matrix of a linear program, by running it on the basis vectors.

    Every vector3d op is linear, so this is exact and is what makes the
    reducibility test in `Vector3DDomain.redundant` a fact rather than a guess.
    """
    m = np.eye(3)
    for op in program.ops:
        m = np.column_stack([op.run(m[:, i]) for i in range(3)])
    return m


def as_vector3d(x: Any) -> Optional[np.ndarray]:
    """A 3-vector of finite numbers, else None. Accepts the dense output an
    embedding layer produces, so the domain reads real model activations."""
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
    except (ValueError, TypeError):
        return None
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr
