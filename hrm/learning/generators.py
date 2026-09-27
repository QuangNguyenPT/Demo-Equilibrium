"""Task generators: practice material with ground truth, per domain and family.

A task is generated from an executable program (or a process such as gravity),
so the expected answer is exact and free. Families name the concept regions
the self-model tracks. Some families are deliberately outside the vocabulary
(gravity): a healthy learner must discover it makes no progress there and stop
spending effort on them. Decoys have no explanation at all: a trustworthy
model abstains on every one of them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

import numpy as np

from hrm.perception.base import Example, Program
from hrm.perception.grid import ColorCycle, Dihedral, GridProgram, RecolorMap, Translate
from hrm.perception.sequence import (Affine, CumSum, Diff, Reverse, Rotate, SequenceProgram,
                                     Sort)
from hrm.perception.vector3d import (ROTATION_SET as V3_ROTATION_SET, SCALES as V3_SCALES,
                                     Reflect, Rotate as VecRotate, Scale, VectorProgram)

DECOY = "decoy"

# Shortest explanation each family admits, in non-identity steps. 0 means the
# family is outside the vocabulary (gravity) or explains nothing (decoy), so a
# correct model must abstain there. Benchmark scoring reads these to report a
# solve rate per depth band, which is the axis v1 could not measure.
_FAMILY_DEPTH = {
    ("grid", "translate"): 1, ("grid", "color_cycle"): 1, ("grid", "recolor"): 1,
    ("grid", "dihedral"): 1, ("grid", "translate+color_cycle"): 2, ("grid", "gravity"): 0,
    ("sequence", "offset"): 1, ("sequence", "scale"): 1, ("sequence", "affine"): 1,
    ("sequence", "rotate"): 1, ("sequence", "reverse"): 1, ("sequence", "sort"): 1,
    ("sequence", "cumsum"): 1, ("sequence", "diff"): 1, ("sequence", "scale+reverse"): 2,
    ("vector3d", "rotate"): 1, ("vector3d", "reflect"): 1, ("vector3d", "scale"): 1,
    ("vector3d", "rotate+reflect"): 2, ("vector3d", "rotate+scale"): 2,
    ("vector3d", "d2_rotate"): 2, ("vector3d", "d3_rotate"): 3, ("vector3d", "d4_rotate"): 4,
    ("vector3d", "gravity"): 0,
}


@dataclass
class GeneratedTask:
    domain: str
    family: str
    demos: List[Example]
    test_input: Any
    expected: Any
    depth: int = 0        # non-identity steps in the generating program (0 = unknown/decoy)


class TaskGenerator(ABC):
    domain: str = "domain"
    families: Tuple[str, ...] = ()
    n_demos: int = 3

    @abstractmethod
    def sample_input(self, rng: np.random.Generator) -> Any:
        ...

    @abstractmethod
    def transform(self, family: str, rng: np.random.Generator) -> Callable[[Any], Any]:
        """A random concrete transformation from `family` (e.g. one fixed shift)."""

    def generate(self, family: str, rng: np.random.Generator) -> GeneratedTask:
        if family not in self.families:
            raise ValueError(f"unknown {self.domain} family '{family}'")
        fn = self.transform(family, rng)
        pairs = [(x, fn(x)) for x in (self.sample_input(rng) for _ in range(self.n_demos + 1))]
        return GeneratedTask(self.domain, family, pairs[:-1], pairs[-1][0], pairs[-1][1],
                             depth=self.depth_of(family))

    def depth_of(self, family: str) -> int:
        """Steps in the shortest program that explains the family (0 = not in the
        vocabulary). Used to score a benchmark by explanation depth."""
        return _FAMILY_DEPTH.get((self.domain, family), 0)

    def decoy(self, rng: np.random.Generator) -> GeneratedTask:
        """Unrelated input/output pairs: no program explains them."""
        pairs = [(self.sample_input(rng), self.sample_input(rng)) for _ in range(self.n_demos + 1)]
        return GeneratedTask(self.domain, DECOY, pairs[:-1], pairs[-1][0], pairs[-1][1])


def _program(p: Program) -> Callable[[Any], Any]:
    return p.apply


class GridTasks(TaskGenerator):
    domain = "grid"
    families = ("translate", "color_cycle", "recolor", "dihedral", "translate+color_cycle",
                "gravity")

    def __init__(self, size: int = 9, max_shift: int = 3):
        self.size = size
        self.max_shift = max_shift

    def sample_input(self, rng: np.random.Generator) -> np.ndarray:
        g = np.zeros((self.size, self.size), dtype=np.int64)
        for _ in range(int(rng.integers(1, 4))):
            h, w = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            y, x = int(rng.integers(0, self.size - h)), int(rng.integers(0, self.size - w))
            g[y:y + h, x:x + w] = int(rng.integers(1, 9))
        return g

    def transform(self, family: str, rng: np.random.Generator) -> Callable[[Any], Any]:
        def shift() -> Translate:
            dx, dy = (int(v) for v in rng.choice(
                [v for v in range(-self.max_shift, self.max_shift + 1) if v], 2))
            return Translate(dx, dy)

        if family == "translate":
            return _program(GridProgram((shift(),)))
        if family == "color_cycle":
            return _program(GridProgram((ColorCycle(int(rng.integers(1, 9))),)))
        if family == "recolor":
            src, dst = (int(c) for c in rng.choice(np.arange(1, 9), 2, replace=False))
            return _program(GridProgram((RecolorMap(((src, dst),)),)))
        if family == "dihedral":
            return _program(GridProgram((Dihedral(str(rng.choice(["r90", "r180", "fx", "fy"]))),)))
        if family == "translate+color_cycle":
            return _program(GridProgram((shift(), ColorCycle(int(rng.integers(1, 9))))))
        return gravity


class SequenceTasks(TaskGenerator):
    domain = "sequence"
    families = ("offset", "scale", "affine", "rotate", "reverse", "sort", "cumsum", "diff",
                "scale+reverse")

    def sample_input(self, rng: np.random.Generator) -> List[int]:
        n = int(rng.integers(4, 7))
        return [int(v) for v in rng.choice(np.arange(0, 10), n, replace=False)]

    def transform(self, family: str, rng: np.random.Generator) -> Callable[[Any], Any]:
        scale = int(rng.choice([-3, -2, 2, 3]))
        offset = int(rng.choice([v for v in range(-9, 10) if v]))
        ops = {
            "offset": (Affine(1, offset),),
            "scale": (Affine(scale, 0),),
            "affine": (Affine(scale, offset),),
            "rotate": (Rotate(int(rng.integers(1, 3))),),
            "reverse": (Reverse(),),
            "sort": (Sort(bool(rng.integers(0, 2))),),
            "cumsum": (CumSum(),),
            "diff": (Diff(),),
            "scale+reverse": (Affine(scale, 0), Reverse()),
        }[family]
        return _program(SequenceProgram(ops))


def gravity(grid: Any) -> np.ndarray:
    """Every colored cell falls to the bottom of its column (outside the grid vocabulary)."""
    g = np.asarray(grid)
    out = np.zeros_like(g)
    h = g.shape[0]
    for x in range(g.shape[1]):
        col = g[:, x][g[:, x] != 0]
        if len(col):
            out[h - len(col):, x] = col
    return out


class Vector3DTasks(TaskGenerator):
    """3-vectors in, 3-vectors out — the geometry axis of benchmark v2.

    Ground truth is built from the exact op set (right angles, cube diagonals,
    signed permutations), so every demonstration is reproducible to the last
    bit and the verifier's tolerance is never what decides a task.

    **Why these families have real depth.** The obvious design — compose k random
    rotations and call the family ``d{k}_rotate`` — does not work, and measuring
    it is why this docstring is here. The enumerated rotations generate the
    octahedral group: only 24 distinct matrices exist, 11 of them single ops, and
    a random k-fold product collapses back to a single op ~40% of the time and
    always to <=2 steps. So ``d5_rotate`` is really a depth-1 task and
    `synthesis_depth=3` clears the whole family. A depth axis built that way
    measures nothing.

    Instead each deep family is *constructed* to need exactly k steps: the
    generator searches for a product whose shortest explanation within the
    enumerated vocabulary is k, and raises rather than shipping a mislabelled
    task. Depth is a verified property of the task, not a label.

    **The vocabulary had to grow for this to be possible.** With rotations alone
    the op set generates the octahedral group (order 24, 11 single ops), whose
    longest shortest-product is 2: no product can require 3 or more steps, so a
    depth axis capped at 2 is all rotations permit, and the first attempt at this
    benchmark silently mislabelled every "deep" family. Reflections reach 3 and
    scales reach 4, which is why `Reflect` and `Scale` are in the shared
    vocabulary rather than being cosmetic. A depth benchmark whose vocabulary
    cannot express depth measures nothing.

    Families stop at d4 rather than going further: `synthesis_depth` is 3, so d4
    already probes the ceiling, and d5+ is the same "unreachable" verdict bought
    at ~6s of search per task.
    """
    domain = "vector3d"
    families = ("rotate", "rotate+reflect", "rotate+scale", "reflect", "scale",
                "d2_rotate", "d3_rotate", "d4_rotate", "gravity")

    # Exact rotation set, shared with the domain so the two cannot drift.
    ROTATION_SET: Tuple[Tuple[str, Tuple[int, ...]], ...] = V3_ROTATION_SET
    # Draws per family before giving up on finding a product of the target depth.
    _MAX_DRAWS = 3000

    def sample_input(self, rng: np.random.Generator) -> np.ndarray:
        # Embedding-like activations: dense, continuous, unscaled.
        v = rng.normal(0.0, 1.0, size=3)
        return np.round(v, 6)

    def _rotation(self, rng: np.random.Generator) -> VecRotate:
        axis, angles = self.ROTATION_SET[int(rng.integers(0, len(self.ROTATION_SET)))]
        return VecRotate(axis, int(rng.choice(angles)))

    def _candidate_ops(self) -> List[VecOp]:
        """Generators of the deep families, from the domain's shared vocabulary.

        Must stay a subset of `Vector3DDomain.candidates()` or the depth label
        describes a vocabulary the reasoner never searches. `Project` is
        excluded: it is singular, and shortest-product is ill-defined for a
        semigroup with zero divisors.
        """
        return [VecRotate(a, d) for a, angles in V3_ROTATION_SET for d in angles] + \
               [Reflect(i) for i in (0, 1, 2)] + [Scale(float(k)) for k in V3_SCALES]

    def _shortest_depth(self, ops: Tuple[VecOp, ...], limit: int) -> Optional[int]:
        """Fewest enumerated single ops whose product equals `ops`, or None past `limit`.

        Breadth-first over compositions up to `limit` steps, comparing exact
        matrices. Rotations and reflections are linear, so a composition is its
        matrix product and the comparison is exact. This is what makes the depth
        label true rather than asserted.
        """
        target = _mat(ops)
        singles = self._candidate_ops()
        for op in singles:
            if np.allclose(_mat((op,)), target, atol=1e-9):
                return 1
        frontier = {_key(_mat((op,))) for op in singles}
        for depth in range(2, limit + 1):
            nxt = set()
            for base in frontier:
                for op in singles:
                    m = _mat((op,)) @ _unkey(base)
                    if np.allclose(m, target, atol=1e-9):
                        return depth
                    nxt.add(_key(m))
            frontier = nxt
        return None

    def _deep_product(self, rng: np.random.Generator, depth: int) -> Tuple[VecOp, ...]:
        """A product of exactly `depth` steps that nothing shorter explains.

        Draws from the same op set `_shortest_depth` searches, so the label and
        the search can never disagree about the vocabulary. Raises instead of
        shipping a mislabelled task: a benchmark that quietly emits depth-1 tasks
        under a `d4` label is worse than no benchmark, because it reports a
        number that means nothing.
        """
        pool = self._candidate_ops()
        for _ in range(self._MAX_DRAWS):
            ops = tuple(pool[int(rng.integers(0, len(pool)))] for _ in range(depth + 1))
            if self._shortest_depth(ops, depth) == depth:
                return ops
        raise RuntimeError(
            f"[BENCHMARK] no product of depth {depth} found in {self._MAX_DRAWS} draws; "
            f"the vocabulary generates too small a group to express depth {depth}")

    def transform(self, family: str, rng: np.random.Generator) -> Callable[[Any], Any]:
        rot = self._rotation(rng)
        if family == "rotate":
            return _program(VectorProgram((rot,)))
        if family == "rotate+reflect":
            return _program(VectorProgram((rot, Reflect(int(rng.integers(0, 3))))))
        if family == "rotate+scale":
            return _program(VectorProgram((rot, Scale(float(rng.choice([2.0, 3.0, 0.5]))))))
        if family == "reflect":
            return _program(VectorProgram((Reflect(int(rng.integers(0, 3))),)))
        if family == "scale":
            return _program(VectorProgram((Scale(float(rng.choice([2.0, 3.0, 0.5, -1.0]))),)))
        if family == "gravity":
            return vector_gravity
        depth = {"d2_rotate": 2, "d3_rotate": 3, "d4_rotate": 4}[family]
        ops = self._deep_product(rng, depth)
        return _program(VectorProgram(ops))

    def decoy(self, rng: np.random.Generator) -> GeneratedTask:
        """Unrelated vectors: no op composition explains them."""
        pairs = [(self.sample_input(rng), self.sample_input(rng)) for _ in range(self.n_demos + 1)]
        return GeneratedTask(self.domain, DECOY, pairs[:-1], pairs[-1][0], pairs[-1][1])


def _mat(ops: Sequence[Any]) -> np.ndarray:
    """Matrix product of rotation/linear ops, rounded to the exact lattice."""
    m = np.eye(3)
    for op in ops:
        m = np.round(op.run(m), 9)
    return m


def _key(m: np.ndarray) -> Tuple[float, ...]:
    return tuple(np.round(np.asarray(m, dtype=np.float64), 6).ravel().tolist())


def _unkey(k: Tuple[float, ...]) -> np.ndarray:
    return np.asarray(k, dtype=np.float64).reshape(3, 3)


def vector_gravity(v: Any) -> np.ndarray:
    """Fall to the +y axis: drop every horizontal component (outside the
    vector3d vocabulary, which can zero only one axis at a time)."""
    out = np.zeros(3, dtype=np.float64)
    out[1] = np.asarray(v, dtype=np.float64)[1]
    return out
