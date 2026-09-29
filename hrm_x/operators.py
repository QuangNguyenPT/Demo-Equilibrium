"""Group-action operators on grids (doc 03, section 4).

Global operators are elements (R, sigma) of D4 x Sym(colours): R acts on centred cell coordinates,
sigma on colours. The set is CLOSED under composition, so a demo produced by any word of generators
(rot90, flips, transpose, colour swaps) is explained by ONE group element: identification cost does
not grow with the word length (the abstraction argument, measured in E6).
Object operators assign each object o a group element g_o = (R_o about its own box centre, t_o, sigma_o)
chosen by a predicate of o; objects with the same predicate value must receive the same element
(determinacy): an object whose predicate value never occurred in the demos is UNDETERMINED.
Anything that is neither (gravity, fill, counting) fails to fit and yields an abstention.

Where this sits in the repo: this is the v0.5 GRID ROUTE. The older route in `hrm_x.domains.grid`
searches over a program of selector/action steps (`candidate_steps` x `fits`); this one solves the
same class of tasks in closed form by restricting the hypothesis space to a group. Only 8 global
candidates (D4) x one colour bijection, and for objects a small product of per-object element
candidates, are ever enumerated, so `solve_grid` returns a calibrated answer or abstains without
any search. Both routes live side by side; `infer_global` / `infer_objects` / `solve_grid` are the
v0.5 entry points and `GridAnswer.status` ("answer" / "abstain") is the decision it reports.

Tensor contract: grids are np.ndarray int64 (H, W), values 0..9, 0 = background.
Component labelling is NOT done here: `objects` delegates to the single 4-connected component
finder in `hrm_x.domains.grid`, so the repo has one notion of "object" and one implementation.

Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from hrm_x.domains.grid import objects as _grid_objects

Grid = np.ndarray
D4: List[np.ndarray] = []
for k in range(4):
    for f in (False, True):
        m = np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), k)
        if f:
            m = m @ np.array([[1, 0], [0, -1]])
        D4.append(m.astype(np.int64))
D4_NAMES = [f"rot{90 * k}{'+flip' if f else ''}" for k in range(4) for f in (False, True)]


def apply_global(R: np.ndarray, sigma: Dict[int, int], g: Grid) -> Grid:
    """Exact action of (R, sigma): coordinates about the grid centre, output shape = R . shape."""
    H, W = g.shape
    corners = np.array([[0, 0], [0, W - 1], [H - 1, 0], [H - 1, W - 1]]) @ R.T
    lo = corners.min(0)
    shape = corners.max(0) - lo + 1
    # np.indices, not np.nindices: the latter does not exist in numpy and the shipped PoC
    # therefore could not execute apply_global at all. Same (rr, cc) index arrays either way.
    rr, cc = np.indices(g.shape)
    out = np.zeros(tuple(shape), dtype=np.int64)
    pts = np.stack([rr.ravel(), cc.ravel()], 1) @ R.T - lo
    out[pts[:, 0], pts[:, 1]] = [sigma.get(int(v), int(v)) for v in g.ravel()]
    return out


def infer_global(demos: Sequence[Tuple[Grid, Grid]]) -> Tuple[List[Tuple[int, Dict[int, int]]], int]:
    """All (R index, sigma) consistent with every demo; sigma must be a bijection on colours seen.
    Cost: 8 candidates x cells, independent of how many generator steps produced the demos."""
    out, ops = [], 0
    for i, R in enumerate(D4):
        sigma: Dict[int, int] = {}
        ok = True
        for x, y in demos:
            ops += x.size
            z = apply_global(R, {}, x)
            if z.shape != y.shape:
                ok = False; break
            for a, b in zip(z.ravel(), y.ravel()):
                a, b = int(a), int(b)
                if sigma.setdefault(a, b) != b:
                    ok = False; break
            if not ok:
                break
        if ok and len(set(sigma.values())) == len(sigma):
            out.append((i, sigma))
    return out, ops


# ---------------- objects ----------------

@dataclass(frozen=True)
class Obj:
    color: int
    cells: FrozenSet[Tuple[int, int]]

    @property
    def box(self) -> Tuple[int, int, int, int]:
        r = [p[0] for p in self.cells]; c = [p[1] for p in self.cells]
        return min(r), min(c), max(r), max(c)

    @property
    def size(self) -> int:
        return len(self.cells)

    def shape_key(self) -> FrozenSet[Tuple[int, int]]:
        r0, c0, _, _ = self.box
        return frozenset((r - r0, c - c0) for r, c in self.cells)


def objects(g: Grid) -> List[Obj]:
    """4-connected same-colour components, rewrapped with `shape_key`.

    The labelling itself is the repo's single component finder (`hrm_x.domains.grid.objects`); this
    module only adds the shape-insensitive key the group route needs. Ordering is reading order of
    each component's first cell, which is what the candidate enumeration below relies on.
    """
    return [Obj(o.color, o.cells) for o in _grid_objects(g)]


def transform_obj(o: Obj, Ri: int, t: Tuple[int, int], color: int) -> Obj:
    """R about the object's own box (min corner kept), then translation t, then recolour."""
    R = D4[Ri]
    pts = np.array(sorted(o.shape_key())) @ R.T
    pts -= pts.min(0)
    r0, c0, _, _ = o.box
    return Obj(color, frozenset((int(r + r0 + t[0]), int(c + c0 + t[1])) for r, c in pts))


PREDICATES = {
    "all": lambda o, objs: "all",
    "color": lambda o, objs: o.color,
    "is_largest": lambda o, objs: o.size == max(p.size for p in objs),
    "is_smallest": lambda o, objs: o.size == min(p.size for p in objs),
}
Element = Tuple[int, Tuple[int, int], Optional[int]]      # (R index, translation, new colour or None=keep)
DELETE: Element = (-1, (0, 0), None)


def _elements_for(o: Obj, outs: List[Obj], H: int, W: int) -> List[Element]:
    """Every group element (R, t, colour) mapping o onto some output object, or DELETE if none can."""
    cands = []
    for q in outs:
        for Ri in range(8):
            z = transform_obj(o, Ri, (0, 0), q.color)
            if z.shape_key() != q.shape_key():
                continue
            t = (q.box[0] - z.box[0], q.box[1] - z.box[1])
            cands.append((Ri, t, q.color if q.color != o.color else None))
    return cands or [DELETE]


def render(objs: List[Obj], shape: Tuple[int, int]) -> Optional[Grid]:
    g = np.zeros(shape, dtype=np.int64)
    for o in objs:
        for r, c in o.cells:
            if not (0 <= r < shape[0] and 0 <= c < shape[1]) or g[r, c]:
                return None                                  # leaves the grid or collides: not applicable
            g[r, c] = o.color
    return g


def apply_objects(rule: Dict, pred: str, g: Grid) -> Optional[Grid]:
    """Apply a predicate-conditioned element map; None if some object's predicate value is undetermined."""
    objs = objects(g)
    out = []
    for o in objs:
        key = PREDICATES[pred](o, objs)
        if key not in rule:
            return None                                      # determinacy: never seen -> no opinion
        Ri, t, col = rule[key]
        if Ri == -1:
            continue
        out.append(transform_obj(o, Ri, t, o.color if col is None else col))
    return render(out, g.shape)


def infer_objects(demos: Sequence[Tuple[Grid, Grid]], max_rules: int = 64) -> Tuple[List[Tuple[str, Dict]], int]:
    """Rules pred -> element consistent with all demos (output reproduced exactly). Search is over
    per-object candidate elements, constrained so equal predicate values share an element."""
    ops = 0
    found: List[Tuple[str, Dict]] = []
    for pred in PREDICATES:
        options: Dict = {}
        ok = True
        for x, y in demos:
            if x.shape != y.shape:
                ok = False; break
            xo, yo = objects(x), objects(y)
            for o in xo:
                key = PREDICATES[pred](o, xo)
                cands = set(_elements_for(o, yo, *x.shape))
                ops += len(yo) * 8
                options[key] = cands if key not in options else options[key] & cands
                if not options[key]:
                    ok = False; break
            if not ok:
                break
        if not ok:
            continue
        keys = sorted(options, key=str)
        for combo in itertools.islice(itertools.product(*[sorted(options[k], key=str) for k in keys]), max_rules):
            rule = dict(zip(keys, combo))
            ops += sum(x.size for x, _ in demos)
            if all((z := apply_objects(rule, pred, x)) is not None and np.array_equal(z, y) for x, y in demos):
                found.append((pred, rule))
    return found, ops


@dataclass
class GridAnswer:
    status: str
    grid: Optional[Grid]
    readings: int
    ops: int
    reason: str


def _bits_sigma(sigma: Dict[int, int]) -> float:
    return sum(2 * np.log2(10) for a, b in sigma.items() if a != b)


def _bits_element(e: Element) -> float:
    Ri, (dr, dc), col = e
    if Ri == -1:
        return 1.0
    return (1.0 if Ri == 0 else 3.0) + 2.0 + np.log2(1 + abs(dr)) + np.log2(1 + abs(dc)) + (0.5 if col is None else np.log2(10))


def solve_grid(demos: Sequence[Tuple[Grid, Grid]], test: Grid, use_global: bool = True, use_objects: bool = True,
               bits_bottom: float = 12.0, wrong_penalty: float = 4.0) -> GridAnswer:
    """Readings that reproduce every demo exactly are weighted 2^-bits (MDL; all fit the demos equally).
    A reading undefined on the test input has no opinion (determinacy). BOTTOM keeps 2^-bits_bottom.
    Answer y if its posterior mass q(y) > L/(1+L); otherwise abstain with the reason."""
    readings: List[Tuple[float, Optional[Grid], str]] = []
    ops = 0
    if use_global:
        gl, o1 = infer_global(demos); ops += o1
        for Ri, sigma in gl:
            b = 1.0 + np.log2(8) + _bits_sigma(sigma)
            defined = all(int(v) in sigma for v in np.unique(test))   # sigma undetermined on unseen colours
            readings.append((b, apply_global(D4[Ri], sigma, test) if defined else None, f"global {D4_NAMES[Ri]}"))
    if use_objects:
        ob, o2 = infer_objects(demos); ops += o2
        for pred, rule in ob:
            b = 1.0 + np.log2(len(PREDICATES)) + sum(_bits_element(e) for e in rule.values())
            readings.append((b, apply_objects(rule, pred, test), f"objects by {pred}"))
    if not readings:
        return GridAnswer("abstain", None, 0, ops, "no group-action reading fits the demos")
    # MDL: an exact reading pays only its own length; BOTTOM must also encode the demo outputs
    # (log2 10 bits per output cell), so it dominates only when every reading is longer than the data.
    data_bits = sum(y.size for _, y in demos) * np.log2(10)
    m = min(b for b, _, _ in readings)
    ws = [2.0 ** -(b - m) for b, _, _ in readings]
    z = sum(ws) + 2.0 ** -(bits_bottom + data_bits - m)
    q: Dict[bytes, Tuple[float, Grid]] = {}
    for w, (_, g, _) in zip(ws, readings):
        if g is not None:
            k = g.tobytes() + bytes(g.shape)
            q[k] = (q.get(k, (0.0, g))[0] + w / z, g)
    if not q:
        return GridAnswer("abstain", None, len(readings), ops, "readings fit the demos but are undetermined on this input")
    best_q, best_g = max(q.values(), key=lambda t: t[0])
    if best_q > wrong_penalty / (1 + wrong_penalty):
        return GridAnswer("answer", best_g, len(readings), ops, f"dominant reading q={best_q:.2f}")
    return GridAnswer("abstain", None, len(readings), ops, f"{len(q)} predictions, best q={best_q:.2f}")
