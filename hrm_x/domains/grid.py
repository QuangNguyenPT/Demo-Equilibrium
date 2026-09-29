"""Grids: pixels -> objects -> an object-level program induced from the demos -> exact rendering.

Perception is exact: 4-connected same-color components (background 0) are the objects, with
color, cells, size and bounding box. The operator language acts on objects, not pixels:
a step is (selector, action) or a whole-grid symmetry; a program is one or two steps. Because
programs name *which* objects and *what* happens to them, the same program applies to a new
arrangement of new objects (the property a pixel-level mapping lacks).
Hypotheses are the programs that reproduce every demo exactly, grouped by the output they
predict for the test input (observational equivalence on the question), weighted by
2^-size x a prior learned from previously solved tasks (which actions tend to occur).
Verification checks applicability on the new input: every step was exercised by the demos,
singleton selectors pick a unique object, moves stay inside the grid and do not collide.
"""
from __future__ import annotations

import itertools
from collections import Counter as Tally
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from hrm_x import native
from hrm_x.mind import Hyp, Mind, Result

Grid = np.ndarray
Cell = Tuple[int, int]


@dataclass(frozen=True)
class Obj:
    color: int
    cells: FrozenSet[Cell]

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def box(self) -> Tuple[int, int, int, int]:
        rows = [r for r, _ in self.cells]
        cols = [c for _, c in self.cells]
        return min(rows), min(cols), max(rows), max(cols)


def objects(grid: Grid) -> List[Obj]:
    """4-connected same-color components, in reading order of their first cell."""
    seen = np.zeros(grid.shape, dtype=bool)
    out = []
    for r, c in itertools.product(range(grid.shape[0]), range(grid.shape[1])):
        if seen[r, c] or grid[r, c] == 0:
            continue
        color, stack, cells = int(grid[r, c]), [(r, c)], set()
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.add((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and not seen[ny, nx] and grid[ny, nx] == color:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append(Obj(color, frozenset(cells)))
    return out


Step = Tuple
GLOBAL = ("flip_h", "flip_v", "rot90", "rot180", "rot270", "transpose")


def select(selector: Tuple, objs: List[Obj]) -> Tuple[List[bool], bool]:
    """(mask over objects, ambiguous?) for a selector; largest/smallest are ambiguous on ties."""
    kind = selector[0]
    if kind == "all":
        return [True] * len(objs), False
    if kind == "color":
        return [o.color == selector[1] for o in objs], False
    if not objs:
        return [], False
    sizes = [o.size for o in objs]
    target = max(sizes) if kind == "largest" else min(sizes)
    mask = [s == target for s in sizes]
    return mask, sum(mask) > 1


class Invalid(Exception):
    pass


# How strictly a step is checked: LENIENT (search), EXERCISED (a demo must select something,
# unambiguously, without collisions), APPLICABLE (the test input: unambiguous and no collisions,
# but a rule like "delete the red objects" may apply vacuously).
LENIENT, EXERCISED, APPLICABLE = 0, 1, 2


def apply_step(step: Step, grid: Grid, strict: int = LENIENT) -> Grid:
    """One step. Object steps run in C++ when the native backend is enabled (same results, tested)."""
    if step[0] != "grid" and native.enabled():
        out = native.object_step(step[0], step[1], grid, strict)
        if out is None:
            raise Invalid("step does not apply")
        return out
    return apply_step_reference(step, grid, strict)


def apply_step_reference(step: Step, grid: Grid, strict: int = LENIENT) -> Grid:
    """Reference semantics of a step (the blueprint the native kernel is checked against)."""
    if step[0] == "grid":
        op = step[1]
        return {"flip_h": lambda g: g[:, ::-1], "flip_v": lambda g: g[::-1, :], "rot90": lambda g: np.rot90(g, -1),
                "rot180": lambda g: np.rot90(g, 2), "rot270": lambda g: np.rot90(g, 1),
                "transpose": lambda g: g.T}[op](grid).copy()
    selector, action = step
    objs = objects(grid)
    mask, ambiguous = select(selector, objs)
    if strict and (ambiguous or (strict == EXERCISED and not any(mask))):
        raise Invalid("selector is ambiguous, or selects nothing in a demo")
    kind = action[0]
    out = np.zeros_like(grid)
    placed: Dict[Cell, int] = {}

    def paint(cells, color) -> None:
        for (r, c) in cells:
            if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                raise Invalid("object leaves the grid")
            if (r, c) in placed and placed[(r, c)] != color and strict:
                raise Invalid("objects collide")
            placed[(r, c)] = color
            out[r, c] = color

    for o, chosen in zip(objs, mask):
        if kind == "keep":
            if chosen:
                paint(o.cells, o.color)
            continue
        if not chosen:
            paint(o.cells, o.color)
            continue
        if kind == "delete":
            continue
        if kind == "recolor":
            paint(o.cells, action[1])
        elif kind == "move":
            paint({(r + action[1], c + action[2]) for r, c in o.cells}, o.color)
        elif kind == "flip":
            r0, c0, r1, c1 = o.box
            cells = {(r, c1 - (c - c0)) for r, c in o.cells} if action[1] == "h" else {(r1 - (r - r0), c) for r, c in o.cells}
            paint(cells, o.color)
    return out


def run(program: Tuple[Step, ...], grid: Grid, strict: int = LENIENT) -> Optional[Grid]:
    try:
        for step in program:
            grid = apply_step(step, grid, strict)
        return grid
    except Invalid:
        return None


def step_size(step: Step) -> int:
    if step[0] == "grid":
        return 2
    selector, action = step
    # description length: "all" has no condition; a colour needs a parameter; largest/smallest need
    # a comparison over every object (an argmax), so they cost more than "all"
    return {"all": 1, "largest": 2, "smallest": 2, "color": 2}[selector[0]] + \
        {"keep": 1, "delete": 1, "recolor": 2, "flip": 2, "move": 3}[action[0]]


def action_kind(step: Step) -> str:
    return step[1] if step[0] == "grid" else step[1][0]


def candidate_steps(demos: Sequence[Tuple[Grid, Grid]]) -> List[Step]:
    in_colors = sorted({int(v) for x, _ in demos for v in np.unique(x) if v})
    out_colors = sorted({int(v) for _, y in demos for v in np.unique(y) if v})
    selectors = [("all",), ("largest",), ("smallest",)] + [("color", c) for c in in_colors]
    actions = [("keep",), ("delete",), ("flip", "h"), ("flip", "v")] + [("recolor", c) for c in out_colors] + \
        [("move", dr, dc) for dr in range(-3, 4) for dc in range(-3, 4) if (dr, dc) != (0, 0)]
    return [("grid", g) for g in GLOBAL] + [(s, a) for s in selectors for a in actions]


def fits(program: Tuple[Step, ...], demos: Sequence[Tuple[Grid, Grid]]) -> bool:
    for x, y in demos:
        out = run(program, x)
        if out is None or out.shape != y.shape or not np.array_equal(out, y):
            return False
    return True


def distance(program: Tuple[Step, ...], demos: Sequence[Tuple[Grid, Grid]]) -> float:
    total = 0.0
    for x, y in demos:
        out = run(program, x)
        if out is None or out.shape != y.shape:
            return float("inf")
        total += float((out != y).mean())
    return total


@dataclass
class GridTask:
    demos: List[Tuple[Grid, Grid]]
    test: Grid


def as_key(grid: Grid) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in grid)


class Grids:
    name = "grid"

    def __init__(self, beam: int = 24) -> None:
        self.beam = beam
        self.evaluated = 0

    def _prior(self, mind: Mind) -> Dict[str, float]:
        usage: Tally = mind.knowledge.setdefault(self.name, {"usage": Tally()})["usage"]
        kinds = ["keep", "delete", "flip", "recolor", "move"] + list(GLOBAL)
        total = sum(usage.values()) + len(kinds)
        return {k: (usage[k] + 1) / total for k in kinds}

    def propose_cost(self, task: GridTask, level: int) -> float:
        return 400.0 if level == 0 else 12000.0

    def propose(self, task: GridTask, level: int, mind: Mind) -> Optional[Tuple[List[Hyp], int]]:
        if level > 1:
            return None
        prior = self._prior(mind)
        steps = sorted(candidate_steps(task.demos), key=lambda s: (-prior[action_kind(s)], step_size(s)))
        evaluated = 0
        found: List[Tuple[Tuple[Step, ...], float]] = []
        if level == 0:
            programs = [(s,) for s in steps]
        else:
            ranked = sorted(steps, key=lambda s: distance((s,), task.demos))[: self.beam]
            evaluated += len(steps)
            programs = [(a, b) for a in ranked for b in steps if a != b]
        for program in programs:
            evaluated += 1
            if fits(program, task.demos):
                size = sum(step_size(s) for s in program)
                weight = 2.0 ** -size
                for s in program:
                    weight *= prior[action_kind(s)]
                found.append((program, weight))
        self.evaluated += evaluated
        groups: Dict[Tuple, List[Tuple[Tuple[Step, ...], float]]] = {}
        for program, weight in found:
            out = run(program, task.test)
            key = ("none",) if out is None else as_key(out)
            groups.setdefault(key, []).append((program, weight))
        hyps = []
        for key, members in groups.items():
            members.sort(key=lambda m: -m[1])
            program = members[0][0]
            hyp = Hyp(" ; ".join(map(str, program)), sum(w for _, w in members), (program, key),
                      note=f"{len(members)} program(s)")
            # the prediction was executed above, so the hypothesis is already derived
            hyp.derived, hyp.answer = True, None if key == ("none",) else key
            hyps.append(hyp)
        return hyps, evaluated

    def derive_cost(self, task: GridTask, hyp: Hyp, mind: Mind) -> float:
        return float(task.test.size)

    def derive(self, task: GridTask, hyp: Hyp, mind: Mind) -> int:
        program, key = hyp.payload
        hyp.derived = True
        hyp.answer = None if key == ("none",) else key
        if hyp.answer is None:
            hyp.note = "program does not apply to the test input"
        return task.test.size

    def verify(self, task: GridTask, hyp: Hyp, mind: Mind) -> int:
        program, _ = hyp.payload
        problems = []
        if run(program, task.test, strict=APPLICABLE) is None:
            problems.append("on the test input a selector is ambiguous, or objects leave the grid or collide")
        # Support: every step must act in at least one demo (vacuous in others is consistent with
        # the rule) and must be unambiguous and collision-free in all of them.
        exercised = [step[0] == "grid" for step in program]
        for i, (x, _) in enumerate(task.demos):
            grid = x
            for k, step in enumerate(program):
                if step[0] != "grid":
                    exercised[k] |= any(select(step[0], objects(grid))[0])
                try:
                    grid = apply_step(step, grid, APPLICABLE)
                except Invalid:
                    problems.append(f"demo {i}: step {k + 1} is ambiguous or collides")
                    break
        problems += [f"step {k + 1} never acts in any demo (no evidence for it)"
                     for k, done in enumerate(exercised) if not done]
        hyp.verified = not problems
        if problems:
            hyp.note = "; ".join(problems)
        return task.test.size * len(program) * (len(task.demos) + 1)

    def questions(self, task, hyps, mind):
        return []

    def ask(self, task, question, hyps, mind) -> int:
        return 0

    def remember(self, task: GridTask, result: Result, mind: Mind) -> None:
        if result.status != "answer":
            return
        usage: Tally = mind.knowledge.setdefault(self.name, {"usage": Tally()})["usage"]
        for h in result.hypotheses:
            if h.verified and h.answer == result.answer:
                for s in h.payload[0]:
                    usage[action_kind(s)] += 1
                break
