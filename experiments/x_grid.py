"""E6 rule composition, E7 structural transformation on new arrangements, E11 symbol permutation,
E12 (part) search guidance from solved tasks.

Hypothesis: an operator inferred over objects transfers to new arrangements (more objects,
bigger grids) and composes into two-step rules never shown as tasks; a pixel mapping does not.
Falsified if HRM-X loses accuracy on new arrangements / compositions, or the pixel baseline
matches it, or accuracy changes under a colour permutation.

Everything that produces answers here is independent of hrm_x: its own scenes, its own flood
fill, its own implementation of each transformation (numpy), and confound checks so that the
demos identify the rule (e.g. the largest object is never also the only one of its colour).
  python -m experiments.x_grid
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from experiments.common import record, table
from hrm_x.domains.grid import GridTask, Grids, action_kind, candidate_steps, fits, step_size
from hrm_x.mind import Mind

SEED = 0
Grid = np.ndarray


# ---------- independent scene model ----------

def blobs(g: Grid) -> List[Tuple[int, List[Tuple[int, int]]]]:
    """Own 4-connected components: (colour, cells), in reading order."""
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if g[r, c] and not seen[r, c]:
                col, todo, cells = g[r, c], [(r, c)], []
                seen[r, c] = True
                while todo:
                    y, x = todo.pop()
                    cells.append((y, x))
                    for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] == col:
                            seen[ny, nx] = True
                            todo.append((ny, nx))
                out.append((int(col), cells))
    return out


def scene(rng: random.Random, h: int, w: int, count: int, colours: List[int]) -> Optional[Grid]:
    g = np.zeros((h, w), dtype=np.int64)
    for _ in range(count):
        for _attempt in range(60):
            hh, ww = rng.randint(1, 3), rng.randint(1, 3)
            r, c = rng.randint(0, h - hh), rng.randint(0, w - ww)
            if g[max(0, r - 1):r + hh + 1, max(0, c - 1):c + ww + 1].any():
                continue                                   # keep a one-cell gap: objects never touch
            shape = np.ones((hh, ww), dtype=bool)
            if hh * ww > 2 and rng.random() < 0.4:
                shape[rng.randrange(hh), rng.randrange(ww)] = False
                if len(blobs(shape.astype(int))) != 1:
                    shape[:] = True
            g[r:r + hh, c:c + ww][shape] = rng.choice(colours)
            break
        else:
            return None
    return g if len(blobs(g)) == count else None


def paint(h: int, w: int, parts: List[Tuple[int, List[Tuple[int, int]]]]) -> Optional[Grid]:
    out = np.zeros((h, w), dtype=np.int64)
    for col, cells in parts:
        for r, c in cells:
            if not (0 <= r < h and 0 <= c < w) or (out[r, c] and out[r, c] != col):
                return None
            out[r, c] = col
    return out


def sizes(parts):
    return [len(c) for _, c in parts]


# ---------- transformations (own implementations; params drawn per task) ----------

def t_recolor_all(p):      return lambda g: np.where(g > 0, p["to"], 0)
def t_recolor_colour(p):   return lambda g: np.where(g == p["from"], p["to"], g)
def t_delete_colour(p):    return lambda g: np.where(g == p["from"], 0, g)
def t_flip_h(p):           return lambda g: g[:, ::-1].copy()
def t_flip_v(p):           return lambda g: g[::-1, :].copy()
def t_rot90(p):            return lambda g: np.rot90(g, -1).copy()


def t_move_all(p):
    def f(g):
        parts = [(col, [(r + p["dr"], c + p["dc"]) for r, c in cells]) for col, cells in blobs(g)]
        return paint(*g.shape, parts)
    return f


def _by_size(g, pick):
    parts = blobs(g)
    s = sizes(parts)
    target = max(s) if pick == "largest" else min(s)
    if s.count(target) != 1:
        return None, None
    return parts, s.index(target)


def t_recolor_largest(p):
    def f(g):
        parts, k = _by_size(g, "largest")
        if parts is None:
            return None
        parts[k] = (p["to"], parts[k][1])
        return paint(*g.shape, parts)
    return f


def t_delete_smallest(p):
    def f(g):
        parts, k = _by_size(g, "smallest")
        return None if parts is None else paint(*g.shape, parts[:k] + parts[k + 1:])
    return f


def t_keep_largest(p):
    def f(g):
        parts, k = _by_size(g, "largest")
        return None if parts is None else paint(*g.shape, [parts[k]])
    return f


def t_flip_objects_h(p):
    def f(g):
        out = []
        for col, cells in blobs(g):
            c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
            out.append((col, [(r, c1 - (c - c0)) for r, c in cells]))
        return paint(*g.shape, out)
    return f


SINGLE: Dict[str, Callable] = {
    "recolor_all": t_recolor_all, "recolor_colour": t_recolor_colour, "delete_colour": t_delete_colour,
    "move_all": t_move_all, "recolor_largest": t_recolor_largest, "delete_smallest": t_delete_smallest,
    "keep_largest": t_keep_largest, "flip_objects_h": t_flip_objects_h, "flip_h": t_flip_h, "flip_v": t_flip_v,
    "rot90": t_rot90,
}
COMPOSED = [("move_all", "recolor_colour"), ("keep_largest", "flip_h"), ("delete_smallest", "move_all"),
            ("recolor_largest", "rot90"), ("flip_objects_h", "delete_colour")]


def params(rng: random.Random, name: str, colours: List[int]) -> Dict:
    a, b = rng.sample(colours, 2)
    return {"to": b if name != "recolor_all" else rng.choice([c for c in range(1, 10) if c not in colours]),
            "from": a, "dr": rng.choice([-2, -1, 1, 2]), "dc": rng.choice([-2, -1, 0, 1, 2])}


def confounded(name: str, p: Dict, grids: List[Grid]) -> bool:
    """True if a simpler selector explains the demos equally well (the rule would not be identified)."""
    for g in grids:
        parts = blobs(g)
        s = sizes(parts)
        if name in ("recolor_largest", "keep_largest") and s.count(max(s)) == 1:
            biggest = parts[s.index(max(s))][0]
            if sum(col == biggest for col, _ in parts) > 1:
                return False                       # largest is not "the only one of its colour" here
        if name == "delete_smallest" and s.count(min(s)) == 1:
            small = parts[s.index(min(s))][0]
            if sum(col == small for col, _ in parts) > 1:
                return False
        if name in ("recolor_colour", "delete_colour"):
            targets = [len(c) for col, c in parts if col == p["from"]]
            if targets and len(targets) != len(parts) and max(targets) != max(s) and min(targets) != min(s):
                return False                       # colour is not also "largest"/"smallest"
    return name in ("recolor_largest", "keep_largest", "delete_smallest", "recolor_colour", "delete_colour")


def make_task(rng: random.Random, names: Tuple[str, ...], novel: bool) -> Optional[Tuple[GridTask, Grid, str]]:
    colours = rng.sample(range(1, 10), 4)
    fns = []
    for name in names:
        p = params(rng, name, colours)
        fns.append((name, p, SINGLE[name](p)))

    def apply(g):
        for _, _, f in fns:
            if g is None:
                return None
            g = f(g)
        return g

    for _attempt in range(200):
        demos = []
        for _ in range(3):
            g = scene(rng, rng.randint(8, 10), rng.randint(8, 10), rng.randint(2, 4), colours)
            if g is None:
                break
            y = apply(g)
            if y is None or np.array_equal(y, g):
                break
            demos.append((g, y))
        if len(demos) < 3 or any(confounded(n, p, [x for x, _ in demos]) for n, p, _ in fns):
            continue
        big = (rng.randint(11, 14), rng.randint(11, 14)) if novel else (rng.randint(8, 10), rng.randint(8, 10))
        test = scene(rng, *big, rng.randint(4, 6) if novel else rng.randint(2, 4), colours)
        truth = None if test is None else apply(test)
        if truth is None:
            continue
        return GridTask(demos, test), truth, "+".join(names)
    return None


# ---------- baselines ----------

def pixel_lut(task: GridTask) -> Optional[Grid]:
    """Per-colour lookup table learned from the demos (the transductive pixel baseline)."""
    lut: Dict[int, int] = {}
    for x, y in task.demos:
        if x.shape != y.shape:
            return None
        for a, b in zip(x.ravel(), y.ravel()):
            if lut.setdefault(int(a), int(b)) != int(b):
                return None
    return np.vectorize(lambda v: lut.get(int(v), int(v)))(task.test)


class WholeGridOnly(Grids):
    """Ablation: no objects -- only whole-grid symmetries and recolouring everything."""

    def propose(self, task, level, mind):
        import hrm_x.domains.grid as G
        original = G.candidate_steps
        G.candidate_steps = lambda demos: [s for s in original(demos) if s[0] == "grid" or s[0] == ("all",)]
        try:
            return super().propose(task, level, mind)
        finally:
            G.candidate_steps = original


def solve(domain, task: GridTask, mind: Optional[Mind] = None):
    r = (mind or Mind()).solve(domain, task)
    return r


def permute(task: GridTask, truth: Grid, rng: random.Random) -> Tuple[GridTask, Grid]:
    perm = np.array([0] + rng.sample(range(1, 10), 9))
    f = lambda g: perm[g]
    return GridTask([(f(x), f(y)) for x, y in task.demos], f(task.test)), f(truth)


def correct(r, truth) -> bool:
    return r.status == "answer" and r.answer == tuple(map(tuple, truth))


def prior_rank(task: GridTask, prior_mind: Mind, dom: Grids) -> int:
    """How many single-step programs a first-fit search would try before a consistent one."""
    prior = dom._prior(prior_mind)
    steps = sorted(candidate_steps(task.demos), key=lambda s: (-prior[action_kind(s)], step_size(s)))
    for k, s in enumerate(steps, start=1):
        if fits((s,), task.demos):
            return k
    return len(steps) + 1


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--name", default="x_grid")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows, metrics = [], {}
    suites = {
        "single, same-size test": [(n,) for n in SINGLE for _ in range(6)],
        "single, new arrangement (bigger grid, more objects)": [(n,) for n in SINGLE for _ in range(6)],
        "composition (never a task)": [c for c in COMPOSED for _ in range(8)],
    }
    for suite, specs in suites.items():
        novel = "new" in suite or "composition" in suite
        tasks = [t for t in (make_task(rng, s, novel) for s in specs) if t is not None]
        stats = {"hrmx": 0, "hrmx_wrong": 0, "lut": 0, "whole_grid": 0, "permuted": 0, "ops": []}
        per_family: Dict[str, List[int]] = {}
        for task, truth, name in tasks:
            r = solve(Grids(), task)
            ok = correct(r, truth)
            stats["hrmx"] += ok
            stats["hrmx_wrong"] += r.status == "answer" and not ok
            stats["ops"].append(r.ops)
            per_family.setdefault(name, []).append(int(ok))
            lut = pixel_lut(task)
            stats["lut"] += lut is not None and np.array_equal(lut, truth)
            stats["whole_grid"] += correct(solve(WholeGridOnly(), task), truth)
            ptask, ptruth = permute(task, truth, rng)
            stats["permuted"] += correct(solve(Grids(), ptask), ptruth)
        n = len(tasks)
        m = {"n": n, "hrmx": stats["hrmx"] / n, "hrmx_wrong_answers": stats["hrmx_wrong"] / n,
             "pixel_lut": stats["lut"] / n, "whole_grid_only": stats["whole_grid"] / n,
             "hrmx_colour_permuted": stats["permuted"] / n, "hrmx_mean_ops": float(np.mean(stats["ops"])),
             "per_family": {k: sum(v) / len(v) for k, v in per_family.items()}}
        metrics[suite] = m
        rows.append([suite, n, f"{m['hrmx']:.3f}", f"{m['hrmx_wrong_answers']:.3f}", f"{m['pixel_lut']:.3f}",
                     f"{m['whole_grid_only']:.3f}", f"{m['hrmx_colour_permuted']:.3f}", f"{m['hrmx_mean_ops']:.0f}"])
    print(table(rows, ["suite", "n", "HRM-X", "HRM-X wrong", "pixel LUT", "no objects", "colour-permuted", "ops"]))
    for suite, m in metrics.items():
        weak = {k: v for k, v in m["per_family"].items() if v < 1.0}
        if weak:
            print(f"[FAMILIES] {suite}: below 1.0 -> {weak}")

    # E12: does experience make search cheaper? rank of the first consistent program, prior vs uniform
    learner, dom = Mind(), Grids()
    stream = [t for t in (make_task(rng, (rng.choice(["recolor_colour", "move_all", "delete_colour"]),), False)
                          for _ in range(40)) if t is not None]
    ranks_prior, ranks_uniform = [], []
    for task, truth, _ in stream:
        ranks_uniform.append(prior_rank(task, Mind(), dom))
        ranks_prior.append(prior_rank(task, learner, dom))
        learner.solve(dom, task)
    half = len(stream) // 2
    metrics["E12 search rank (programs tried before a fit), second half of stream"] = {
        "uniform_prior": float(np.mean(ranks_uniform[half:])), "learned_prior": float(np.mean(ranks_prior[half:]))}
    print(f"[E12] first-fit rank, 2nd half of a 40-task stream: uniform {np.mean(ranks_uniform[half:]):.1f} "
          f"-> learned prior {np.mean(ranks_prior[half:]):.1f}")
    record(args.name, {"seed": args.seed, "demos": 3, "demo_grids": "8-10 square, 2-4 objects",
                      "novel_test": "11-14, 4-6 objects", "families": list(SINGLE), "compositions": COMPOSED,
                      "confound_control": "demos regenerated until the property selector is identified"}, metrics)


if __name__ == "__main__":
    main()
