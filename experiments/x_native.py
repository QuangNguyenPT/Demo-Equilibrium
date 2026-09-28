"""Native backend speedup on the profiled workloads (the justification for C++, spec v0.4 §23).

Same workloads under HRMX_BACKEND=python and =native; answers must be identical, time is compared.
  python -m experiments.x_native
"""
from __future__ import annotations

import os
import random
import time
from typing import Callable, List

import numpy as np

from experiments.common import record, table
from hrm_x import native

REPEATS = 3


def scene(h, w, rects):
    g = np.zeros((h, w), int)
    for r, c, hh, ww, col in rects:
        g[r:r + hh, c:c + ww] = col
    return g


def grid_workload() -> List:
    from hrm_x.domains.grid import GridTask, Grids
    from hrm_x.mind import Mind

    def two(x):
        y = np.zeros_like(x)
        y[1:, :] = x[:-1, :]
        y[y == 2] = 7
        return y
    demos = [(x, two(x)) for x in [scene(10, 10, [(0, 0, 2, 2, 2), (3, 4, 3, 1, 1), (6, 6, 1, 1, 2)]),
                                   scene(10, 10, [(1, 1, 1, 3, 1), (4, 5, 2, 2, 2), (7, 0, 1, 1, 3)]),
                                   scene(10, 10, [(0, 5, 3, 1, 2), (5, 0, 1, 4, 3), (8, 8, 1, 1, 2)])]]
    test = scene(12, 12, [(2, 2, 2, 3, 2), (6, 6, 1, 1, 1), (9, 7, 1, 2, 2)])
    r = Mind().solve(Grids(), GridTask(demos, test))
    return [r.answer, r.ops]


def arith_workload() -> List:
    from hrm_x.domains.arith import Arith, ArithTask
    from hrm_x.mind import Mind
    mind, dom = Mind(), Arith()
    mind.solve(dom, ArithTask.from_text(["2 × 2 = 4", "2 × 3 = 6"], "10 × 3.1", oracle=lambda a, b: a * b))
    laws, _ = dom.consolidate(mind, "×", max_size=7, probes=16)
    rng = random.Random(0)
    answers = [mind.solve(dom, ArithTask.from_text([], f"{rng.randint(10, 3000)} × {rng.randint(1, 999) / 10:.1f}")).answer
               for _ in range(60)]
    return [len(laws), answers]


def timed(fn: Callable[[], List], backend: str):
    os.environ["HRMX_BACKEND"] = backend
    best, out = float("inf"), None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> None:
    if not native.available():
        raise SystemExit("[NATIVE] build first: .\\scripts\\build_native.ps1")
    rows, metrics = [], {}
    for name, fn in [("grid: 2-step induction", grid_workload), ("arith: laws + 60 questions", arith_workload)]:
        t_py, out_py = timed(fn, "python")
        t_nat, out_nat = timed(fn, "native")
        same = out_py == out_nat
        metrics[name] = {"python_s": t_py, "native_s": t_nat, "speedup": t_py / t_nat, "identical_results": same}
        rows.append([name, f"{t_py:.2f}", f"{t_nat:.2f}", f"{t_py / t_nat:.1f}x", same])
    os.environ["HRMX_BACKEND"] = "auto"
    print(table(rows, ["workload", "python s", "native s", "speedup", "identical"]))
    record("x_native", {"repeats": REPEATS, "timing": "best of repeats, perf_counter",
                        "build": "clang++ -O2 -std=c++17 (scripts/build_native.ps1)"}, metrics)


if __name__ == "__main__":
    main()
