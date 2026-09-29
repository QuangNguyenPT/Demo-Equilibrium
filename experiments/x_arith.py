"""E1 numeric extrapolation, E2 compositional arithmetic, E12 efficiency from discovered laws.

Hypothesis: a concept identified as a program over an exact substrate extrapolates exactly;
learned operators interpolate. Falsified if HRM-X loses accuracy out of range, or if a
neural operator matches it out of range.

Conditions (the truth is always Fraction multiplication, which HRM-X never calls):
  hrmx      : learns × from 2x2=4, 2x3=6 plus questions to the teacher; laws from one sleep
  knn       : 1-nearest-neighbour over the training table (pattern interpolation)
  mlp       : MLP on (a, b) -> a*b, trained on the training range (neural learned operator)
  log-mlp   : MLP on (log a, log b) -> log(a*b) (a neural operator with the right bias)
  exact     : hand-coded rational multiplication (upper bound, nothing learned)
Training range for the learned baselines: a in 1..20 integer, b in 0.1..20.0 (one decimal).
  python -m experiments.x_arith
"""
from __future__ import annotations

import math
import random
import time
from fractions import Fraction
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from experiments.common import record, table
from hrm_x.domains.arith import Arith, ArithTask
from hrm_x.mind import Mind

SEED = 0
Pair = Tuple[Fraction, Fraction]


def dec(x: float) -> Fraction:
    return Fraction(f"{x:.1f}")


def split(rng: random.Random, n: int, a_rng: Tuple[int, int], b_rng: Tuple[float, float]) -> List[Pair]:
    return [(Fraction(rng.randint(*a_rng)), dec(rng.uniform(*b_rng))) for _ in range(n)]


def exact_match(pred: float, truth: Fraction) -> bool:
    return abs(pred - float(truth)) <= 1e-6 * max(1.0, abs(float(truth)))


def train_mlp(train: List[Pair], log_space: bool, seed: int) -> Callable[[Pair], float]:
    torch.manual_seed(seed)
    x = torch.tensor([[float(a), float(b)] for a, b in train])
    y = torch.tensor([[float(a * b)] for a, b in train])
    if log_space:
        x, y = torch.log(x), torch.log(y)
        mx, sx, my, sy = x.mean(0), x.std(0), y.mean(), y.std()
    else:
        mx, sx, my, sy = x.mean(0), x.std(0), y.mean(), y.std()
    net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(3000):
        loss = ((net((x - mx) / sx) - (y - my) / sy) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    @torch.no_grad()
    def predict(pair: Pair) -> float:
        v = torch.tensor([[float(pair[0]), float(pair[1])]])
        v = torch.log(v) if log_space else v
        out = float(net((v - mx) / sx)[0, 0] * sy + my)
        return math.exp(out) if log_space else out
    return predict


def main() -> None:
    rng = random.Random(SEED)
    train = split(rng, 2000, (1, 20), (0.1, 20.0))
    tests = {
        "T1 in-range": split(rng, 200, (1, 20), (0.1, 20.0)),
        "T2 large integers": [(Fraction(rng.randint(100, 10000)), Fraction(rng.randint(100, 10000))) for _ in range(200)],
        "T3 large decimals": split(rng, 200, (21, 999), (20.1, 999.9)),
        "T4 10 x 3.1": [(Fraction(10), Fraction("3.1"))],
    }
    table_knn = {(a, b): a * b for a, b in train}
    keys = np.array([[float(a), float(b)] for a, b in table_knn])
    vals = list(table_knn.values())
    knn = lambda p: float(vals[int(np.argmin(((keys - np.array([float(p[0]), float(p[1])])) ** 2).sum(1)))])
    t0 = time.time()
    mlp = train_mlp(train, False, SEED)
    log_mlp = train_mlp(train, True, SEED)
    train_s = time.time() - t0

    mind, dom = Mind(), Arith()
    learn = mind.solve(dom, ArithTask.from_text(["2 × 2 = 4", "2 × 3 = 6"], "10 × 3.1", oracle=lambda a, b: a * b))
    laws, law_evals = dom.consolidate(mind, "×", max_size=7, probes=16)

    def text(v: Fraction) -> str:
        return str(v.numerator) if v.denominator == 1 else f"{float(v):.1f}"

    def hrmx(pair: Pair) -> Tuple[float, int]:
        r = mind.solve(dom, ArithTask.from_text([], f"{text(pair[0])} × {text(pair[1])}"))
        ok = r.status == "answer"
        return (float(r.answer) if ok else float("nan")), r.ops

    methods: Dict[str, Callable] = {"knn": knn, "mlp": mlp, "log-mlp": log_mlp, "exact": lambda p: float(p[0] * p[1])}
    rows, metrics = [], {}
    for split_name, pairs in tests.items():
        row = [split_name, len(pairs)]
        m: Dict[str, float] = {}
        hits, ops = 0, []
        for p in pairs:
            pred, o = hrmx(p)
            hits += exact_match(pred, p[0] * p[1])
            ops.append(o)
        m["hrmx"] = hits / len(pairs)
        m["hrmx_mean_ops"] = float(np.mean(ops))
        for name, f in methods.items():
            preds = [f(p) for p in pairs]
            m[name] = float(np.mean([exact_match(v, p[0] * p[1]) for v, p in zip(preds, pairs)]))
            m[name + "_median_rel_err"] = float(np.median([abs(v - float(p[0] * p[1])) / max(1e-9, float(p[0] * p[1]))
                                                          for v, p in zip(preds, pairs)]))
        metrics[split_name] = m
        rows.append(row + [f"{m['hrmx']:.3f}", f"{m['knn']:.3f}", f"{m['mlp']:.3f} ({m['mlp_median_rel_err']:.1e})",
                           f"{m['log-mlp']:.3f} ({m['log-mlp_median_rel_err']:.1e})", f"{m['exact']:.3f}",
                           f"{m['hrmx_mean_ops']:.0f}"])
    print(table(rows, ["split", "n", "HRM-X", "kNN", "MLP (med rel err)", "log-MLP (med rel err)", "exact", "HRM-X ops"]))

    # E12: cost of 10^k x 3.1 before and after sleep (fresh mind, same concept)
    fresh, fdom = Mind(), Arith()
    fresh.solve(fdom, ArithTask.from_text(["2 × 2 = 4", "2 × 3 = 6"], "10 × 3.1", oracle=lambda a, b: a * b))
    eff = []
    for k in range(1, 6):
        q = f"{10 ** k} × 3.1"
        before = fresh.solve(fdom, ArithTask.from_text([], q))
        eff.append([q, str(before.answer), before.ops])
    fdom.consolidate(fresh, "×", max_size=7, probes=16)
    for i, k in enumerate(range(1, 6)):
        after = fresh.solve(fdom, ArithTask.from_text([], f"{10 ** k} × 3.1"))
        saving = "refused before (cost > value)" if eff[i][1] == "None" else f"{eff[i][2] / after.ops:.1f}x"
        eff[i] += [str(after.answer), after.ops, saving]
    print(table(eff, ["question", "answer (no laws)", "ops", "answer (laws)", "ops", "saving"]))

    # E2: random nested expressions (depth 1-3) over + and ×, values outside the evidence
    comp_rng = random.Random(SEED + 1)

    def expr(depth: int) -> Tuple[str, Fraction]:
        if depth == 0:
            v = Fraction(comp_rng.randint(2, 60)) if comp_rng.random() < 0.6 else dec(comp_rng.uniform(0.1, 30))
            return str(v if v.denominator == 1 else f"{float(v):.1f}"), v
        (ls, lv), (rs, rv) = expr(depth - 1), expr(depth - 1)
        if comp_rng.random() < 0.5 and lv.denominator == 1:
            return f"({ls} × {rs})", lv * rv
        return f"({ls} + {rs})", lv + rv

    comp = {}
    for depth in (1, 2, 3):
        items = [expr(depth) for _ in range(40)]
        ok = sum(mind.solve(dom, ArithTask.from_text([], s)).answer == v for s, v in items)
        comp[f"depth {depth}"] = ok / len(items)
    print("[E2] compositional accuracy:", comp)

    record("x_arith", {"seed": SEED, "train_pairs": len(train), "train_range": "a 1..20 int, b 0.1..20.0",
                       "mlp": "2-64-64-1 tanh, Adam 3e-3, 3000 full-batch steps", "baseline_train_seconds": train_s,
                       "hrmx_evidence": "2x2=4, 2x3=6 + questions to teacher",
                       "hrmx_learning_trace": [(s.action, s.target, s.note) for s in learn.trace],
                       "laws": [str(l) for l in laws], "law_evaluations": law_evals},
           {"E1": metrics, "E12": eff, "E2": comp})


if __name__ == "__main__":
    main()
