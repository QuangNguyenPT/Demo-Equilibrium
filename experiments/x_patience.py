"""E9b: is a *learned* price of computation better than a fixed one?

`experiments/x_control` compares the VOC controller against greedy and fixed
policies. It leaves lambda fixed. `hrm_x.mind.Patience` can learn it, but was
never evaluated, so `Controller.learn_patience` is still off and unmeasured.

Two facts from the implementation constrain any honest test, and both are easy
to get wrong:

  * `Mind.solve` reads the price once per episode (mind.py:337) and folds the
    finished episode in at the end (mind.py:434). So patience can only adapt
    *between* episodes. One held-out task tests nothing. Each arm below is
    therefore a PERSISTENT mind walking a stream of held-out tasks.
  * `Mind._utility` is `2*weight - 1` clipped at 0 (mind.py:301) -- the mind's
    own confidence, not ground truth. Patience's fixed point is
    E[confidence]/ops, which is self-consistent Bayesian utility but is *not*
    accuracy. Known miscalibration (see ADR-009) therefore matters here, and
    the lambda trajectory is reported rather than assumed stable.

Baseline is the hard one: comparing a learned price only against the default
2e-05 would prove nothing, because "learned" might just be "different". So the
fixed baseline is a sweep across patience's whole reachable range, and the
learned arms are judged against the BEST fixed arm, not the default.

Verdict rule, fixed before running:
  patience pays off iff some learned arm, on some stream, has accuracy >=
  the best fixed arm's accuracy AND mean ops <= that arm's mean ops (relative
  tolerance 1e-9), with the price not pinned at a bound for the whole run.
  A learned arm that merely matches the default reports as "no evidence".

Design limitation, stated up front: a stream cannot interleave
induce-from-examples arithmetic with a persistent mind, because that mind's own
learned knowledge would answer them and the task would stop being hard. The
cheap regime is therefore arithmetic over known concepts; the expensive regime
is carried by novel grid tasks, which no prior knowledge solves.

    python -m experiments.x_patience
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Tuple

import numpy as np

from experiments.common import record, table
from experiments.x_control import arith_tasks, learned_arith_mind
from experiments.x_grid import COMPOSED, SINGLE, correct as grid_correct, make_task
from hrm_x.domains.arith import Arith
from hrm_x.domains.grid import Grids
from hrm_x.mind import DEFAULT_LAM, Controller, Mind, Patience

SEED = 101
ARITH_N = 30
PATIENCE_BOUND = Patience.bound

# Seven decades, exactly patience's reachable range, so "best fixed" is a fair
# ceiling rather than a lucky hand-picked constant.
FIXED_SWEEP = [10.0 ** e for e in range(-7, 0)]


def build_streams(rng: random.Random) -> Dict[str, List[Tuple[str, Any, Any]]]:
    """Held-out task streams. Arithmetic inputs are random; the concepts are known
    (the limitation named in the module docstring)."""
    streams: Dict[str, List[Tuple[str, Any, Any]]] = {
        "arith": [("arith", t[1], t[2]) for t in arith_tasks(rng, ARITH_N, hard=False)],
        "grid": [],
    }
    for name in rng.sample(list(SINGLE), 8):
        t = make_task(rng, (name,), False)
        if t is not None:
            streams["grid"].append(("grid", t[0], t[1]))
    for spec in COMPOSED:
        t = make_task(rng, spec, True)
        if t is not None:
            streams["grid"].append(("grid", t[0], t[1]))
    return streams


def _arm(arm: str, kind: str, start: float, learn: bool) -> Tuple[Any, Mind]:
    """A fresh persistent mind for one arm, carrying only the knowledge it needs."""
    if kind == "arith":
        dom: Any = Arith()
        mind = copy.deepcopy(learned_arith_mind())
    else:
        dom, mind = Grids(), Mind()
    mind.control = Controller(lam=start, learn_patience=learn)
    if learn:
        # Inherit explicitly rather than leaning on Mind.__post_init__ ordering.
        mind.patience = Patience(lam=start, base=start)
    return dom, mind


def _scored(kind: str, r, truth: Any) -> Tuple[bool, bool]:
    if kind == "grid":
        right = grid_correct(r, truth)
    else:
        right = r.status == "answer" and r.answer == truth
    return right, (r.status == "answer" and not right)


def run_arm(arm: str, start: float, learn: bool, kind: str, tasks: List[Tuple[str, Any, Any]]) -> Dict[str, Any]:
    dom, mind = _arm(arm, kind, start, learn)
    right, wrong, ops, cycles, lams = [], [], [], [], []
    for _, task, truth in tasks:
        r = mind.solve(dom, task)
        ok, bad = _scored(kind, r, truth)
        right.append(ok)
        wrong.append(bad)
        ops.append(r.ops)
        cycles.append(max(r.cycles, 1))
        if learn:
            lams.append(mind.patience.lam)
    out: Dict[str, Any] = {
        "arm": arm, "n": len(tasks), "lam_start": start, "learned": learn,
        "accuracy": float(np.mean(right)), "wrong_answers": float(np.mean(wrong)),
        "mean_ops": float(np.mean(ops)), "mean_cycles": float(np.mean(cycles)),
        "total_ops": int(np.sum(ops)),
    }
    if learn:
        lo, hi = PATIENCE_BOUND
        out["lam_final"] = mind.patience.lam
        out["lam_min"] = float(min(lams))
        out["lam_max"] = float(max(lams))
        out["density_ema"] = mind.patience.density
        out["pinned_at_bound"] = bool(
            all(abs(l - mind.patience.lam) < 1e-18 for l in lams) and
            (abs(mind.patience.lam - lo) < 1e-18 or abs(mind.patience.lam - hi) < 1e-18)
        )
        out["lam_trace"] = [float(f"{l:.6g}") for l in lams]
    return out


def main() -> None:
    rng = random.Random(SEED)
    streams = build_streams(rng)
    arms: List[Tuple[str, float, bool]] = [(f"fixed:{l:.0e}", l, False) for l in FIXED_SWEEP]
    arms.append((f"fixed:default:{DEFAULT_LAM:.0e}", DEFAULT_LAM, False))
    arms.append((f"learned@default:{DEFAULT_LAM:.0e}", DEFAULT_LAM, True))
    arms.append((f"learned@high:{PATIENCE_BOUND[1]:.0e}", PATIENCE_BOUND[1], True))

    metrics: Dict[str, Any] = {}
    for kind, tasks in streams.items():
        rows = []
        for arm, start, learn in arms:
            m = run_arm(arm, start, learn, kind, tasks)
            metrics[f"{kind}/{arm}"] = m
            rows.append([arm, m["n"], f"{m['accuracy']:.3f}", f"{m['wrong_answers']:.3f}",
                         f"{m['mean_ops']:.0f}", f"{m['mean_cycles']:.1f}"])
        print(f"\n[{kind}] n={len(tasks)} held-out tasks")
        print(table(rows, ["arm", "n", "accuracy", "wrong", "ops", "cycles"]))
        for arm, _, learn in arms:
            if not learn:
                continue
            m = metrics[f"{kind}/{arm}"]
            print(f"  lam {m['lam_start']:.0e} -> {m['lam_final']:.6g} "
                  f"(range {m['lam_min']:.3g}..{m['lam_max']:.3g}, pinned={m['pinned_at_bound']})")

    # Verdict against the best fixed arm per stream, not against the default.
    verdict: Dict[str, Any] = {}
    pays_off = False
    for kind in streams:
        fixed = {a: metrics[f"{kind}/{a}"] for a, s, l in arms if not l}
        best = max(fixed.values(), key=lambda m: (round(m["accuracy"], 6), -m["mean_ops"]))
        learned = {a: metrics[f"{kind}/{a}"] for a, s, l in arms if l}
        for arm, m in learned.items():
            dominates = (m["accuracy"] >= best["accuracy"] - 1e-12
                         and m["mean_ops"] <= best["mean_ops"] * (1 + 1e-9)
                         and not m["pinned_at_bound"])
            verdict[f"{kind}/{arm}"] = {
                "best_fixed_arm": best["arm"], "best_fixed_accuracy": best["accuracy"],
                "best_fixed_mean_ops": best["mean_ops"],
                "accuracy_delta": m["accuracy"] - best["accuracy"],
                "ops_ratio_vs_best_fixed": m["mean_ops"] / best["mean_ops"],
                "pays_off": bool(dominates),
            }
            pays_off = pays_off or dominates
        print(f"\n[{kind}] best fixed = {best['arm']} "
              f"(acc {best['accuracy']:.3f}, ops {best['mean_ops']:.0f})")
        for arm, v in verdict.items():
            if arm.startswith(f"{kind}/"):
                print(f"  {arm}: acc {v['accuracy_delta']:+.3f} vs best fixed, "
                      f"ops x{v['ops_ratio_vs_best_fixed']:.2f}, pays_off={v['pays_off']}")

    if pays_off:
        conclusion = ("A learned price matched or beat the best fixed price on at least one "
                      "stream. See per-arm deltas; this is an enable-and-measure signal, not a proof.")
    else:
        conclusion = ("NO EVIDENCE that a learned price beats the best fixed price. The best fixed "
                      "price is at least as accurate and no more expensive, so learn_patience "
                      "stays off. A learned arm that only matches the default means the default "
                      "was already the right price, not that learning works.")
    print("\n[E9b] " + conclusion)
    metrics["verdict"] = verdict
    metrics["pays_off"] = bool(pays_off)
    metrics["conclusion"] = conclusion
    record("x_patience", {
        "seed": SEED, "arith_n": ARITH_N, "default_lam": DEFAULT_LAM,
        "patience_bound": list(PATIENCE_BOUND), "fixed_sweep": FIXED_SWEEP,
        "arms": [a for a, _, _ in arms],
        "streams": {k: len(v) for k, v in streams.items()},
        "verdict_rule": ("accuracy >= best fixed AND mean ops <= best fixed, price not pinned "
                         "at a bound; judged against the best fixed arm, not the default"),
    }, metrics)


if __name__ == "__main__":
    main()
