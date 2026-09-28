"""Curiosity and volatility: does self-directed study pay, and does the NE-like signal adapt the mind?

C  Curiosity. Hypothesis: asking the question with the most uncertain answer under the concept's
   surviving hypotheses identifies the concept in fewer questions than random study, and a
   studied concept answers later questions (no teacher) correctly where an unstudied one abstains
   or errs. Falsified if random study is as efficient, or study does not change later accuracy.
V  Volatility. Hypothesis: when reasoning suddenly becomes unreliable (a false law enters memory)
   and verification is expensive, discounting stale reliability evidence on surprise makes the
   mind resume verification after fewer wrong answers. Falsified if wrong answers after the
   change are the same with the signal on and off.
Truth: Fraction arithmetic (oracle), never called by the mind except as a teacher.
  python -m experiments.x_metacog
"""
from __future__ import annotations

import random
from fractions import Fraction
from typing import Callable, Dict

import numpy as np

from experiments.common import record, table
from hrm_x.domains.arith import PROBES, Arith, ArithTask, Counter, Exhausted, programs, run, show
from hrm_x.induction import Law, behaviour_classes, consistent
from hrm_x.mind import Controller, Mind, Modulators

SEED = 0
CONCEPTS: Dict[str, Callable] = {"×": lambda a, b: a * b, "⊕": lambda a, b: a + b,
                                 "⊗": lambda a, b: 2 * a + b, "⊘": lambda a, b: a + 2 * b}


def ev(prog, xy):
    try:
        return run(prog, xy[0], xy[1], Counter(2000))
    except Exhausted:
        return None


def seeded_mind(symbol: str, truth: Callable, rng: random.Random) -> Mind:
    """A mind that has seen two examples of `symbol` and kept every consistent reading."""
    mind = Mind()
    pairs = [(Fraction(rng.randint(1, 4)), Fraction(rng.randint(1, 4))) for _ in range(2)]
    evidence = [(p, truth(*p)) for p in pairs]
    classes = consistent(behaviour_classes(programs(6), ev, PROBES), evidence, ev)
    mind.knowledge["arith"] = {"concepts": {symbol: {"hypotheses": [(h.program, h.weight) for h in classes],
                                                     "evidence": evidence}}, "laws": {}, "uses": {symbol: 1}}
    return mind


def later_accuracy(mind: Mind, symbol: str, truth: Callable, rng: random.Random, n: int = 50):
    dom = Arith()
    right = wrong = 0
    for _ in range(n):
        a = Fraction(rng.randint(2, 60))
        b = Fraction(f"{rng.uniform(0.5, 40):.1f}")
        r = mind.solve(dom, ArithTask.from_text([], f"{a} {symbol} {float(b):.1f}", symbol))
        right += r.status == "answer" and r.answer == truth(a, b)
        wrong += r.status == "answer" and r.answer != truth(a, b)
    return right / n, wrong / n


def curiosity() -> Dict:
    rows, out = [], {}
    for symbol, truth in CONCEPTS.items():
        for strategy in ("none", "random", "curious"):
            asks, accs, wrongs = [], [], []
            for trial in range(6):
                rng = random.Random(SEED * 1000 + trial)
                mind = seeded_mind(symbol, truth, rng)
                dom = Arith()
                asked = 0 if strategy == "none" else dom.study(mind, symbol, truth, budget=12, strategy=strategy,
                                                                seed=trial)
                acc, wrong = later_accuracy(mind, symbol, truth, random.Random(trial + 77))
                asks.append(asked)
                accs.append(acc)
                wrongs.append(wrong)
            out[f"{symbol}/{strategy}"] = {"questions": float(np.mean(asks)), "later_accuracy": float(np.mean(accs)),
                                           "later_wrong": float(np.mean(wrongs))}
            rows.append([symbol, strategy, f"{np.mean(asks):.1f}", f"{np.mean(accs):.3f}", f"{np.mean(wrongs):.3f}"])
    print(table(rows, ["concept", "study", "questions", "later accuracy", "later wrong"]))
    return out


def volatility() -> Dict:
    out = {}
    for enabled in (True, False):
        mind, dom = Mind(modulators=Modulators(enabled=enabled)), Arith()
        mind.control = Controller(verify_ops=5000.0)       # verification is expensive here
        mind.solve(dom, ArithTask.from_text(["2 × 2 = 4", "2 × 3 = 6"], "10 × 3.1", oracle=CONCEPTS["×"]))
        rng = random.Random(SEED)
        wrong_after, verified_after, first_verify = 0, 0, None
        for t in range(150):
            if t == 100:
                key = ("×", f"concept:{show(dom.best(mind, '×'))}")
                mind.knowledge["arith"]["laws"][key] = [Law(("f", "?x", "?z"), ("add", "?x", "?z"), 0)]   # applies to every task
            a, b = Fraction(rng.randint(2, 40)), Fraction(f"{rng.uniform(0.5, 30):.1f}")
            r = mind.solve(dom, ArithTask.from_text([], f"{a} × {float(b):.1f}"))
            correct = r.status == "answer" and r.answer == a * b
            if r.status == "answer":
                mind.feedback("arith", correct)
            if t >= 100:
                wrong_after += r.status == "answer" and not correct
                did_verify = any(s.action == "verify" for s in r.trace)
                verified_after += did_verify
                if did_verify and first_verify is None:
                    first_verify = t - 100
        out[f"volatility {'on' if enabled else 'off'}"] = {"wrong_answers_after_change": wrong_after,
                                                           "tasks_until_verification_resumed": first_verify,
                                                           "verified_after_change": verified_after}
        print(f"[V] volatility {'on ' if enabled else 'off'}: {wrong_after} wrong answers after the change; "
              f"verification resumed after {first_verify} tasks")
    return out


def main() -> None:
    metrics = {"curiosity": curiosity(), "volatility": volatility()}
    record("x_metacog", {"seed": SEED, "concepts": list(CONCEPTS), "study_budget": 12, "trials": 6,
                         "hypothesis_language": "programs up to size 6 over x, y, 0, 1, add, repeat",
                         "volatility_setting": "verify_ops=5000 (verification expensive), false law at task 100"},
           metrics)


if __name__ == "__main__":
    main()
