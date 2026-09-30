"""E17 — does a stored abstraction lower the evidence needed to acquire a related concept?

Property U7 (abstraction transfer). A concept built on a stored one has a shorter description, so
under the same Occam price it needs fewer evidence rows before the posterior can prefer it over
"outside my language". That is a quantitative prediction: for each target, the smallest number of
rows at which the mind answers correctly should drop when `max` is stored, and nothing should
change for targets unrelated to `max`.

No teacher (rows are the only evidence), calibrated answer space (v0.6 default), 12 random tasks per (target, rows), queries outside the
evidence range. Memory conditions: empty; `max` admitted; `max` admitted but memory switched off
(the ablation that must restore the empty-memory numbers).

Run: `python -m experiments.x_transfer`  (writes `experiments/results/x_transfer.json`)
"""
from __future__ import annotations

import copy
import random
from fractions import Fraction
from typing import Dict

from experiments.common import record
from hrm_x.concept_memory import ConceptMemory
from hrm_x.concepts import ConceptConfig
from hrm_x.construct import IFGT, X, Y
from hrm_x.curriculum import make_task
from hrm_x.domains.concept import Concept
from hrm_x.mind import Controller, Mind

Q = Fraction
TARGETS = {
    "|a-b|": lambda a, b: abs(a - b), "relu(a-b)": lambda a, b: max(a - b, Q(0)),
    "2max+1": lambda a, b: 2 * max(a, b) + 1, "max(a,b^2)": lambda a, b: max(a, b * b),
    "a^2+b (unrelated)": lambda a, b: a * a + b,
}
ROWS = (3, 4, 5, 6, 7)
TRIALS = 12


def with_max() -> Dict:
    mind = Mind(Controller())
    mem = ConceptMemory.of(mind)
    k = mem.record(lambda x, y: max(x, y), (X, Y, X, Y, IFGT), "seed-a", [])
    mem.record(lambda x, y: max(x, y), (X, Y, X, Y, IFGT), "seed-b", [])
    mem.name(k, "max")
    assert mem.concepts[k].status == "admitted"
    return mind.knowledge


def run(knowledge: Dict, use_memory: bool, answer_space: str = "empirical") -> Dict:
    out = {}
    for name, op in TARGETS.items():
        row = {}
        for n in ROWS:
            rng = random.Random(1000 * n + len(name))
            t = {"correct": 0, "wrong": 0, "abstain": 0, "ops": 0}
            for i in range(TRIALS):
                task, truth = make_task(name, op, rng, n, f"t{i}", teacher=False)
                mind = Mind(Controller(), knowledge=copy.deepcopy(knowledge))
                r = mind.solve(Concept(ConceptConfig(answer_space=answer_space), use_memory=use_memory), task)
                t["ops"] += r.ops
                t["abstain" if r.status != "answer" else "correct" if r.answer == truth else "wrong"] += 1
            t["mean_ops"] = round(t.pop("ops") / TRIALS)
            row[str(n)] = t
        first = next((n for n in ROWS if row[str(n)]["correct"] >= TRIALS * 0.75), None)
        out[name] = {"by_rows": row, "rows_for_75pct_correct": first,
                     "wrong_total": sum(v["wrong"] for v in row.values())}
    return out


def main() -> Dict:
    stored = with_max()
    res = {"empty memory": run({}, True), "max stored": run(stored, True),
           "max stored, memory off": run(stored, False),
           "fixed |Y| (v0.5), empty memory": run({}, True, "fixed"),
           "fixed |Y| (v0.5), max stored": run(stored, True, "fixed")}
    for cond, v in res.items():
        print(f"[E17] {cond:24s} " + "  ".join(f"{k}: n75={x['rows_for_75pct_correct']} wrong={x['wrong_total']}"
                                               for k, x in v.items()), flush=True)
    return res


if __name__ == "__main__":
    record("x_transfer", {"experiment": "E17 evidence rows needed with and without a stored abstraction (U7)",
                          "rows": ROWS, "trials": TRIALS, "teacher": False}, main())
