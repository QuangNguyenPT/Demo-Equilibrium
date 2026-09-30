"""E14 — ADR-011 step A: does the rival margin separate right answers from wrong ones?

ADR-011 proposes answering only when the best *disagreeing* reading is `d` bits behind,
and insists the margin be measured before it is implemented: if the confidently wrong
out-of-language answers of E13 carry margins as large as the correct in-language ones,
the margin does not discriminate and proposal A is the wrong fix.

This re-runs every E13 answer of the shipped configuration (concept route, open world,
teacher, L = 1), refits the evidence the mind ended with (examples + questions it paid
for), and records per answer: correct or wrong, the rival margin in bits
(`concepts.rival_margin`) and the surplus rows of the winning reading. The separating
threshold, if any, is reported as the interval between the largest wrong margin and the
smallest correct one.

The E13 tables were written by hand and the surplus pattern seen on them is post hoc, so a
second, seeded set of operators nobody looked at while designing the rule (`HELD_OUT`:
random bi-affine and monomial operators inside the language, and seven non-algebraic ones
outside it, on random evidence) is scored with the same fields. A gate earns a place in
the code only if it separates the held-out set too.

Run: `python -m experiments.x_margin`  (writes `experiments/results/x_margin.json`)
"""
from __future__ import annotations

import math
import random
from fractions import Fraction as Q
from typing import Callable, Dict, List, Tuple

from experiments.common import record
from experiments.x_v05 import SYMBOL_OF, TABLES, _mind
from hrm_x.concepts import ConceptConfig, rival_margin
from hrm_x.domains.concept import Concept, ConceptTask


SEED = 14
OUTSIDE: Dict[str, Callable[[Q, Q], Q]] = {
    "x^2+y": lambda a, b: a * a + b, "max": max, "min": min, "|x-y|": lambda a, b: abs(a - b),
    "x^y+x": lambda a, b: a ** int(b) + a, "xy^2+1": lambda a, b: a * b * b + 1,
    "x+y+xy^2": lambda a, b: a + b + a * b * b,
}


def held_out(n_inside: int = 40, seed: int = SEED) -> List[Tuple[str, str, Callable, bool, List, List]]:
    """(table, name, op, in_language, evidence, [query]) with 2-4 random evidence pairs over 1..6."""
    rng = random.Random(seed)
    ops: List[Tuple[str, Callable, bool]] = []
    for i in range(n_inside):
        if i % 2:
            c = [Q(rng.randint(-3, 3)) for _ in range(4)]
            if not any(c[:3]):
                c[0] = Q(1)
            ops.append((f"{c[0]}xy+{c[1]}x+{c[2]}y+{c[3]}",
                        (lambda a, b, c=c: c[0] * a * b + c[1] * a + c[2] * b + c[3]), True))
        else:
            k, e1, e2 = Q(rng.choice([-2, -1, 1, 2, 3])), rng.randint(0, 2), rng.randint(1, 2)
            ops.append((f"{k}x^{e1}y^{e2}", (lambda a, b, k=k, e1=e1, e2=e2: k * a ** e1 * b ** e2), True))
    ops += [(name, op, False) for name, op in OUTSIDE.items() for _ in range(3)]
    out = []
    for name, op, inside in ops:
        pairs = rng.sample([(Q(a), Q(b)) for a in range(1, 7) for b in range(1, 7)], rng.randint(2, 4) + 1)
        out.append(("held-out", name, op, inside, pairs[:-1], [pairs[-1]]))
    return out


def run(cfg: ConceptConfig, teacher: bool, rows_in=None) -> List[Dict]:
    rows = []
    entries = rows_in if rows_in is not None else [(t, *e) for t, es in TABLES.items() for e in es]
    for i, (table, name, op, inside, evidence, queries) in enumerate(entries):
        symbol = SYMBOL_OF.get((table, name), f"g{i}")
        for query in queries:
            task = ConceptTask.from_op(op, evidence, query, symbol=symbol, oracle=op if teacher else None)
            mind, dom = _mind(teacher, 1.0), Concept(cfg)
            result = mind.solve(dom, task)
            if result.status != "answer":
                rows.append({"table": table, "op": name, "in_language": inside, "status": "abstain"})
                continue
            bel = dom.belief(task, mind)          # written + constructed (v0.6) + stored families
            winner = max((f for f in bel.fitted if f.value(*query, coverage=False) == result.answer),
                         key=lambda f: f.log2w)
            rows.append({"table": table, "op": name, "in_language": inside, "status": "answer",
                         "query": [str(q) for q in query], "correct": result.answer == op(*query),
                         "margin_bits": rival_margin(bel, query[0], query[1], result.answer),
                         "surplus": winner.surplus, "evidence_rows": len(task.evidence()),
                         "winner": winner.family.name})
    return rows


def summarise(rows: List[Dict]) -> Dict:
    answered = [r for r in rows if r["status"] == "answer"]
    right = sorted(r["margin_bits"] for r in answered if r["correct"])
    wrong = sorted(r["margin_bits"] for r in answered if not r["correct"])
    finite = lambda v: v if math.isfinite(v) else ("inf" if v > 0 else "-inf")
    out = {"answered": len(answered), "abstained": len(rows) - len(answered),
           "correct_margins": [finite(round(v, 2)) for v in right],
           "wrong_margins": [finite(round(v, 2)) for v in wrong],
           "correct_surplus": sorted(r["surplus"] for r in answered if r["correct"]),
           "wrong_surplus": sorted(r["surplus"] for r in answered if not r["correct"])}
    for k in (1, 2):
        kept = [r for r in answered if r["surplus"] >= k]
        out[f"surplus>={k}"] = {"kept_correct": sum(r["correct"] for r in kept),
                                "kept_wrong": sum(not r["correct"] for r in kept)}
    if right and wrong:
        lo, hi = max(wrong), min(right)
        out["separating_interval_bits"] = [finite(round(lo, 2)), finite(round(hi, 2))] if lo < hi else None
        out["correct_kept_at_best_threshold"] = sum(1 for v in right if v > lo)
    out["rows"] = [{k: (finite(round(v, 2)) if isinstance(v, float) else v) for k, v in r.items()}
                   for r in rows]
    return out


def outcome(rows: List[Dict]) -> Dict:
    answered = [r for r in rows if r["status"] == "answer"]
    right = sum(r["correct"] for r in answered)
    return {"correct": right, "wrong": len(answered) - right, "abstain": len(rows) - len(answered)}


def main() -> Dict:
    ho = held_out()
    out = {"open world + teacher (shipped)": summarise(run(ConceptConfig(), True)),
           "open world, no teacher": summarise(run(ConceptConfig(), False)),
           "held-out, teacher": summarise(run(ConceptConfig(), True, ho)),
           "held-out, no teacher": summarise(run(ConceptConfig(), False, ho)),
           "E13, teacher, min_surplus=2": summarise(run(ConceptConfig(min_surplus=2), True)),
           "held-out, teacher, min_surplus=2": summarise(run(ConceptConfig(min_surplus=2), True, ho))}
    # Sensitivity of the one constant ADR-012 made load-bearing. Reported, not tuned: the
    # shipped value stays the spec's 4 bits.
    out["bits_bottom sweep (teacher)"] = {
        str(bb): {"E13": outcome(run(ConceptConfig(bits_bottom=bb), True)),
                  "held-out": outcome(run(ConceptConfig(bits_bottom=bb), True, ho))}
        for bb in (0.0, 4.0, 8.0, 12.0, math.inf)}
    return out


if __name__ == "__main__":
    metrics = main()
    for bb, v in metrics["bits_bottom sweep (teacher)"].items():
        print(f"[MARGIN] bits_bottom={bb}: E13 {v['E13']}  held-out {v['held-out']}")
    for label, m in metrics.items():
        if "sweep" in label:
            continue
        print(f"[MARGIN] {label}: answered {m['answered']}, abstained {m['abstained']}")
        print(f"         correct margins {m['correct_margins']}")
        print(f"         wrong margins   {m['wrong_margins']}")
        print(f"         surplus correct {m['correct_surplus']} wrong {m['wrong_surplus']}")
        print(f"         separating interval {m.get('separating_interval_bits')}  "
              f"surplus>=1 {m['surplus>=1']}  surplus>=2 {m['surplus>=2']}")
    record("x_margin", {"experiment": "E14 rival margin on the E13 answers (ADR-011 step A)",
                        "source": "experiments/x_v05.py TABLES + seeded held-out set",
                        "held_out_seed": SEED}, metrics)
