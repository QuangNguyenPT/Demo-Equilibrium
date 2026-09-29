"""E13 — the v0.5 concept route against the v0.4 program route, one controller.

This is the experiment the whole port exists to make possible: both routes are
driven by the *same* `Mind`, the same value-of-computation controller, the same
modulators and the same patience. The only thing that differs is the domain --
`Concept` (v0.5: exact fits of law-closed families) against `Arith` (v0.4:
search over a program DSL). Running them through one controller is what makes the
comparison fair; the delivery's standalone `concepts.decide()` loop is not used
here for exactly that reason.

Two task lists, because they probe different things.

**E13a, `OPS`** -- the delivery's ten operators, seven the hypothesis language
can express and three it cannot (`a^2 + b`, `max`, `a^b + a`). This is the
direct test of audit finding F3, which measured the v0.4 route answering
out-of-DSL probes **with status `answer`** and a wrong value. Note what these
probes cannot test: where the evidence varies in both arguments, the open-world
coverage rule never fires, because every argument is covered. The closed-world
ablation is vacuous here and the run says so by printing near-identical numbers
for it. E13b is the list that exercises it.

**E13b, `PINNED`** -- the same operators with the *second* argument held fixed in
the evidence, so a reading that is not symmetric in `x, y` is refused at an
unseen `y`. These are the only cells in the design where the coverage rule and
the reserved BOTTOM hypothesis can change an answer, and therefore the only
cells where the ablation means anything.

Conditions, on both lists:
  concept, open world + teacher    the shipped configuration
  concept, closed world + teacher  bits_bottom = inf, coverage off (the ablation)
  concept, no teacher              Controller.use_questions = False
  concept, open world + L=4        the delivery's answer threshold, not ours
  arith v0.4, teacher              the incumbent route
  arith v0.4, no teacher

Run: `python -m experiments.x_v05`  (writes `experiments/results/x_v05.json`)

Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
"""
from __future__ import annotations

from fractions import Fraction as F
from typing import Callable, Dict, List, Tuple

from experiments.common import record
from hrm_x.concepts import ConceptConfig
from hrm_x.domains.arith import Arith, ArithTask
from hrm_x.domains.concept import Concept, ConceptTask
from hrm_x.mind import Controller, Mind

Q = F
Op = Callable[[Q, Q], Q]
Row = Tuple[str, Op, bool, List[Tuple[Q, Q]], List[Tuple[Q, Q]]]


def _pow(a: Q, b: Q) -> Q:
    return a ** int(b)


OPS: List[Row] = [
    # name, operator, inside the v0.5 hypothesis language, evidence pairs, queries
    ("multiply", lambda a, b: a * b, True,
     [(Q(2, 1), Q(3, 1)), (Q(3, 1), Q(4, 1)), (Q(1, 1), Q(5, 1))],
     [(Q(4, 1), Q(6, 1)), (Q(5, 1), Q(5, 1)), (Q(7, 1), Q(8, 1))]),
    ("add", lambda a, b: a + b, True,
     [(Q(2, 1), Q(3, 1)), (Q(4, 1), Q(1, 1))],
     [(Q(5, 1), Q(7, 1)), (Q(9, 1), Q(2, 1))]),
    ("subtract", lambda a, b: a - b, True,
     [(Q(5, 1), Q(3, 1)), (Q(9, 1), Q(4, 1))],
     [(Q(7, 1), Q(2, 1)), (Q(11, 1), Q(5, 1))]),
    ("divide", lambda a, b: a / b, True,
     [(Q(4, 1), Q(2, 1)), (Q(9, 1), Q(3, 1))],
     [(Q(7, 1), Q(2, 1)), (Q(10, 1), Q(4, 1))]),
    ("power", _pow, True,
     [(Q(2, 1), Q(3, 1)), (Q(3, 1), Q(2, 1))],
     [(Q(2, 1), Q(5, 1)), (Q(4, 1), Q(2, 1))]),
    ("ab+1", lambda a, b: a * b + 1, True,
     [(Q(2, 1), Q(3, 1)), (Q(4, 1), Q(1, 1))],
     [(Q(5, 1), Q(2, 1)), (Q(3, 1), Q(3, 1))]),
    ("2a+3b", lambda a, b: 2 * a + 3 * b, True,
     [(Q(2, 1), Q(3, 1)), (Q(4, 1), Q(1, 1))],
     [(Q(5, 1), Q(2, 1)), (Q(1, 1), Q(3, 1))]),
    # --- outside the language on purpose ---
    ("square+add", lambda a, b: a * a + b, False,
     [(Q(2, 1), Q(3, 1)), (Q(4, 1), Q(1, 1))],
     [(Q(5, 1), Q(2, 1)), (Q(3, 1), Q(3, 1))]),
    ("max", max, False,
     [(Q(2, 1), Q(3, 1)), (Q(4, 1), Q(1, 1))],
     [(Q(5, 1), Q(2, 1)), (Q(3, 1), Q(3, 1))]),
    ("pow+self", lambda a, b: _pow(a, b) + a, False,
     [(Q(2, 1), Q(3, 1)), (Q(3, 1), Q(2, 1))],
     [(Q(2, 1), Q(5, 1)), (Q(4, 1), Q(2, 1))]),
]

# y is pinned at 3 in every evidence row, so `seen_y` has one member and a
# reading that is not symmetric in x, y loses its coverage at an unseen y. For
# `square+add` the symmetric family xy + 3x + 3y - 14 fits both rows exactly, so
# this is where a coincidental fit has to be met with something other than
# confidence.
PINNED: List[Row] = [
    ("multiply", lambda a, b: a * b, True,
     [(Q(2, 1), Q(3, 1)), (Q(4, 1), Q(3, 1))], [(Q(5, 1), Q(7, 1))]),
    ("add", lambda a, b: a + b, True,
     [(Q(2, 1), Q(3, 1)), (Q(4, 1), Q(3, 1))], [(Q(5, 1), Q(7, 1))]),
    ("square+add", lambda a, b: a * a + b, False,
     [(Q(2, 1), Q(3, 1)), (Q(4, 1), Q(3, 1))], [(Q(5, 1), Q(7, 1))]),
    ("max", max, False,
     [(Q(2, 1), Q(3, 1)), (Q(4, 1), Q(3, 1))], [(Q(5, 1), Q(7, 1))]),
    ("pow+self", lambda a, b: _pow(a, b) + a, False,
     [(Q(2, 1), Q(3, 1)), (Q(4, 1), Q(3, 1))], [(Q(5, 1), Q(7, 1))]),
]

TABLES = {"E13a general": OPS, "E13b pinned-argument": PINNED}

# Both domains key learned concepts on the operator symbol, so every operator
# needs its own glyph: one run's concept would otherwise answer the next one's
# question out of semantic memory. `Arith` additionally parses its question from
# text, so the string has to be the real expression.
_GLYPHS = ["×", "⊕", "⊖", "⊘", "⊙", "⊛", "⊞", "⊠", "⊡", "⊢",
           "⊤", "⊥", "⊎", "⊓", "⊔"]
SYMBOL_OF: Dict[Tuple[str, str], str] = {}
for _table, _rows in TABLES.items():
    for _i, _row in enumerate(_rows):
        SYMBOL_OF.setdefault((_table, _row[0]), _GLYPHS[_i % len(_GLYPHS)])


def _mind(use_questions: bool, wrong_penalty: float) -> Mind:
    return Mind(Controller(use_questions=use_questions, wrong_penalty=wrong_penalty))


def _outcome(result, truth: Q) -> Dict:
    return {"status": result.status, "answer": result.answer, "ops": result.ops,
            "cycles": result.cycles, "truth": truth,
            "asks": sum(1 for s in result.trace if s.action == "ask")}


def run_concept(table: str, name: str, op: Op, evidence, query, cfg: ConceptConfig,
                teacher: bool, penalty: float) -> Dict:
    task = ConceptTask.from_op(op, evidence, query, symbol=SYMBOL_OF[(table, name)],
                               oracle=op if teacher else None)
    return _outcome(_mind(teacher, penalty).solve(Concept(cfg), task), op(*query))


def run_arith(table: str, name: str, op: Op, evidence, query, teacher: bool,
              penalty: float) -> Dict:
    symbol = SYMBOL_OF[(table, name)]
    task = ArithTask(symbol, [(a, b, op(a, b)) for a, b in evidence],
                     f"{query[0]} {symbol} {query[1]}", oracle=op if teacher else None)
    return _outcome(_mind(teacher, penalty).solve(Arith(), task), op(*query))


def tally(rows: List[Dict]) -> Dict:
    correct = sum(1 for r in rows if r["status"] == "answer" and r["answer"] == r["truth"])
    wrong = sum(1 for r in rows if r["status"] == "answer" and r["answer"] != r["truth"])
    abstain = sum(1 for r in rows if r["status"] == "abstain")
    n = max(1, len(rows))
    return {"n": len(rows), "correct": correct, "wrong": wrong, "abstain": abstain,
            "mean_ops": round(sum(r["ops"] for r in rows) / n, 1),
            "mean_asks": round(sum(r["asks"] for r in rows) / n, 3)}


# (route, ConceptConfig or None, teacher available, Controller.wrong_penalty)
#
# The L column is the delivery's own calibration: it answers only when
# q > L/(1+L), and picked L = 4 (q > 0.8). `Controller` ships L = 1 (q > 0.5).
# Both are hand-set constants, and doc 13 lists exactly this one among the five
# it says are unjustified, so it is measured rather than assumed.
CONDITIONS = {
    "concept, open world + teacher": ("concept", ConceptConfig(), True, 1.0),
    "concept, closed world + teacher": ("concept", ConceptConfig(bits_bottom=float("inf"),
                                                                coverage=False), True, 1.0),
    "concept, no teacher": ("concept", ConceptConfig(), False, 1.0),
    "concept, open world + L=4": ("concept", ConceptConfig(), True, 4.0),
    "arith v0.4, teacher": ("arith", None, True, 1.0),
    "arith v0.4, no teacher": ("arith", None, False, 1.0),
}


def main() -> Dict[str, Dict]:
    rows: List[Dict] = []
    for label, (route, cfg, teacher, penalty) in CONDITIONS.items():
        for table, entries in TABLES.items():
            for name, op, inside, evidence, queries in entries:
                for query in queries:
                    if route == "concept":
                        got = run_concept(table, name, op, evidence, query, cfg, teacher, penalty)
                    else:
                        got = run_arith(table, name, op, evidence, query, teacher, penalty)
                    rows.append(dict(got, condition=label, table=table, op=name,
                                     in_language=inside))

    out: Dict[str, Dict] = {}
    for label in CONDITIONS:
        for table in TABLES:
            for scope in ("all", "in-language", "out-of-language"):
                sel = [r for r in rows if r["condition"] == label and r["table"] == table
                       and (scope == "all" or r["in_language"] == (scope == "in-language"))]
                if sel:
                    out[f"{label} [{table}, {scope}]"] = tally(sel)
    return out


if __name__ == "__main__":
    conditions = main()
    width = max(len(k) for k in conditions)
    for key, value in conditions.items():
        print(f"{key:<{width}}  n={value['n']:>2}  correct={value['correct']:>2}"
              f"  wrong={value['wrong']:>2}  abstain={value['abstain']:>2}"
              f"  mean_ops={value['mean_ops']:>9}  mean_asks={value['mean_asks']}")
    record("x_v05", {"experiment": "E13 v0.5 concept route vs v0.4 arith route, one mind",
                     "tables": {k: len(v) for k, v in TABLES.items()}}, conditions)
