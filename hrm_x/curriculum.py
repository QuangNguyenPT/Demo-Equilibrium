"""The task field HRM-X v0.6 trains on: operators over rationals, from written to constructible.

EQB learned from generated few-shot tasks (demos + query) and memorised them (ADR-012). The same
shape is kept -- a task is worked examples plus one query -- but what is *learned* changes: not
weights that map the task to an answer, but concepts that survive exact verification and are
stored for reuse. Four pools, ordered by what the mind has to do:

  written       inside the 68 families (a b, a + b, 2a + 3b, a / b, a^b, ab + 1)
  constructible outside them, reachable by construction from primitives (max, x^2 + y, relu(x - y),
                |x - y|, x / (x + y), min, (x + y)^2)
  composite     reachable within budget only by reusing a stored concept (max(x, y^2) is 9 tokens
                from scratch, 5 once max is stored)
  outside       outside the primitives altogether (gcd, x mod y) -- must be refused

Evidence points are small positive integers; queries are drawn from a wider range (magnitude x5,
fractions), so an interpolating learner is visible as an extrapolation error (U4).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Sequence, Tuple

from hrm_x.domains.concept import ConceptTask

Q = Fraction
Op = Callable[[Q, Q], Q]


def _pow(a: Q, b: Q) -> Q:
    return a ** int(b)


def _mod(a: Q, b: Q) -> Q:
    return Q(int(a) % int(b))


POOLS: Dict[str, Dict[str, Op]] = {
    "written": {
        "a*b": lambda a, b: a * b, "a+b": lambda a, b: a + b, "2a+3b": lambda a, b: 2 * a + 3 * b,
        "a/b": lambda a, b: a / b, "ab+1": lambda a, b: a * b + 1, "a-b": lambda a, b: a - b,
    },
    "constructible": {
        "max": max, "min": min, "|a-b|": lambda a, b: abs(a - b), "a^2+b": lambda a, b: a * a + b,
        "relu(a-b)": lambda a, b: max(a - b, Q(0)), "a/(a+b)": lambda a, b: a / (a + b),
        "(a+b)^2": lambda a, b: (a + b) ** 2, "2max+1": lambda a, b: 2 * max(a, b) + 1,
    },
    "composite": {
        "max(a,b^2)": lambda a, b: max(a, b * b), "max*b": lambda a, b: max(a, b) * b,
        "max(a,b)^2": lambda a, b: max(a, b) ** 2,
    },
    "outside": {
        "gcd": lambda a, b: Q(math.gcd(int(a), int(b))), "a mod b": _mod,
    },
}
INTEGER_ONLY = {"gcd", "a mod b"}


@dataclass
class Sample:
    pool: str
    name: str
    task: ConceptTask
    truth: Q


def make_task(name: str, op: Op, rng: random.Random, n_rows: int, symbol: str, teacher: bool = True,
              wide_query: bool = True) -> Tuple[ConceptTask, Q]:
    """Distinct evidence pairs over 1..9; a query outside the evidence range unless wide_query=False."""
    pairs: List[Tuple[Q, Q]] = []
    tried = set()
    while len(pairs) < n_rows:
        p = (Q(rng.randint(1, 9)), Q(rng.randint(1, 9)))
        if p in tried:
            continue
        tried.add(p)
        pairs.append(p)
    while True:
        if wide_query and name not in INTEGER_ONLY and rng.random() < 0.3:
            q = (Q(rng.randint(1, 40), 2), Q(rng.randint(1, 40), 2))
        elif wide_query:
            q = (Q(rng.randint(1, 45)), Q(rng.randint(1, 45)))
        else:
            q = (Q(rng.randint(1, 9)), Q(rng.randint(1, 9)))
        if q not in tried:
            break
    task = ConceptTask.from_op(op, pairs, q, symbol=symbol, oracle=op if teacher else None)
    return task, op(*q)


class Curriculum:
    """A seeded stream of tasks, pool by pool (the order matters: composites need stored concepts)."""

    def __init__(self, pools: Sequence[str] = ("written", "constructible", "composite"), seed: int = 0,
                 rows: Tuple[int, int] = (4, 6), teacher: bool = True, repeats: int = 2) -> None:
        self.pools, self.rng, self.rows, self.teacher, self.repeats = list(pools), random.Random(seed), rows, teacher, repeats
        self.counter = 0

    def __iter__(self):
        for pool in self.pools:
            for _ in range(self.repeats):
                for name, op in POOLS[pool].items():
                    self.counter += 1
                    task, truth = make_task(name, op, self.rng, self.rng.randint(*self.rows),
                                            f"op{self.counter}", self.teacher)
                    yield Sample(pool, name, task, truth)
