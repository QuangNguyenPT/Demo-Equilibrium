"""Concept memory: constructed features that earned a place, stored as reusable operators.

A concept is kept as its *expression* over the primitives (and over earlier concepts), not as a
fitted reading. That is the difference from v0.5's `Concept.remember`, which stored a version space
nothing ever read back (ADR-012 E9). Stored here, a concept is used three ways:

  1. as a base feature at propose level 0 -- a task that needs it costs one exact fit, no search;
  2. as a primitive token in construction -- concepts built on concepts stay short (compression);
  3. as a readout channel of the router (hrm_x/mbi.py) -- which concept verified on similar evidence.

Admission is two-stage, which is ADR-011 C made live: a construction that answers a task is a
*candidate*; it is *admitted* (becomes a level-0 feature and a primitive) once it has been verified
on a second task with different evidence, or at sleep when its MDL gain over the tasks that used it
is positive. A candidate that later fails verification loses support; one that reaches
`FAILURES_TO_RETIRE` more failures than successes is retired, never deleted, so the record of what
was believed survives.

Every stored concept is evaluated exactly (Fractions), so memory can make search cheaper but can
never change what a verified answer is.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hrm_x.construct import (LIB, PROBES, Expr, Feature, _base_projector, _key, evaluate, expr_bits,
                             render)

Q = Fraction
FAILURES_TO_RETIRE = 2


@dataclass
class StoredConcept:
    expr: Expr
    name: str
    status: str = "candidate"              # candidate | admitted | retired
    tasks: List[str] = field(default_factory=list)
    successes: int = 0
    failures: int = 0
    evidence: List[List[str]] = field(default_factory=list)


class ConceptMemory:
    """Ordered list of stored concepts; concept k may call concepts 0..k-1 (tokens LIB + j).

    The state is a plain JSON-able dict (`Mind.save` writes JSON), held by reference inside
    `mind.knowledge`, so every mutation made through this view persists with the mind.
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None) -> None:
        self.state = state if state is not None else {"concepts": []}
        self.concepts: List[StoredConcept] = [_View(d) for d in self.state["concepts"]]  # type: ignore[misc]

    # ----- exact evaluation -----
    def fn(self, k: int) -> Callable[[Q, Q], Optional[Q]]:
        expr, prefix = self.concepts[k].expr, [self.fn(j) for j in range(k)]
        return lambda x, y, e=expr, p=prefix: evaluate(e, x, y, p)

    def fns(self) -> List[Callable[[Q, Q], Optional[Q]]]:
        return [self.fn(k) for k in range(len(self.concepts))]

    def names(self) -> List[str]:
        return [c.name for c in self.concepts]

    def usable(self) -> List[int]:
        """Indices a construction may call and level 0 may fit: admitted concepts."""
        return [k for k, c in enumerate(self.concepts) if c.status == "admitted"]

    def primitives(self) -> List[Tuple[Expr, Callable[[Q, Q], Optional[Q]]]]:
        """(expr, exact fn) for every concept, in index order (LIB + k must mean concept k)."""
        return [(c.expr, self.fn(k)) for k, c in enumerate(self.concepts)]

    def text(self, k: int) -> str:
        return render(self.concepts[k].expr, self.names())

    # ----- identity: behaviour on the probes, modulo the base span -----
    def key_of(self, fn: Callable[[Q, Q], Optional[Q]]) -> Optional[Tuple[float, ...]]:
        vals = [fn(x, y) for x, y in PROBES]
        if any(v is None for v in vals):
            return None
        v = np.array([float(u) for u in vals])
        r = v - _base_projector() @ v
        if np.max(np.abs(r)) < 1e-7 * max(1.0, float(np.max(np.abs(v)))):
            return None
        return _key(r / r[np.argmax(np.abs(r))])

    def find(self, fn: Callable[[Q, Q], Optional[Q]]) -> Optional[int]:
        """Index of a stored concept with the same behaviour class, if any (symbol-free identity)."""
        key = self.key_of(fn)
        if key is None:
            return None
        for k in range(len(self.concepts)):
            if self.concepts[k].status != "retired" and self.key_of(self.fn(k)) == key:
                return k
        return None

    def feature(self, k: int) -> Feature:
        fn = self.fn(k)
        vals = tuple(float(fn(x, y)) for x, y in PROBES)
        # A stored concept is one token at level 0: its cost is the choice among concepts, not its body.
        return Feature((LIB + k,), vals, expr_bits((LIB + k,), len(self.concepts)))

    # ----- learning -----
    def record(self, fn: Callable[[Q, Q], Optional[Q]], expr: Expr, task: str,
               evidence: Sequence[Tuple[Tuple[Q, Q], Q]]) -> int:
        """A construction answered and verified a task: store it, or strengthen the stored twin."""
        k = self.find(fn)
        rows = [[str(a), str(b), str(c)] for (a, b), c in evidence]
        if k is None:
            k = len(self.concepts)
            d = asdict(StoredConcept(list(expr), f"c{k}", tasks=[task], successes=1, evidence=rows))
            self.state["concepts"].append(d)
            self.concepts.append(_View(d))  # type: ignore[arg-type]
            return k
        c = self.concepts[k]
        c.successes += 1
        if task not in c.tasks:
            c.tasks.append(task)
        c.evidence = (c.evidence + rows)[-32:]
        if c.status == "candidate" and len(c.tasks) >= 2:
            c.status = "admitted"          # verified on a second, different task: reuse is real
        return k

    def admit(self, k: int) -> None:
        self.concepts[k].status = "admitted"

    def fail(self, k: int) -> None:
        c = self.concepts[k]
        c.failures += 1
        if c.failures - c.successes >= FAILURES_TO_RETIRE:
            c.status = "retired"

    def name(self, k: int, word: str) -> None:
        """Bind a word to a concept (the language interface's 'call it ...')."""
        self.concepts[k].name = word

    def by_name(self, word: str) -> Optional[int]:
        for k, c in enumerate(self.concepts):
            if c.name == word and c.status != "retired":
                return k
        return None

    @staticmethod
    def of(mind: Any) -> "ConceptMemory":
        """A view of the memory living in a Mind's semantic store (created on first use)."""
        return ConceptMemory(mind.knowledge.setdefault("concept_memory", {"concepts": []}))


class _View:
    """Attribute access over one stored-concept dict (writes go to the dict, i.e. to the mind)."""

    __slots__ = ("_d",)

    def __init__(self, d: Dict[str, Any]) -> None:
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name: str) -> Any:
        v = self._d[name]
        return tuple(v) if name == "expr" else v

    def __setattr__(self, name: str, value: Any) -> None:
        self._d[name] = list(value) if name == "expr" else value
