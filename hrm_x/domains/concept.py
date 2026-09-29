"""The v0.5 concept route: structural operator inference behind the v0.4 mind.

`hrm_x/domains/arith.py` infers an operator by *searching* it: it enumerates
programs up to a size bound, evaluates them on shared probes, and groups the
observationally equivalent ones into behaviour classes. That works, and ADR-008
measured it working -- but the hypothesis language is a wall of DSL handed to the
mind in advance, and audit finding F3 measured the consequence: on operators the
DSL does not contain, v0.4 answered `3 of the 4` out-of-language probes **with a
confident wrong value** rather than abstaining.

v0.5 changes what a hypothesis *is*. A hypothesis is a **family**
`m(x, y) = sum_j p_j psi_j(x, y)` that is the exact solution set of a law set
(Cauchy-type functional equations). The rational parameters `p` are fitted
exactly by linear solve, so there is no search and no program to invent. Three
things are then computed per family, and they are what close the wrong-answer
hole:

  - **consistency** -- the evidence lies in the family (exact solve over Q);
  - **evidence weight** -- the Bayes factor `|Y|^(n - rank)` (Prop. 4): fitted
    parameters buy no evidence, only the *surplus* rows do;
  - **determinacy** -- Theorem 3: the family has an opinion at the query only if
    `psi(query)` lies in the row space of the evidence matrix. Otherwise it says
    nothing there, and its mass is reported as undetermined rather than guessed
    at.

Plus the **open-world coverage rule**: moving an argument to a value that never
varied in the evidence is supported only if that argument varied, or the reading
is symmetric in `x, y` and the other one varied. Any `g(x) h(y)` with `h` fixed
on the seen values fits evidence in which `y` never moved, so without this rule
the mind is confidently wrong the moment a query leaves the training numbers.
And a reserved hypothesis **BOTTOM**, "outside my language", keeps prior mass
`2^-bits_bottom`, so a corpus that no family explains is not explained by the
family that happens to fit best.

This module is a `Domain` implementation, not a second mind. The H-level loop is
still `Mind.solve`: the same value-of-computation controller, the same modulators,
the same patience, the same trace writer. The delivery's standalone
`concepts.decide()` is the reference implementation the upstream numbers were
first measured against; it is deliberately not imported, because two controllers
in one repository is one too many, and a head-to-head is only fair if both
systems are driven by the same loop.

`derive` here is a table lookup, not a saturation: the fit already happened in
`propose` and its cost is charged there. That asymmetry is the point of the
route -- the L level stops being a search.

Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from hrm_x.concepts import (
    FAMILIES,
    QUESTION_POOL,
    Belief,
    ConceptConfig,
    Family,
    Fitted,
    fit,
)
from hrm_x.mind import UNKNOWN, Hyp, Mind, Result
from hrm_x.qlinalg import solve

Q = Fraction
Pair = Tuple[Q, Q]
Datum = Tuple[Pair, Q]


def reading_key(fitted: Fitted) -> str:
    """Identity of a reading: the family plus its exact fitted parameters.

    The family name alone is not enough. Two macro families built by
    `hrm_x.consolidation` can share a name and differ in their parameters, and
    those are different hypotheses with different weights.
    """
    return f"{fitted.family.name} | " + ", ".join(str(p) for p in fitted.params)


@dataclass
class ConceptTask:
    """A concept: a symbol over rationals, worked examples, and one query.

    Mirrors `ArithTask` -- `examples` is what was observed, `observed` is what
    the mind paid to ask for, and `evidence()` is the union. An `oracle` is the
    teacher; without one the mind has nobody to ask.
    """

    symbol: str
    examples: List[Tuple[Q, Q, Q]]
    question: Pair
    oracle: Optional[Callable[[Q, Q], Q]] = None
    observed: List[Datum] = field(default_factory=list)

    def evidence(self) -> List[Datum]:
        return [((a, b), c) for a, b, c in self.examples] + list(self.observed)

    @staticmethod
    def from_op(op: Callable[[Q, Q], Q], examples: Sequence[Tuple[Q, Q]],
                question: Pair, symbol: str = "?",
                oracle: Optional[Callable[[Q, Q], Q]] = None) -> "ConceptTask":
        """Build a task from the operator itself, so the ground truth is never
        re-implemented by hand in a way the induction could copy."""
        return ConceptTask(symbol, [(a, b, op(a, b)) for a, b in examples],
                           question, oracle if oracle is not None else op)


class Concept:
    """The v0.5 domain: `name = "concept"`."""

    name = "concept"

    def __init__(self, cfg: Optional[ConceptConfig] = None,
                 families: Sequence[Family] = FAMILIES) -> None:
        self.cfg = cfg if cfg is not None else ConceptConfig()
        self.families = list(families)
        # Reading keys already proposed for a task, so a later propose returns
        # only what new evidence made available rather than duplicating the
        # version space. Keyed by `id(task)` for the same reason `Arith` keys its
        # plan cache: the task object is the lifetime.
        self._seen: Dict[int, set] = {}

    # ----- the L level: one exact fit, charged once -----

    def belief(self, task: ConceptTask) -> Belief:
        return fit(task.evidence(), self.cfg, self.families)

    def propose_cost(self, task: ConceptTask, level: int) -> float:
        """One exact fit: every family, every evidence row, its RREF.

        `families` has 68 members whose features number up to 4, and each fit is
        an exact RREF of the augmented matrix, so this is priced as
        `|G| * n * k^2` Fraction operations rather than by the flat constant the
        search route needed for its own enumeration.
        """
        n = max(1, len(task.evidence()))
        k = max(1, max(len(f.features) for f in self.families))
        return float(len(self.families) * n * k * k)

    def propose(self, task: ConceptTask, level: int,
                mind: Mind) -> Optional[Tuple[List[Hyp], int]]:
        """Fit every family, then hand the controller the readings it has not
        seen yet for this task.

        Returning `None` marks the language exhausted, which is what makes the
        controller stop proposing. Returning an empty list instead would leave
        the level looking productive forever. And the honest statement of the
        assumption ADR-008 named as most dangerous is that this language is
        *given*: no amount of evidence here invents a family that was not
        written down.
        """
        bel = self.belief(task)
        post, _ = bel.posterior()
        scale = max(post) or 1.0
        seen = self._seen.setdefault(id(task), set())
        out: List[Hyp] = []
        for fitted, p in zip(bel.fitted, post):
            if p <= 0.0:
                continue
            key = reading_key(fitted)
            if key in seen:
                continue
            seen.add(key)
            out.append(Hyp(key=key, weight=p / scale, payload=fitted,
                           note=f"{fitted.surplus} surplus rows, rank {fitted.rank}"))
        if not out:
            return None, 0
        return out, bel.ops

    # ----- deriving is a lookup: the solve already happened -----

    def derive_cost(self, task: ConceptTask, hyp: Hyp, mind: Mind) -> float:
        """One row-space membership test: an RREF of the evidence matrix against
        one extra row, which is `O(n^2 k)` Fraction operations."""
        fitted: Fitted = hyp.payload
        n = len(fitted.A) + 1
        return float(n * n * max(1, len(fitted.family.features)))

    def derive(self, task: ConceptTask, hyp: Hyp, mind: Mind) -> int:
        fitted: Fitted = hyp.payload
        x, y = task.question
        value = fitted.value(x, y, self.cfg.coverage)
        hyp.derived = True
        ops = len(fitted.A) + 1
        if value is None:
            hyp.answer = UNKNOWN
            hyp.note = ("an argument here never varied in the evidence"
                        if not fitted.covered(x, y)
                        else "this reading is not determined at the question")
        else:
            hyp.answer = value
            hyp.note = ""
        return ops

    def predict(self, task: ConceptTask, hyp: Hyp) -> Tuple[Any, int]:
        """For the controller's derive-group action. Ops are charged only the
        first time a hypothesis is asked, or the macro-action would be priced for
        a lookup it can already do."""
        fitted: Fitted = hyp.payload
        value = fitted.value(task.question[0], task.question[1], self.cfg.coverage)
        spent = 0 if hyp.derived else len(fitted.A) + 1
        return (UNKNOWN, spent) if value is None else (value, spent)

    # ----- verification in a different representation -----

    def verify(self, task: ConceptTask, hyp: Hyp, mind: Mind) -> int:
        """Re-derive the same number by the *dual* of the solve that produced it.

        `derive` evaluated `sum_j p_j r_j`, where `p` solves the primal
        `A p = b`. Theorem 3 says that number is determined only because `r` is
        in the row space of `A` -- equivalently, only because the dual system
        `A^T lambda = r` is consistent. So solve the dual, read the answer back
        as `sum_i lambda_i b_i`, and require the two to be *exactly* equal as
        rationals.

        This is not the same computation twice. The second number is derived
        from the right-hand side rather than from the parameters, so a transposed
        basis, a free-variable artefact or a contaminated parameter vector shows
        up as a disagreement. It also fails for the right reason when the query
        genuinely is undetermined, because `solve` returns `None` on an
        inconsistent dual rather than inventing a least-squares answer.
        """
        fitted: Fitted = hyp.payload
        if hyp.answer is None or hyp.answer == UNKNOWN:
            hyp.verified = False
            hyp.note = "no value to re-derive"
            return 0
        r = fitted.family.row(task.question[0], task.question[1])
        # `A` was built from the evidence as it stood at fit time. If the mind
        # has since paid for a new datum, `task.evidence()` is longer, and its
        # first `len(A)` entries are the ones the parameters were fitted on.
        b = [z for _, z in task.evidence()][:len(fitted.A)]
        k = max(1, len(fitted.family.features))
        ops = 2 * (len(fitted.A) + 1) ** 2 * k
        if r is None:
            hyp.verified = False
            hyp.note = "the family has no value here to re-derive"
            return ops
        dual = solve([list(col) for col in zip(*fitted.A)], list(r))
        if dual is None:
            hyp.verified = False
            hyp.note = "the query row is not in the evidence row space"
            return ops
        value = sum((lam * z for lam, z in zip(dual, b)), Q(0))
        hyp.verified = (value == hyp.answer)
        hyp.note = "" if hyp.verified else f"the dual says {value}, the primal said {hyp.answer}"
        return ops

    # ----- asking -----

    def questions(self, task: ConceptTask, hyps: List[Hyp], mind: Mind):
        """`(question, at_q, goals, cost)` tuples, as the `Domain` protocol asks.

        `at_q` is every reading's *imagined* answer at the candidate question
        and `goals` is every reading's answer at the **real** query; the
        controller pairs them to value the question. `coverage=False` is
        deliberate: a reading is scored on whether it would answer differently
        from the others, not on whether it is currently entitled to answer. A
        question nobody disagrees about buys no information.
        """
        if task.oracle is None:
            return []
        asked = {pq for pq, _ in task.evidence()}
        goals = [self._imagined(h, task.question) for h in hyps]
        out = []
        for pq in QUESTION_POOL:
            if pq in asked:
                continue
            at_q = [self._imagined(h, pq) for h in hyps]
            if len({str(v) for v in at_q}) > 1:
                out.append((pq, at_q, goals, len(hyps)))
        return out

    @staticmethod
    def _imagined(hyp: Hyp, pq: Pair) -> Optional[Q]:
        """What this reading would say at `pq` if it were entitled to."""
        return hyp.payload.value(pq[0], pq[1], coverage=False)

    def ask(self, task: ConceptTask, question: Pair, hyps: List[Hyp], mind: Mind) -> int:
        """Buy one datum and re-weight the version space against it.

        The weights are *refit*, not hand-pruned: a family that cannot fit the
        new evidence drops out of the belief on its own, and a family that can
        fit it but whose new parameters disagree with the old ones keeps a
        correspondingly smaller weight. Setting weights by hand would be a second
        inference rule living next to the first one.
        """
        if task.oracle is None:
            return 0
        answer = task.oracle(*question)
        task.observed.append((question, answer))
        bel = self.belief(task)
        post, _ = bel.posterior()
        by_key = {reading_key(f): p for f, p in zip(bel.fitted, post)}
        for h in hyps:
            h.weight = by_key.get(h.key, 0.0)
            if h.weight == 0.0:
                h.note = f"refuted by {question[0]} {task.symbol} {question[1]} = {answer}"
        return len(hyps)

    # ----- memory -----

    def remember(self, task: ConceptTask, result: Result, mind: Mind) -> None:
        """Keep the concept's version space -- every reading the evidence has not
        refuted, with its weight, plus the evidence -- not just the winner.

        Parameters are stored as strings because `Mind.save` writes JSON and a
        `Fraction` is not a JSON scalar. That is the whole reason the memory
        format is not the posterior object itself.
        """
        if result.status != "answer" or not any(
                h.verified and h.answer == result.answer for h in result.hypotheses):
            return
        memory = self._memory(mind)
        entry = memory["concepts"].setdefault(task.symbol, {"readings": [], "evidence": []})
        seen: List[List[str]] = []
        known = set()
        for datum in task.evidence():
            encoded = self._encode_datum(datum)
            if tuple(encoded) not in known:
                known.add(tuple(encoded))
                seen.append(encoded)
        alive = []
        for h in result.hypotheses:
            if h.weight <= 0 or h.verified is False:
                continue
            fitted: Fitted = h.payload
            alive.append({"key": h.key,
                          "family": fitted.family.name,
                          "params": [str(p) for p in fitted.params],
                          "weight": h.weight,
                          "answer": None if h.answer is None else str(h.answer)})
        entry["readings"] = sorted(alive, key=lambda r: -r["weight"])
        entry["evidence"] = seen
        memory["uses"][task.symbol] = memory["uses"].get(task.symbol, 0) + 1

    @staticmethod
    def _encode_datum(d: Datum) -> List[str]:
        (a, b), c = d
        return [str(a), str(b), str(c)]

    @staticmethod
    def _memory(mind: Mind) -> Dict[str, Any]:
        store = mind.knowledge.setdefault("concepts", {})
        store.setdefault("concepts", {})
        store.setdefault("uses", {})
        return store

    def ambiguity(self, mind: Mind, symbol: str) -> float:
        """Entropy in bits of the stored version space's answers -- the same
        quantity `Arith.ambiguity` reports, so the two routes are comparable
        without translating between scales."""
        import math

        readings = self._memory(mind)["concepts"].get(symbol, {}).get("readings", [])
        total = sum(r["weight"] for r in readings)
        if total <= 0:
            return 0.0
        dist: Dict[str, float] = {}
        for r in readings:
            dist[str(r["answer"])] = dist.get(str(r["answer"]), 0.0) + r["weight"]
        return -sum((w / total) * math.log2(w / total) for w in dist.values() if w > 0)
