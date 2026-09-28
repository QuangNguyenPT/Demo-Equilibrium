"""Induction: hypotheses about an unknown operator, and laws about a known one (spec v0.4 §8).

Both rest on one idea, observational equivalence: evaluate candidate programs (or terms) on
shared probe inputs and group identical fingerprints.
  - hypotheses: programs consistent with the evidence, grouped into behaviour classes; a class's
    prior mass is the sum of 2^-size over its members (shorter programs count more);
  - laws: terms in one class give conjectured equations (QuickSpec). A conjecture is kept only
    if the laws accepted so far do not already derive it by saturation in a Workspace.
Tested laws are conjectures with a stated domain, not proofs: a law false on a fraction ε of
the probe domain survives n independent probes with probability at most (1-ε)^n.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from hrm_x.rules import Rule, close
from hrm_x.workspace import EGraphAPI, make_workspace

@dataclass
class Hypothesis:
    key: str
    program: Any
    size: int
    weight: float
    members: int = 1


def _fingerprint(program: Any, evaluate: Callable[[Any, Any], Any], probes: Sequence[Any]) -> Tuple:
    return tuple(evaluate(program, p) for p in probes)


def behaviour_classes(programs: Iterable[Tuple[Any, int]], evaluate: Callable[[Any, Any], Any],
                      probes: Sequence[Any]) -> List[Hypothesis]:
    """Group (program, size) pairs by fingerprint; the first (smallest) program represents its class."""
    classes: Dict[Tuple, Hypothesis] = {}
    for program, size in programs:
        fp = _fingerprint(program, evaluate, probes)
        if all(v is None for v in fp):
            continue
        h = classes.get(fp)
        if h is None:
            classes[fp] = Hypothesis(repr(program), program, size, 2.0 ** -size)
        else:
            h.weight += 2.0 ** -size
            h.members += 1
    return list(classes.values())


def consistent(hypotheses: Sequence[Hypothesis], evidence: Sequence[Tuple[Any, Any]],
               evaluate: Callable[[Any, Any], Any]) -> List[Hypothesis]:
    return [h for h in hypotheses if all(evaluate(h.program, x) == y for x, y in evidence)]


# ---------- law discovery (theory exploration) ----------

Pattern = Any


def pattern_size(p: Pattern) -> int:
    return 1 if not isinstance(p, tuple) else 1 + sum(pattern_size(k) for k in p[1:])


def _vars(p: Pattern) -> set:
    return {p} if isinstance(p, str) else set().union(*(_vars(k) for k in p[1:]))


def _build(ws: EGraphAPI, p: Pattern) -> int:
    if isinstance(p, str):
        return ws.add("var", (), p)
    op, *kids = p
    return ws.add(op, [_build(ws, k) for k in kids])


@dataclass(frozen=True)
class Law:
    lhs: Pattern
    rhs: Pattern
    probes: int

    def __str__(self) -> str:
        return f"{_show(self.lhs)} = {_show(self.rhs)}"


def _show(p: Pattern) -> str:
    if isinstance(p, str):
        return p.lstrip("?")
    op, *kids = p
    return f"{op}({', '.join(_show(k) for k in kids)})"


def discover_laws(universe: Sequence[Pattern], evaluate: Callable[[Pattern, Any], Any],
                  probes: Sequence[Any], make_rules: Callable[[Law], List[Rule]]) -> List[Law]:
    """Conjecture equations by testing, then keep only those not derivable from earlier ones.

    "Derivable" is QuickSpec's decidable approximation: congruence closure over instances of the
    accepted laws whose both sides are terms of the universe.
    """
    classes: Dict[Tuple, List[Pattern]] = {}
    for term in universe:
        fp = tuple(evaluate(term, env) for env in probes)
        if any(v is None for v in fp):
            continue                      # a law must hold wherever it is tested: partial terms are out
        classes.setdefault(fp, []).append(term)
    candidates = []
    for members in classes.values():
        members.sort(key=pattern_size)
        candidates += [(members[0], other) for other in members[1:]]
    # QuickSpec order: smaller first, and among equal sizes the more general (more distinct variables)
    candidates.sort(key=lambda e: (pattern_size(e[0]) + pattern_size(e[1]), -len(_vars(e[0]) | _vars(e[1])), repr(e)))
    ws = make_workspace()
    ids = {term: _build(ws, term) for term in universe}
    laws: List[Law] = []
    for lhs, rhs in candidates:
        if ws.same(ids[lhs], ids[rhs]):
            continue
        law = Law(lhs, rhs, len(probes))
        laws.append(law)
        close(ws, make_rules(law))        # only the new law's instances; congruence propagates the rest
    return laws
