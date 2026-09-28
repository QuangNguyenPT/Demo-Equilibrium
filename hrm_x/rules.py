"""Rules: pattern-matching modulo equality (e-matching) and bounded saturation (the fast, L-level step).

A rule is `lhs => rhs` with an optional guard and the assumptions it rests on. Firing adds
the equality lhs ≡ rhs with Why(rule, assumptions); nothing is ever deleted, so every
conclusion stays explainable. rhs may be a pattern or a callable that computes a class,
which is how exact primitive operations (constant folding) enter the same machinery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Hashable, Iterator, List, Optional, Sequence, Tuple, Union

from hrm_x.workspace import Why, Workspace

Subst = Dict[str, int]


@dataclass(frozen=True)
class Lit:
    """Pattern leaf matching the class whose constant equals `value` (or any constant if None)."""

    value: Hashable = None


Pattern = Union[str, Lit, Tuple]          # '?x' | Lit | (op, *children) | (op, payload-marker?)
Builder = Callable[[Workspace, Subst], Optional[int]]


@dataclass(frozen=True)
class Rule:
    name: str
    lhs: Pattern
    rhs: Union[Pattern, Builder]
    guard: Optional[Callable[[Workspace, Subst], bool]] = None
    assumptions: FrozenSet[str] = field(default_factory=frozenset)


def _match(ws: Workspace, pat: Pattern, cls: int, subst: Subst) -> Iterator[Subst]:
    if isinstance(pat, str):
        bound = subst.get(pat)
        if bound is None:
            yield {**subst, pat: ws.find(cls)}
        elif ws.same(bound, cls):
            yield subst
        return
    if isinstance(pat, Lit):
        v = ws.value(cls)
        if v is not None and (pat.value is None or v == pat.value):
            yield subst
        return
    op, *kids = pat
    for _, (nop, args, _payload) in ws.nodes(cls):
        if nop != op or len(args) != len(kids):
            continue
        partial = [subst]
        for kid, arg in zip(kids, args):
            partial = [s2 for s in partial for s2 in _match(ws, kid, arg, s)]
            if not partial:
                break
        yield from partial


class _Snapshot(dict):
    """Marker: the native backend holds this round's operator index itself."""


SNAPSHOT = _Snapshot()


def _python_index(ws: Workspace) -> Dict[str, List[int]]:
    index: Dict[str, set] = {}
    for cls in ws.classes():
        for _, (op, _, _) in ws.nodes(cls):
            index.setdefault(op, set()).add(cls)
    return {op: sorted(c) for op, c in index.items()}


def op_index(ws: Workspace) -> Dict[str, List[int]]:
    """Classes containing at least one e-node with each operator (taken once per round)."""
    if getattr(ws, "native", False):
        ws.snapshot_index()
        return SNAPSHOT
    return _python_index(ws)


def matches(ws: Workspace, rule: Rule, index: Optional[Dict[str, List[int]]] = None) -> List[Tuple[int, Subst]]:
    """Every (anchor e-node, substitution) where the lhs matches and the guard holds.

    The anchor is the e-node the pattern's root matched (so the rule's union is justified by
    that very node, not by whatever else shares its class). A native workspace matches in C++
    (same order as below); patterns it cannot compile use this Python matcher.
    """
    if getattr(ws, "native", False):
        compiled = ws.compile(rule.lhs)
        if compiled is not None:
            hits = ws.ematch(compiled, use_index=index is not None and isinstance(rule.lhs, tuple))
            return [(a, s) for a, s in hits if rule.guard is None or rule.guard(ws, s)]
        if index is SNAPSHOT:
            index = _python_index(ws)
    out = []
    if not isinstance(rule.lhs, tuple):
        for cls in ws.classes():
            for s in _match(ws, rule.lhs, cls, {}):
                if rule.guard is None or rule.guard(ws, s):
                    out.append((cls, s))
        return out
    op, *kids = rule.lhs
    roots = index.get(op, []) if index is not None else ws.classes()
    for cls in roots:
        for nid, (nop, args, _) in ws.nodes(cls):
            if nop != op or len(args) != len(kids):
                continue
            partial = [{}]
            for kid, arg in zip(kids, args):
                partial = [s2 for s in partial for s2 in _match(ws, kid, arg, s)]
                if not partial:
                    break
            for s in partial:
                if rule.guard is None or rule.guard(ws, s):
                    out.append((nid, s))
    return out


def premises(ws: Workspace, subst: Subst) -> Tuple[Tuple[int, int], ...]:
    """(variable's class, its constant) pairs: the values a rule may have computed with."""
    out = []
    for cls in subst.values():
        k = ws.const_node(cls)
        if k is not None and k != cls:
            out.append((cls, k))
    return tuple(out)


def instantiate(ws: Workspace, pat: Union[Pattern, Builder], subst: Subst) -> Optional[int]:
    if callable(pat) and not isinstance(pat, (str, Lit, tuple)):
        return pat(ws, subst)
    if isinstance(pat, str):
        return subst[pat]
    if isinstance(pat, Lit):
        raise ValueError("a literal pattern cannot be built without a sort; use a builder")
    op, *kids = pat
    ids = [instantiate(ws, k, subst) for k in kids]
    if any(i is None for i in ids):
        return None
    return ws.add(op, ids)


def existing(ws: Workspace, pat: Pattern, subst: Subst) -> Optional[int]:
    """Class of the instantiated pattern if every sub-term already exists; never adds terms."""
    if isinstance(pat, str):
        return subst.get(pat)
    if not isinstance(pat, tuple):
        return None
    op, *kids = pat
    ids = [existing(ws, k, subst) for k in kids]
    if any(i is None for i in ids):
        return None
    hit = ws.lookup(op, ids)
    return None if hit is None else ws.find(hit)


def close(ws: Workspace, rules: Sequence[Rule]) -> int:
    """Congruence closure under rule instances between existing terms only (QuickSpec pruning)."""
    merges, changed = 0, True
    while changed:
        changed = False
        index = op_index(ws)
        for rule in rules:
            for root, s in matches(ws, rule, index):
                other = existing(ws, rule.rhs, s)
                if other is not None and ws.union(root, other, Why(rule.name, rule.assumptions, premises=premises(ws, s))):
                    merges += 1
                    changed = True
        if ws.rebuild():
            changed = True
    return merges


@dataclass
class Saturation:
    firings: int
    rounds: int
    saturated: bool


def saturate(ws: Workspace, rules: Sequence[Rule], max_firings: int, max_rounds: int = 64) -> Saturation:
    """Fire all matching rules round by round until nothing changes or the budget is spent."""
    firings = 0
    for rounds in range(1, max_rounds + 1):
        changed = False
        index = op_index(ws)
        for rule in rules:
            for root, s in matches(ws, rule, index):
                if firings >= max_firings:
                    return Saturation(firings, rounds, False)
                new = instantiate(ws, rule.rhs, s)
                firings += 1
                why = Why(rule.name, rule.assumptions, premises=premises(ws, s))
                if new is not None and ws.union(root, new, why):
                    changed = True
        if ws.rebuild():
            changed = True
        if not changed:
            return Saturation(firings, rounds, True)
    return Saturation(firings, max_rounds, False)
