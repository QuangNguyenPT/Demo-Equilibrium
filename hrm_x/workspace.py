"""Workspace: the situation model, as typed terms in an e-graph where every equality has a reason.

A task's current knowledge is one object (spec v0.4 §7):
  - terms f(c1..ck) over e-class ids; constants carry (sort, value);
  - equality a ≡ b closed under congruence (a ≡ b  =>  f(a) ≡ f(b)), as in egglog;
  - a fact is a term equal to True; a function's value is the constant in its class;
  - every union records a Why (rule, assumptions). explain(a, b) returns the reasons, so a
    contradiction (two distinct constants forced into one class) names the assumptions
    behind it instead of just failing.
"B follows from the premises" means B's two sides share a class. Vector similarity never
creates an equality here.

`Workspace` is the Python reference (the blueprint). `hrm_x.native.NativeWorkspace` implements
the same primitives in C++; everything defined on `EGraphAPI` is shared, so both backends
explain, extract and report conflicts through one piece of code. `make_workspace()` picks the
backend (env HRMX_BACKEND = auto | python | native).
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, FrozenSet, Hashable, Iterator, List, Optional, Sequence, Set, Tuple

ENode = Tuple[str, Tuple[int, ...], Hashable]      # (op, child ids, payload)
Term = Tuple[str, Hashable, Tuple["Term", ...]]    # (op, payload, children), as returned by extract()


@dataclass(frozen=True)
class Why:
    """Reason for one union.

    Congruence edges keep the two e-nodes so their arguments can be explained; a rule firing
    keeps `premises`, pairs of e-nodes whose equality the match relied on (e.g. a variable's
    class holding the constant the rule computed with), so explanations reach the facts used.
    """

    rule: str
    assumptions: FrozenSet[str] = frozenset()
    nodes: Tuple[int, int] = (-1, -1)
    premises: Tuple[Tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Conflict:
    """Two different constants were forced equal; `assumptions` are the premises to blame."""

    values: Tuple[Hashable, Hashable]
    assumptions: FrozenSet[str]
    rules: Tuple[str, ...]


def natural(value: Hashable) -> bool:
    return isinstance(value, (int, Fraction)) and not isinstance(value, bool) \
        and Fraction(value).denominator == 1 and value >= 0


class EGraphAPI:
    """Operations defined purely on the primitives; shared by both backends."""

    conflicts: List[Conflict]

    # primitives each backend provides
    def find(self, i: int) -> int: raise NotImplementedError
    def nodes(self, c: int) -> Iterator[Tuple[int, ENode]]: raise NotImplementedError
    def classes(self) -> List[int]: raise NotImplementedError
    def lookup(self, op: str, args: Sequence[int] = (), payload: Hashable = None) -> Optional[int]: raise NotImplementedError
    def const_node(self, c: int) -> Optional[int]: raise NotImplementedError
    def explain(self, a: int, b: int) -> List[Why]: raise NotImplementedError
    def original_args(self, node: int) -> Tuple[int, ...]: raise NotImplementedError
    def payload_of(self, node: int) -> Hashable: raise NotImplementedError

    def const(self, sort: str, value: Hashable) -> int:
        if sort == "Q":
            value = Fraction(value)
        return self.add("const", (), (sort, value))

    def value(self, c: int) -> Optional[Hashable]:
        """The constant in c's class, if the class has one."""
        k = self.const_node(c)
        return None if k is None else self.payload_of(k)[1]

    def same(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def expand(self, why: Why) -> List[Why]:
        """One reason with its congruence arguments / premises explained (used for conflicts)."""
        if why.rule == "congruence":
            x, y = why.nodes
            return [r for ax, ay in zip(self.original_args(x), self.original_args(y)) for r in self.explain(ax, ay)]
        return [why] + [r for px, py in why.premises for r in self.explain(px, py)]

    def record_conflict(self, a: int, b: int, why: Why) -> None:
        ca, cb = self.const_node(a), self.const_node(b)
        reasons = self.explain(ca, a) + self.expand(why) + self.explain(b, cb)
        self.conflicts.append(Conflict((self.payload_of(ca)[1], self.payload_of(cb)[1]),
                                       frozenset().union(*(r.assumptions for r in reasons)),
                                       tuple(r.rule for r in reasons)))

    def assumptions(self, a: int, b: int) -> FrozenSet[str]:
        return frozenset().union(*(w.assumptions for w in self.explain(a, b))) if self.same(a, b) else frozenset()

    def node_id(self, c: int, op: str, payload: Hashable = None) -> Optional[int]:
        """Some e-node id with this op/payload inside c's class (for explanations)."""
        for i, (o, _, p) in self.nodes(c):
            if o == op and p == payload:
                return i
        return None

    def extract(self, c: int, cost: Callable[["EGraphAPI", ENode, List[float]], float]
                ) -> Optional[Tuple[float, Term]]:
        """Cheapest term in c's class under `cost(ws, enode, child_costs)` (fixpoint over classes)."""
        table = [(cls, [node for _, node in self.nodes(cls)]) for cls in self.classes()]
        best: Dict[int, Tuple[float, Term]] = {}
        changed = True
        while changed:
            changed = False
            for cls, nodes in table:
                for node in nodes:
                    op, args, payload = node
                    if any(x not in best for x in args):
                        continue
                    total = cost(self, node, [best[x][0] for x in args])
                    if total < best.get(cls, (float("inf"),))[0] - 1e-12:
                        best[cls] = (total, (op, payload, tuple(best[x][1] for x in args)))
                        changed = True
        return best.get(self.find(c))

    def term_class(self, term: Term) -> Optional[int]:
        """Class of an extracted term, or None if some sub-term is absent."""
        op, payload, kids = term
        ids = []
        for k in kids:
            c = self.term_class(k)
            if c is None:
                return None
            ids.append(c)
        i = self.lookup(op, ids, payload)
        return None if i is None else self.find(i)


class Workspace(EGraphAPI):
    """Python reference e-graph with provenance. `ops` counts every union attempt."""

    native = False

    def __init__(self) -> None:
        self._parent: List[int] = []
        self._enode: List[ENode] = []
        self._members: Dict[int, List[int]] = {}
        self._memo: Dict[ENode, int] = {}
        self._const: Dict[int, int] = {}
        self._proof: Dict[int, List[Tuple[int, Why]]] = {}
        self.conflicts: List[Conflict] = []
        self.ops = 0

    def find(self, i: int) -> int:
        root = i
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[i] != root:
            self._parent[i], i = root, self._parent[i]
        return root

    def add(self, op: str, args: Sequence[int] = (), payload: Hashable = None) -> int:
        """Class of op(args); an existing equal e-node is reused (hash-consing)."""
        node: ENode = (op, tuple(self.find(a) for a in args), payload)
        hit = self._memo.get(node)
        if hit is not None:
            return self.find(hit)
        i = len(self._parent)
        self._parent.append(i)
        self._enode.append(node)
        self._members[i] = [i]
        self._memo[node] = i
        self._proof[i] = []
        if op == "const":
            self._const[i] = i
        return i

    def classes(self) -> List[int]:
        return list(self._members)

    def nodes(self, c: int) -> Iterator[Tuple[int, ENode]]:
        """(id, canonical e-node) for every e-node in c's class."""
        for i in self._members[self.find(c)]:
            op, args, payload = self._enode[i]
            yield i, (op, tuple(self.find(a) for a in args), payload)

    def lookup(self, op: str, args: Sequence[int] = (), payload: Hashable = None) -> Optional[int]:
        """Id of the e-node op(args) if it exists (does not add)."""
        return self._memo.get((op, tuple(self.find(a) for a in args), payload))

    def const_node(self, c: int) -> Optional[int]:
        return self._const.get(self.find(c))

    def original_args(self, node: int) -> Tuple[int, ...]:
        return self._enode[node][1]

    def payload_of(self, node: int) -> Hashable:
        return self._enode[node][2]

    def union(self, a: int, b: int, why: Why) -> bool:
        """Assert a ≡ b. Returns True if the state changed. A constant clash is recorded, not merged."""
        self.ops += 1
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        ca, cb = self._const.get(ra), self._const.get(rb)
        if ca is not None and cb is not None and self._enode[ca][2] != self._enode[cb][2]:
            self.record_conflict(a, b, why)
            return False
        self._proof[a].append((b, why))
        self._proof[b].append((a, why))
        if len(self._members[ra]) < len(self._members[rb]):
            ra, rb, ca, cb = rb, ra, cb, ca          # the constant follows its class across the swap
        self._parent[rb] = ra
        self._members[ra].extend(self._members.pop(rb))
        if ca is None and cb is not None:
            self._const[ra] = cb
        self._const.pop(rb, None)
        return True

    def rebuild(self) -> int:
        """Restore congruence closure after unions; returns the number of congruence merges."""
        merges, changed = 0, True
        while changed:
            changed = False
            memo: Dict[ENode, int] = {}
            for node, i in list(self._memo.items()):
                op, args, payload = node
                canon = (op, tuple(self.find(x) for x in args), payload)
                j = memo.get(canon)
                if j is not None and self.find(j) != self.find(i):
                    if self.union(j, i, Why("congruence", nodes=(j, i))):
                        merges += 1
                        changed = True
                elif j is None:
                    memo[canon] = i
            self._memo = memo
        return merges

    def explain(self, a: int, b: int, _seen: Optional[Set[Tuple[int, int]]] = None) -> List[Why]:
        """Reasons along the proof path a ~ b, with congruence edges expanded into their arguments."""
        if a == b:
            return []
        seen = _seen if _seen is not None else set()
        key = (min(a, b), max(a, b))
        if key in seen:
            return []
        seen.add(key)
        back: Dict[int, Tuple[int, Why]] = {a: (a, Why("start"))}
        queue = deque([a])
        while queue and b not in back:
            u = queue.popleft()
            for v, why in self._proof[u]:
                if v not in back:
                    back[v] = (u, why)
                    queue.append(v)
        if b not in back:
            return []
        reasons: List[Why] = []
        node = b
        while node != a:
            prev, why = back[node]
            if why.rule == "congruence":
                x, y = why.nodes
                for ax, ay in zip(self._enode[x][1], self._enode[y][1]):
                    reasons.extend(self.explain(ax, ay, seen))
            else:
                reasons.append(why)
                for px, py in why.premises:
                    reasons.extend(self.explain(px, py, seen))
            node = prev
        return reasons


def make_workspace() -> EGraphAPI:
    """The native backend when built and allowed (HRMX_BACKEND=auto|native), else the Python reference."""
    choice = os.environ.get("HRMX_BACKEND", "auto")
    if choice != "python":
        from hrm_x import native
        if native.enabled():
            return native.NativeWorkspace()
        if choice == "native":
            raise RuntimeError("HRMX_BACKEND=native but the library is not built (scripts/build_native.ps1)")
    return Workspace()
