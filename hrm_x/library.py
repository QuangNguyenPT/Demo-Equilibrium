"""Learned primitives: the DSL grows by compressing the programs that are already known.

Why this exists. `Arith` identifies a concept by enumerating every program up to a size bound and
grouping observationally equivalent ones. A concept like multiplication is found, but only as a
program whose size sits near the bound, so the enumeration is the expensive part. The fix is not a
better enumerator; it is to stop paying the enumerator again for structure that has already been
paid for. Once a loop step is named, the same concept is expressible in a much smaller language, and
a smaller bound is a smaller search.

The rule is minimum description length, as in DreamCoder. Naming a body `t` with holes `H` costs
`|t| + 1` to declare, and saves

    |t| - 1 - sum(|arg|) over arg in the arguments of that use

at every use, because one application stands in for the whole subtree. Both terms matter and the
first version of this file got it wrong: it assumed a use saved `|t| - 1`, so abstracting
`add(a, x)` -- whose only argument is the single node `a` -- appeared to save one node per use when in
fact it saved nothing, and it declined every primitive the arith domain actually needs. A name is
only worth having if the arguments it replaces are bigger than the arguments it takes.

Which variables become holes is a choice, so it is searched rather than assumed. For a body with
free variables `V` every non-empty subset of `V` is tried and the best gain wins; a hole that occurs
more than once in the body is rejected, because then "the" argument is ambiguous. The usual winner
in arith binds `a` and captures `x`: `add(a, x)` becomes `step(a)`, which is one node smaller.

Four properties are load-bearing, and all four are tested in tests/test_x_library.py:

* `expand(abstract(p)) == p` for every accepted abstraction. A primitive is a name, never a
  semantic change, so the derived program is still the program that was verified.
* Abstraction is therefore value-preserving, structurally, rather than by agreement on a sample.
* Only strictly positive gain is accepted, so a library entry pays for itself on the corpus that
  produced it and the library can never add complexity for free.
* Mining is repeated on the *encoded* corpus. After one primitive is taken the remaining structure
  is a different candidate; mining the raw corpus each pass would re-propose the same abstraction
  forever and never see anything else.

This module is deliberately free of arithmetic. It manipulates the shape of a program and is handed
an evaluator by the domain, so the same growth rule can compress a grid or a story program.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple


def children(prog: Tuple) -> List[Tuple]:
    return [c for c in prog[1:] if isinstance(c, tuple)] if isinstance(prog, tuple) else []


def size(prog: Tuple) -> int:
    """Node count: a terminal is 1, an application is 1 plus its tuple children."""
    return 1 + sum(size(c) for c in children(prog)) if isinstance(prog, tuple) else 1


def is_prim(prog: Tuple) -> bool:
    return isinstance(prog, tuple) and bool(prog) and prog[0] == "prim"


def leaves(prog: Tuple) -> FrozenSet[str]:
    """The free variables a subtree mentions.

    A terminal is a tuple with no tuple children, and it is a variable only when it is a bare
    one-element tag: `("x",)` is x, while `("c", 3)` is a constant with no name. Treating either as
    an internal node reports no variables at all, which is the difference between a step that can
    bind `a` and one that cannot.
    """
    if not isinstance(prog, tuple):
        return frozenset({prog}) if isinstance(prog, str) else frozenset()
    kids = children(prog)
    if not kids:
        return frozenset({prog[0]}) if len(prog) == 1 and isinstance(prog[0], str) else frozenset()
    out: set = set()
    for c in kids:
        out |= leaves(c)
    return frozenset(out)


def tags_of(prog: Tuple) -> FrozenSet[str]:
    """Every operator or variable tag in a program, terminals included."""
    if not isinstance(prog, tuple):
        return frozenset({prog}) if isinstance(prog, str) else frozenset()
    out = {prog[0]} if isinstance(prog[0], str) else set()
    for c in children(prog):
        out |= tags_of(c)
    return frozenset(out)


def arg_slots(body: Tuple, holes: Sequence[str]) -> Optional[Tuple[int, ...]]:
    """Child positions of `body` holding exactly the leaf `(h,)` for each hole, in `holes` order.

    None if a hole is absent or occurs more than once, in which case "the argument at that use" is
    not well defined and the abstraction is not offered.
    """
    if not isinstance(body, tuple):
        return None
    slots = []
    for h in holes:
        pos = [i for i, c in enumerate(body[1:]) if c == (h,)]
        if len(pos) != 1:
            return None
        slots.append(pos[0])
    return tuple(slots)


def arg_sizes(body: Tuple, slots: Sequence[int]) -> List[int]:
    return [size(body[i + 1]) for i in slots]


def occurrences(prog: Tuple) -> Counter:
    """Every subtree of `prog`, counted, the program itself included."""
    counts: Counter = Counter()
    stack = [prog]
    while stack:
        node = stack.pop()
        if not isinstance(node, tuple):
            continue
        counts[node] += 1
        stack.extend(children(node))
    return counts


def apply(prim: "Primitive", args: Sequence[Tuple]) -> Tuple:
    if len(args) != len(prim.holes):
        raise ValueError(f"{prim.name} takes {len(prim.holes)} arguments, got {len(args)}")
    return ("prim", prim.name) + tuple(args)


def show(prog: Tuple) -> str:
    if not isinstance(prog, tuple):
        return str(prog)
    if is_prim(prog):
        return f"{prog[1]}({', '.join(show(c) for c in prog[2:])})"
    if prog and prog[0] == "c":
        return str(prog[1])
    if not children(prog):
        return str(prog[0])
    return f"{prog[0]}({', '.join(show(c) for c in prog[1:] if isinstance(c, tuple))})"


@dataclass
class Primitive:
    """One learned name. `body` is in the base language; `holes` are the arguments it takes."""

    name: str
    body: Tuple
    holes: Tuple[str, ...]
    slots: Tuple[int, ...] = ()
    uses: int = 0
    corpus: int = 0          # corpus size when it was mined, for the report

    def __post_init__(self) -> None:
        if not self.slots:
            found = arg_slots(self.body, self.holes)
            if found is None:
                raise ValueError(f"{self.name}: holes {self.holes} are not single occurrences "
                                 f"in {show(self.body)}")
            self.slots = found

    @property
    def cost(self) -> int:
        """Description length of declaring it: the body, plus one for the name."""
        return size(self.body) + 1

    @property
    def saving(self) -> int:
        """Nodes removed at one use: the body, minus the application and its arguments."""
        return size(self.body) - 1 - sum(arg_sizes(self.body, self.slots))

    def gain(self, count: int) -> int:
        return count * self.saving - self.cost

    def show(self) -> str:
        if not self.holes:
            return show(self.body)
        params = ", ".join(h.upper() for h in self.holes)
        return f"{self.name}({params}) = {show(self.body)}"


def _fill(body: Tuple, holes: Tuple[str, ...], args: Sequence[Tuple]) -> Tuple:
    """Substitute hole i with args[i] throughout `body`.

    A hole is a bare variable tag `("a",)`, so the substitution is on `body[0]` and not on `body`
    itself: matching the tuple against the mapping silently misses every hole and returns the body
    unchanged, which looks like a working expansion because the primitive is simply never applied.
    Holes are distinct variables, so one pass with a mapping is unambiguous.
    """
    table = dict(zip(holes, args))
    if not isinstance(body, tuple):
        return body
    if len(body) == 1 and isinstance(body[0], str) and body[0] in table:
        return table[body[0]]
    return tuple(_fill(c, holes, args) for c in body)


def expand(prog: Tuple, library: Sequence[Primitive]) -> Tuple:
    """Every `("prim", name, ...)` replaced by the program it names, recursively.

    This is the exact inverse of `abstract`, and the whole correctness argument for a primitive is
    that it is only a name: the expanded program is byte-for-byte the program that was verified.
    An unknown name is a KeyError rather than a pass-through, because silently returning an
    unexpanded program would let an un-evaluated `("prim", ...)` reach the evaluator.
    """
    table = {p.name: p for p in library}

    def go(node: Any) -> Tuple:
        # Loop, not recurse: filling a body can expose a primitive nested inside it, and that one
        # has to be expanded too before the node is back in the base language.
        while is_prim(node):
            prim = table[node[1]]
            if len(node) - 2 != len(prim.holes):
                raise ValueError(f"{prim.name} applied to {len(node) - 2} arguments, "
                                 f"wants {len(prim.holes)}")
            node = _fill(prim.body, prim.holes, node[2:])
        if not isinstance(node, tuple):
            return node
        return tuple(go(c) for c in node)

    return go(prog)


def _contains(prog: Tuple, sub: Tuple) -> bool:
    if not isinstance(prog, tuple):
        return False
    return prog == sub or any(_contains(c, sub) for c in children(prog))


def abstract_once(prog: Tuple, prim: Primitive) -> Optional[Tuple]:
    """`prog` with one maximal occurrence of `prim.body` replaced by an application. None if absent.

    Only maximal occurrences are taken, so a name never ends up nested inside its own definition.
    """
    if not isinstance(prog, tuple) or is_prim(prog):
        return None
    if prog == prim.body:
        return apply(prim, [prog[i + 1] for i in prim.slots])
    for i, c in enumerate(prog[1:]):
        if is_prim(c):
            continue
        if c == prim.body and not _contains(prog[:i + 1] + prog[i + 2:], prim.body):
            args = [c[j + 1] for j in prim.slots]
            return prog[:i + 1] + (apply(prim, args),) + prog[i + 2:]
        inner = abstract_once(c, prim)
        if inner is not None:
            return prog[:i + 1] + (inner,) + prog[i + 2:]
    return None


def abstract_all(prog: Tuple, library: Sequence[Primitive]) -> Tuple:
    """Apply every primitive in the library that occurs, most saving first."""
    out = prog
    for prim in sorted(library, key=lambda p: -p.saving):
        while True:
            nxt = abstract_once(out, prim)
            if nxt is None:
                break
            out = nxt
    return out


def candidates(programs: Iterable[Tuple], arity_vars: Sequence[str] = ("x", "y", "a"),
               tags: Optional[Collection[str]] = None,
               max_subsets: int = 8) -> List[Tuple[Tuple, Tuple[str, ...], int, int]]:
    """`(body, holes, count, best_gain)` for every binding of every repeated subtree.

    A subtree using a tag outside `arity_vars` cannot be given a hole, so it is skipped -- otherwise
    expansion could not fill it in. `tags` is how a caller states the base language strictly: pass
    the operator set and a foreign tag is rejected, because expanding it would produce something the
    evaluator cannot run. `tags=None` trusts the corpus and takes every tag it sees, which is the
    right default for a domain whose programs are by construction in its own language.
    For each survivor every non-empty subset of its free variables is priced, since which variables
    become holes is a choice: capturing `x` inside `add(a, x)` is what makes `step(a)` shorter.
    """
    counts: Counter = Counter()
    for p in programs:
        for sub, k in occurrences(p).items():
            if isinstance(sub, tuple) and children(sub):
                counts[sub] += k
    out: List[Tuple[Tuple, Tuple[str, ...], int, int]] = []
    for sub, k in counts.items():
        if leaves(sub) - set(arity_vars):
            continue
        if tags is not None and tags_of(sub) - (set(arity_vars) | set(tags)):
            continue
        free = sorted(leaves(sub))
        best: Optional[Tuple[Tuple[str, ...], int]] = None
        subsets = 0
        for mask in range(1, 1 << len(free)):
            hs = tuple(free[i] for i in range(len(free)) if mask >> i & 1)
            if len(hs) > len(arity_vars) or subsets >= max_subsets:
                continue
            slots = arg_slots(sub, hs)
            if slots is None:
                continue
            subsets += 1
            saving = size(sub) - 1 - sum(arg_sizes(sub, slots))
            if saving <= 0:
                continue
            g = k * saving - (size(sub) + 1)
            if best is None or g > best[1]:
                best = (hs, g)
        if best is not None and best[1] > 0:
            out.append((sub, best[0], k, best[1]))
    return out


@dataclass
class Library:
    """An ordered set of learned primitives, with the accounting the report needs."""

    prims: List[Primitive] = field(default_factory=list)
    min_gain: int = 2            # a primitive must save at least this many nodes overall
    min_count: int = 2           # and appear at least this many times, or it is noise
    credit: float = 0.0          # nodes the library removes per program, measured on its own corpus

    def measure_credit(self, corpus: Sequence[Tuple]) -> float:
        """Nodes the library actually removes, per program, on `corpus`.

        This is the number a search bound can be reduced by, and it is measured rather than derived
        from the primitives' sizes. An earlier version divided the bound by a compression *ratio*
        taken over the primitive bodies alone, which for a step that saves one node gave a factor of
        1.5 and cut the bound from 6 to 4 -- far enough to lose the very concept the library existed
        to find. Saving is in nodes, so the bound moves in nodes.
        """
        if not corpus:
            self.credit = 0.0
            return 0.0
        saved = [size(p) - self.size_of(abstract_all(p, self.prims)) for p in corpus]
        self.credit = sum(max(0, s) for s in saved) / len(saved)
        return self.credit

    def __contains__(self, name: str) -> bool:
        return any(p.name == name for p in self.prims)

    def __len__(self) -> int:
        return len(self.prims)

    def __iter__(self):
        return iter(self.prims)

    def name(self, tag: str) -> str:
        """The next free name, so a library is a function of its own history and not of a counter
        a caller could reset."""
        i = 1
        while f"{tag}{i}" in self:
            i += 1
        return f"{tag}{i}"

    def add(self, prim: Primitive) -> Primitive:
        self.prims.append(prim)
        return prim

    def size_of(self, prog: Tuple) -> int:
        """Description length of `prog` in the extended language; the library is already paid for."""
        if not isinstance(prog, tuple):
            return 1
        if is_prim(prog):
            return 1 + sum(self.size_of(c) for c in prog[2:])
        return 1 + sum(self.size_of(c) for c in children(prog))

    def mdl(self, progs: Sequence[Tuple]) -> int:
        """Total description length: every primitive declared, plus every program in that language."""
        return sum(p.cost for p in self.prims) + sum(self.size_of(p) for p in progs)

    def report(self) -> Dict[str, Any]:
        return {"primitives": [{"name": p.name, "arity": len(p.holes), "body": show(p.body),
                                "saving": p.saving, "uses": p.uses, "corpus": p.corpus}
                               for p in self.prims],
                "n": len(self.prims), "declared_cost": sum(p.cost for p in self.prims)}


def grow(corpus: Sequence[Tuple], lib: Optional[Library] = None, tag: str = "g",
         arity_vars: Sequence[str] = ("x", "y", "a"), tags: Optional[Collection[str]] = None,
         max_arity: int = 2, passes: int = 4, min_count: int = 2, min_gain: int = 2) -> Tuple[Library, List[Dict[str, Any]]]:
    """Learn primitives from programs, greedily by MDL gain. Returns the library and a log.

    Mining is repeated on the encoded corpus: once a primitive is taken, the residual structure is a
    different candidate, and mining the raw corpus each pass would re-propose the same abstraction
    forever. `max_arity` caps the subset search; the useful binding in arith is arity one.
    """
    lib = lib if lib is not None else Library()
    lib.min_count, lib.min_gain = min_count, min_gain
    log: List[Dict[str, Any]] = []
    current = list(corpus)
    for _ in range(passes):
        before = lib.mdl(current)
        best = None
        for sub, hs, k, gain in candidates(current, arity_vars, tags):
            if k < min_count or gain < min_gain or len(hs) > max_arity:
                continue
            if best is None or gain > best[3]:
                best = (sub, hs, k, gain)
        if best is None:
            break
        sub, hs, k, gain = best
        prim = lib.add(Primitive(name=lib.name(tag), body=sub, holes=hs, corpus=len(current)))
        encoded = [abstract_all(p, lib.prims) for p in current]
        application = apply(prim, [(h,) for h in hs])     # a use passes the hole as a program
        for p in encoded:
            prim.uses += occurrences(p).get(application, 0)
        log.append({"name": prim.name, "arity": len(hs), "body": show(sub), "occurrences": k,
                    "saving": prim.saving,                     "gain": gain, "mdl_before": before,
                    "mdl_after": lib.mdl(encoded), "uses": prim.uses})
        current = encoded
    lib.measure_credit(corpus)
    return lib, log
