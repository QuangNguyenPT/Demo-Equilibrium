"""Arithmetic: numbers are exact rationals; an unknown operator symbol is a concept to identify.

Substrate (given, exact, not learned): rational constants, `add`, and `iter(n, body, init)`
(apply body to an accumulator `a`, n times, n a natural number).
Concept (learned): the meaning of a symbol such as `×`, as a program in that language, chosen
by posterior over programs consistent with examples and with answers to questions it asks.
Laws (discovered in sleep): equations tested over a stated domain (x, y natural; z rational).
Derivation: the question becomes an e-graph; place-value notation (hypothesis-checked) and
laws add equal forms; extraction picks the cheapest; exact execution computes it.
Verification: the question is re-evaluated *from the definition* in residue-phasor space
(different representation, closed-form exponentiation instead of iteration), together with
every piece of evidence. It catches wrong laws and wrong execution, not a wrong concept that
fits all evidence (that is what questions are for).
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from hrm_x.induction import Law, behaviour_classes, consistent, discover_laws
from hrm_x.mind import UNKNOWN, Hyp, Mind, Result
from hrm_x.rules import Rule, saturate
from hrm_x.workspace import EGraphAPI as Workspace, Why, make_workspace, natural

# ---------- the program language ----------

X, Y, A = ("x",), ("y",), ("a",)
ZERO, ONE = ("c", Fraction(0)), ("c", Fraction(1))
ITER_CAP = 10 ** 6
UNKNOWN_COST = 100.0     # prior cost of a loop whose count is only known after executing its argument
FOLD_LIMIT = 200         # planning computes applications this cheap exactly, so later counts are known


class Exhausted(Exception):
    pass


class Counter:
    """Counts primitive operations; raises Exhausted past `limit`."""

    def __init__(self, limit: int = 10 ** 7) -> None:
        self.used = 0
        self.limit = limit

    def spend(self, n: int = 1) -> None:
        self.used += n
        if self.used > self.limit:
            raise Exhausted


def run(prog: Tuple, x: Any, y: Any, counter: Counter, a: Any = None) -> Optional[Fraction]:
    """Exact evaluation; None = undefined (a non-natural iteration count)."""
    tag = prog[0]
    if tag == "x":
        return x
    if tag == "y":
        return y
    if tag == "a":
        return a
    if tag == "c":
        return prog[1]
    if tag == "add":
        left, right = run(prog[1], x, y, counter, a), run(prog[2], x, y, counter, a)
        if left is None or right is None:
            return None
        counter.spend()
        return left + right
    n = run(prog[1], x, y, counter, a)
    if n is None or not natural(n) or n > ITER_CAP:
        return None
    acc = run(prog[3], x, y, counter, a)
    for _ in range(int(n)):
        if acc is None:
            return None
        acc = run(prog[2], x, y, counter, acc)
    return acc


def uses(prog: Tuple, tag: str) -> bool:
    return prog[0] == tag or any(isinstance(p, tuple) and uses(p, tag) for p in prog[1:])


def programs(max_size: int) -> Iterator[Tuple[Tuple, int]]:
    """Every program up to max_size, by size. `a` only in a loop body; loops are not nested."""
    outer: Dict[int, List[Tuple]] = {1: [X, Y, ZERO, ONE]}
    body: Dict[int, List[Tuple]] = {1: [X, Y, ZERO, ONE, A]}
    for s in range(2, max_size + 1):
        outer[s], body[s] = [], []
        for table in (outer, body):
            for s1 in range(1, s - 1):
                for left in table[s1]:
                    for right in table.get(s - 1 - s1, []):
                        if repr(left) <= repr(right):
                            table[s].append(("add", left, right))
        for s1 in range(1, s):
            for s2 in range(1, s - s1):
                s3 = s - 1 - s1 - s2
                if s3 < 1:
                    continue
                for n in outer[s1]:
                    for b in body[s2]:
                        if not uses(b, "a") or b == A:
                            continue
                        for init in outer[s3]:
                            outer[s].append(("iter", n, b, init))
    for s in range(1, max_size + 1):
        for p in outer[s]:
            yield p, s


def show(prog: Tuple) -> str:
    tag = prog[0]
    if tag in ("x", "y", "a"):
        return tag
    if tag == "c":
        return str(prog[1])
    if tag == "add":
        return f"({show(prog[1])} + {show(prog[2])})"
    return f"repeat {show(prog[1])} times [a := {show(prog[2])}] from {show(prog[3])}"


def static_cost(prog: Tuple, x: Any, y: Any) -> float:
    """Operations needed to run prog, from known argument values; inf if a loop count is unknown."""
    tag = prog[0]
    if tag in ("x", "y", "a", "c"):
        return 0.0
    if tag == "add":
        return 1.0 + static_cost(prog[1], x, y) + static_cost(prog[2], x, y)
    try:
        n = run(prog[1], x, y, Counter(10 ** 4))
    except Exhausted:
        return math.inf
    if n is None or not natural(n):
        return math.inf
    return float(n) * static_cost(prog[2], x, y) + static_cost(prog[3], x, y) + static_cost(prog[1], x, y)


# ---------- questions as expressions ----------

TOKEN = re.compile(r"\d+(?:\.\d+)?|[()+]|\S")


def parse(text: str, symbol: str) -> Tuple:
    """'10 × 3.1 + 2' -> ('add', ('op', 10, 31/10), 2). The concept symbol binds tighter than '+'."""
    tokens = TOKEN.findall(text)
    pos = 0

    def expr() -> Tuple:
        nonlocal pos
        node = term()
        while pos < len(tokens) and tokens[pos] == "+":
            pos += 1
            node = ("add", node, term())
        return node

    def term() -> Tuple:
        nonlocal pos
        node = factor()
        while pos < len(tokens) and tokens[pos] == symbol:
            pos += 1
            node = ("op", node, factor())
        return node

    def factor() -> Tuple:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            node = expr()
            pos += 1
            return node
        return ("num", Fraction(tok))

    tree = expr()
    if pos != len(tokens):
        raise ValueError(f"cannot parse '{text}' with operator '{symbol}'")
    return tree


def evaluate_tree(tree: Tuple, prog: Tuple, counter: Counter) -> Optional[Fraction]:
    if tree[0] == "num":
        return tree[1]
    left, right = evaluate_tree(tree[1], prog, counter), evaluate_tree(tree[2], prog, counter)
    if left is None or right is None:
        return None
    if tree[0] == "add":
        counter.spend()
        return left + right
    return run(prog, left, right, counter)


def literals(tree: Tuple) -> List[Fraction]:
    return [tree[1]] if tree[0] == "num" else literals(tree[1]) + literals(tree[2])


# ---------- independent verifier: residue phasors ----------

class ResidueCode:
    """Integers mod M as unit phasors (residue hyperdimensional computing, Kymn et al. 2024)."""

    def __init__(self, moduli: Sequence[int] = (101, 103, 107, 109, 113, 127), replicas: int = 8, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.moduli = tuple(moduli)
        self.M = math.prod(moduli)
        chan = np.repeat(np.arange(len(moduli)), replicas)
        self.p = np.array(moduli)[chan]
        self.q = np.array([rng.choice([k for k in range(1, m) if math.gcd(k, m) == 1]) for m in self.p])
        self.chan = chan
        self.ops = 0
        # conj(phi_c(r)) for every residue r of every modulus: decode scores = Re(table @ z_c)
        self._tables = [np.exp(-2j * np.pi * np.outer(np.arange(m), self.q[chan == c]) / m)
                        for c, m in enumerate(self.moduli)]

    def encode(self, n: int) -> np.ndarray:
        return np.exp(2j * np.pi * self.q * (int(n) % self.M) / self.p)

    def cleanup(self, z: np.ndarray) -> np.ndarray:
        """Snap each component to the nearest valid root of unity (exact within sin(pi/p))."""
        j = np.round(np.angle(z) * self.p / (2 * np.pi)).astype(np.int64) % self.p
        return np.exp(2j * np.pi * j / self.p)

    def bind(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.ops += 1
        return self.cleanup(u * v)

    def power(self, u: np.ndarray, k: int) -> np.ndarray:
        """u^k: adding the encoded value k times, in one step (A2)."""
        self.ops += 1
        j = np.round(np.angle(u) * self.p / (2 * np.pi)).astype(np.int64) % self.p
        return np.exp(2j * np.pi * ((j * (k % self.p)) % self.p) / self.p)

    def decode(self, z: np.ndarray) -> int:
        """Per-modulus residue by best match against every candidate phasor (one matrix product), then CRT."""
        self.ops += 1
        residues = []
        for c, table in enumerate(self._tables):
            residues.append(int(np.argmax(np.real(table @ z[self.chan == c]))))
        n = 0
        for r, m in zip(residues, self.moduli):
            k = self.M // m
            n += r * k * pow(k, -1, m)
        n %= self.M
        return n - self.M if n >= self.M // 2 else n


def _free_of_a(prog: Tuple) -> bool:
    return not uses(prog, "a")


def phasor_run(prog: Tuple, xv, yv, scale: int, code: ResidueCode, av=None):
    """The program, evaluated on scaled phasor values (value v is encoded as v * scale)."""
    tag = prog[0]
    if tag == "x":
        return xv
    if tag == "y":
        return yv
    if tag == "a":
        return av
    if tag == "c":
        return code.encode(int(prog[1] * scale))
    if tag == "add":
        left, right = phasor_run(prog[1], xv, yv, scale, code, av), phasor_run(prog[2], xv, yv, scale, code, av)
        return None if left is None or right is None else code.bind(left, right)
    nv = phasor_run(prog[1], xv, yv, scale, code, av)
    if nv is None:
        return None
    scaled = code.decode(nv)
    if scaled < 0 or scaled % scale:
        return None
    n = scaled // scale
    init = phasor_run(prog[3], xv, yv, scale, code, av)
    if init is None:
        return None
    body = prog[2]
    if body[0] == "add" and A in body[1:]:
        other = body[2] if body[1] == A else body[1]
        if _free_of_a(other):
            step = phasor_run(other, xv, yv, scale, code, av)
            return None if step is None else code.bind(init, code.power(step, n))
    acc = init
    for _ in range(n):
        acc = phasor_run(body, xv, yv, scale, code, acc)
        if acc is None:
            return None
    return acc


def phasor_tree(tree: Tuple, prog: Tuple, scale: int, code: ResidueCode):
    if tree[0] == "num":
        return code.encode(int(tree[1] * scale))
    left, right = phasor_tree(tree[1], prog, scale, code), phasor_tree(tree[2], prog, scale, code)
    if left is None or right is None:
        return None
    if tree[0] == "add":
        return code.bind(left, right)
    return phasor_run(prog, left, right, scale, code)


# ---------- the domain ----------

PROBES = [(Fraction(a), Fraction(b)) for a, b in
          [(0, 0), (1, 1), (2, 3), (3, 2), (1, 4), (4, 1), (5, 3), (0, 2), (3, 0)]] + \
         [(Fraction(2), Fraction(1, 2)), (Fraction(3, 2), Fraction(2)), (Fraction(3), Fraction(5, 2)),
          (Fraction(1, 2), Fraction(1, 3))]
QUESTION_POOL = [(Fraction(a), Fraction(b)) for a in range(1, 7) for b in range(1, 7)] + \
                [(Fraction(2), Fraction(1, 2)), (Fraction(3), Fraction(3, 2)), (Fraction(1, 2), Fraction(2)),
                 (Fraction(5, 2), Fraction(3))]
N_VARS = ("?x", "?y")


@dataclass
class ArithTask:
    symbol: str
    examples: List[Tuple[Fraction, Fraction, Fraction]]
    question: str
    oracle: Optional[Callable[[Fraction, Fraction], Fraction]] = None
    observed: List[Tuple[Tuple[Fraction, Fraction], Fraction]] = field(default_factory=list)

    @staticmethod
    def from_text(examples: Sequence[str], question: str, symbol: str = "×",
                  oracle: Optional[Callable[[Fraction, Fraction], Fraction]] = None) -> "ArithTask":
        parsed = []
        for line in examples:
            lhs, rhs = line.split("=")
            a, b = lhs.split(symbol)
            parsed.append((Fraction(a.strip()), Fraction(b.strip()), Fraction(rhs.strip())))
        return ArithTask(symbol, parsed, question, oracle)

    def evidence(self) -> List[Tuple[Tuple[Fraction, Fraction], Fraction]]:
        return [((a, b), c) for a, b, c in self.examples] + list(self.observed)


def is_nat(ws: Workspace, cls: int, _memo: Optional[Dict[int, bool]] = None) -> bool:
    """Sort analysis: a class is natural if it holds a natural constant or is built from naturals."""
    v = ws.value(cls)
    if v is not None:
        return natural(v)
    memo = _memo if _memo is not None else {}
    root = ws.find(cls)
    if root in memo:
        return memo[root]              # False while in progress: breaks cycles without poisoning siblings
    memo[root] = False
    for _, (op, args, payload) in ws.nodes(root):
        if (op == "var" and payload in N_VARS) or \
                (op in ("add", "f") and args and all(is_nat(ws, x, memo) for x in args)):
            memo[root] = True
            break
    return memo[root]


def law_rules(law: Law) -> List[Rule]:
    """Both directions of a law, guarded by its tested domain (?x, ?y natural)."""
    def vars_of(p) -> List[str]:
        return [p] if isinstance(p, str) else [v for k in p[1:] for v in vars_of(k)]

    needs = sorted({v for v in vars_of(law.lhs) + vars_of(law.rhs) if v in N_VARS})

    def guard(ws: Workspace, s: Dict[str, int]) -> bool:
        return all(v not in s or is_nat(ws, s[v]) for v in needs)

    name = f"law:{law}"
    return [Rule(name, law.lhs, law.rhs, guard, frozenset({name})),
            Rule(name, law.rhs, law.lhs, guard, frozenset({name}))]


class Arith:
    name = "arith"

    def __init__(self, max_size: Tuple[int, ...] = (6, 7), rewrite_budget: int = 400) -> None:
        self.max_size = max_size
        self.rewrite_budget = rewrite_budget
        self.code = ResidueCode()
        self._trees: Dict[str, Tuple] = {}
        self._goal_cache: Dict[Tuple[int, str], Any] = {}
        self._plans: Dict[Tuple[int, str], Any] = {}

    # ----- memory -----

    def _memory(self, mind: Mind) -> Dict[str, Any]:
        return mind.knowledge.setdefault(self.name, {"concepts": {}, "laws": {}, "uses": {}})

    def _tree(self, task: ArithTask) -> Tuple:
        key = f"{task.symbol}|{task.question}"
        if key not in self._trees:
            self._trees[key] = parse(task.question, task.symbol)
        return self._trees[key]

    # ----- propose -----

    def propose_cost(self, task: ArithTask, level: int) -> float:
        return 1500.0 if level == 0 else 60000.0

    def propose(self, task: ArithTask, level: int, mind: Mind) -> Optional[Tuple[List[Hyp], int]]:
        known = self._memory(mind)["concepts"].get(task.symbol)
        if known is not None:
            if level > 0:
                return None
            fits = [(p, w) for p, w in known["hypotheses"]
                    if all(_safe_run(p, a, b) == c for (a, b), c in task.evidence())]
            return [Hyp(f"concept:{show(p)}", w, p, note="recalled from semantic memory") for p, w in fits], len(fits)
        if level >= len(self.max_size):
            return None
        counter = Counter(10 ** 9)

        def ev(prog: Tuple, xy: Tuple[Fraction, Fraction]) -> Any:
            try:
                sub = Counter(2000)
                value = run(prog, xy[0], xy[1], sub)
                counter.used += sub.used + 1
                return value
            except Exhausted:
                return None

        classes = behaviour_classes(programs(self.max_size[level]), ev, PROBES)
        fits = consistent(classes, task.evidence(), ev)
        hyps = [Hyp(f"concept:{show(h.program)}", h.weight, h.program, note=f"{h.members} programs") for h in fits]
        return hyps, counter.used

    # ----- derive -----

    def derive_cost(self, task: ArithTask, hyp: Hyp, mind: Mind) -> float:
        """Rewrite cost + execution cost of the cheapest plan (the plan is made once and cached)."""
        ws, _, _, exec_cost = self._plan(task, hyp, mind)
        return ws.ops + exec_cost

    def _plan(self, task: ArithTask, hyp: Hyp, mind: Mind):
        """Think about how to compute before computing: e-graph + notation + laws, then extraction."""
        key = (id(task), hyp.key)
        if key not in self._plans:
            self._plans[key] = self._make_plan(task, hyp, mind)
        return self._plans[key]

    def derive(self, task: ArithTask, hyp: Hyp, mind: Mind) -> int:
        prog = hyp.payload
        ws, goal, term, _ = self._plan(task, hyp, mind)
        counter = Counter()
        try:
            value = self._execute(term, prog, counter)
        except Exhausted:
            value = None
        top = ws.lookup(term[0], [ws.term_class(k) for k in term[2]], term[1])
        used = ws.assumptions(goal, top) if top is not None else frozenset()
        hyp.derived = True
        hyp.answer = UNKNOWN if value is None else value
        hyp.culprits = frozenset(a for a in used if a.startswith(("law:", "notation")))
        hyp.note = f"{_term(term)} = {hyp.answer} ({counter.used} exact ops, {len(hyp.culprits)} rewrite reason(s))"
        return ws.ops + counter.used

    def _make_plan(self, task: ArithTask, hyp: Hyp, mind: Mind):
        prog, tree = hyp.payload, self._tree(task)
        ws = make_workspace()

        def build(node: Tuple) -> int:
            if node[0] == "num":
                return ws.const("Q", node[1])
            op = "add" if node[0] == "add" else "f"
            return ws.add(op, [build(node[1]), build(node[2])])

        goal = build(tree)                      # a fresh class: its id is also its first e-node's id
        rules = [self._notation_rule(hyp)]
        laws = self._memory(mind)["laws"].get((task.symbol, hyp.key), [])
        for law in laws:
            rules += law_rules(law)
        saturate(ws, rules, max_firings=self.rewrite_budget, max_rounds=8)
        self._fold_cheap(ws, prog, hyp, goal)

        def cost(w: Workspace, node, kids: List[float]) -> float:
            op, args, _ = node
            if op == "const":
                return 1.0
            if op == "add":
                return 1.0 + sum(kids)
            c = static_cost(prog, w.value(args[0]), w.value(args[1]))
            return (c if math.isfinite(c) else UNKNOWN_COST) + sum(kids)   # count not known until executed

        total, term = ws.extract(goal, cost)
        return ws, goal, term, total

    @staticmethod
    def _fold_cheap(ws: Workspace, prog: Tuple, hyp: Hyp, goal: int, rounds: int = 4) -> None:
        """Compute, exactly and before extraction, every concept application whose arguments are known
        constants and whose cost is at most FOLD_LIMIT. Loop counts that other plans depend on then
        become known, so extraction compares true costs instead of guessing (a guess once chose a
        99,070-step plan over a 415-step one)."""
        for _ in range(rounds):
            changed = False
            for cls in ws.classes():
                if ws.value(cls) is not None or ws.same(cls, goal):   # the goal itself is computed by execution
                    continue
                for _, (op, args, _) in ws.nodes(cls):
                    if op != "f":
                        continue
                    va, vb = ws.value(args[0]), ws.value(args[1])
                    if va is None or vb is None or static_cost(prog, va, vb) > FOLD_LIMIT:
                        continue
                    counter = Counter()
                    value = run(prog, va, vb, counter)
                    ws.ops += counter.used
                    if value is not None and ws.union(cls, ws.const("Q", value), Why("exact:f", frozenset({hyp.key}))):
                        changed = True
                    break
            if not ws.rebuild() and not changed:
                return

    def _notation_rule(self, hyp: Hyp) -> Rule:
        """Place value: a natural n ending in 0 equals the concept applied to (10, n/10), if the hypothesis agrees."""
        prog = hyp.payload

        def guard(ws: Workspace, s: Dict[str, int]) -> bool:
            n = ws.value(s["?n"])
            if not (natural(n) and n >= 20 and n % 10 == 0):
                return False
            try:
                return run(prog, Fraction(10), n / 10, Counter(200)) == n
            except Exhausted:
                return False

        def build(ws: Workspace, s: Dict[str, int]) -> int:
            n = ws.value(s["?n"])
            return ws.add("f", [ws.const("Q", 10), ws.const("Q", n / 10)])

        return Rule("notation", "?n", build, guard, frozenset({"notation", hyp.key}))

    def _execute(self, term, prog: Tuple, counter: Counter) -> Optional[Fraction]:
        op, payload, kids = term
        if op == "const":
            return payload[1]
        values = [self._execute(k, prog, counter) for k in kids]
        if any(v is None for v in values):
            return None
        if op == "add":
            counter.spend()
            return values[0] + values[1]
        return run(prog, values[0], values[1], counter)

    # ----- verify -----

    def verify(self, task: ArithTask, hyp: Hyp, mind: Mind) -> int:
        code, prog, tree = self.code, hyp.payload, self._tree(task)
        before = code.ops
        numbers = literals(tree) + [v for (a, b), c in task.evidence() for v in (a, b, c)]
        if hyp.answer is not None and hyp.answer != UNKNOWN:
            numbers.append(hyp.answer)
        scale = math.lcm(*[Fraction(v).denominator for v in numbers])
        ok, why = True, ""
        for (a, b), c in task.evidence():
            got = phasor_run(prog, code.encode(int(a * scale)), code.encode(int(b * scale)), scale, code)
            if got is None or code.decode(got) != int(c * scale):
                ok, why = False, f"disagrees with evidence {a} {task.symbol} {b} = {c}"
                break
        if ok:
            got = phasor_tree(tree, prog, scale, code)
            if hyp.answer == UNKNOWN:
                ok = got is None
            else:
                ok = got is not None and code.decode(got) == int(hyp.answer * scale)
            if not ok:
                why = "derivation disagrees with the definition (residue check)"
        hyp.verified = ok
        if not ok:
            hyp.note = why
        return code.ops - before

    # ----- questions -----

    def _goal(self, task: ArithTask, hyp: Hyp) -> Any:
        return self._predicted(task, hyp)[0]

    def _predicted(self, task: ArithTask, hyp: Hyp) -> Tuple[Any, int]:
        """(prediction, ops), cached. Not an evaluation of the question (that is a derivation without
        the laws and would erase what they save) but observational equivalence: UNKNOWN if the
        reading is undefined for the question's kinds, else `("like", fingerprint)` on the probes of
        those kinds. Readings with equal fingerprints are expected, not asserted, to agree; the
        answer comes only from derivation and is checked by the verifier."""
        if hyp.derived:
            return hyp.answer, 0
        key = (id(task), hyp.key)
        if key not in self._goal_cache:
            self._goal_cache[key] = self._analogy(task, hyp)
        return self._goal_cache[key]

    def _analogy(self, task: ArithTask, hyp: Hyp) -> Tuple[Any, int]:
        """Abstract interpretation over kinds {natural: True, non-natural: False, undefined, unknown: None}.
        An application's kind is the kind the reading produced on every probe of its argument kinds
        (None if the probes disagree). An undefined root predicts UNKNOWN; otherwise the prediction is
        the reading's behaviour on the probes of the question's application kinds."""
        counter, seen, fingerprint = Counter(), set(), []

        def on(kind: Tuple) -> List[Optional[Fraction]]:
            out = []
            for p in PROBES:
                if all(k is None or natural(v) == k for v, k in zip(p, kind)):
                    try:
                        out.append(run(hyp.payload, p[0], p[1], counter))
                    except Exhausted:
                        out.append(None)
            if kind not in seen:
                seen.add(kind)
                fingerprint.extend(out)
            return out

        def kind(v: Any) -> Any:
            return natural(v) if isinstance(v, Fraction) else v

        def kind_of(tree: Tuple) -> Any:
            """A Fraction where the value is known without the concept (literals and their sums)."""
            if tree[0] == "num":
                return tree[1]
            ka, kb = kind_of(tree[1]), kind_of(tree[2])
            if UNKNOWN in (ka, kb):
                return UNKNOWN
            if tree[0] == "add":
                if isinstance(ka, Fraction) and isinstance(kb, Fraction):
                    counter.spend()
                    return ka + kb
                ka, kb = kind(ka), kind(kb)
                return True if ka is True and kb is True else (False if {ka, kb} == {True, False} else None)
            kinds = {UNKNOWN if v is None else natural(v) for v in on((kind(ka), kind(kb)))}
            return kinds.pop() if len(kinds) == 1 else None

        root = kind_of(self._tree(task))
        return (UNKNOWN if root == UNKNOWN else ("like", tuple(fingerprint))), counter.used

    def predict(self, task: ArithTask, hyp: Hyp) -> Tuple[Any, int]:
        """For the controller's derive-group action; ops are charged only the first time."""
        first = (id(task), hyp.key) not in self._goal_cache
        value, ops = self._predicted(task, hyp)
        return value, ops if first else 0

    def questions(self, task: ArithTask, hyps: List[Hyp], mind: Mind):
        if task.oracle is None:
            return []
        goals = [self._goal(task, h) for h in hyps]
        asked = {q for q, _ in task.observed} | {(a, b) for a, b, _ in task.examples}
        out = []
        for q in QUESTION_POOL:
            if q in asked:
                continue
            at_q = []
            for h in hyps:
                try:
                    at_q.append(run(h.payload, q[0], q[1], Counter(2000)))
                except Exhausted:
                    at_q.append(None)
            if len(set(map(str, at_q))) > 1:
                out.append((q, at_q, goals, len(hyps)))
        return out

    def ask(self, task: ArithTask, question: Tuple[Fraction, Fraction], hyps: List[Hyp], mind: Mind) -> int:
        answer = task.oracle(*question)
        task.observed.append((question, answer))
        for h in hyps:
            try:
                got = run(h.payload, question[0], question[1], Counter(2000))
            except Exhausted:
                got = None
            if got != answer:
                h.weight = 0.0
                h.note = f"refuted by {question[0]} {task.symbol} {question[1]} = {answer}"
        return len(hyps)

    # ----- memory -----

    def remember(self, task: ArithTask, result: Result, mind: Mind) -> None:
        """Semantic memory keeps the concept's version space: every hypothesis the evidence has not
        refuted (with its weight), plus the evidence itself -- not just the winner."""
        if result.status != "answer" or not any(h.verified and h.answer == result.answer for h in result.hypotheses):
            return
        memory = self._memory(mind)
        entry = memory["concepts"].setdefault(task.symbol, {"hypotheses": [], "evidence": []})
        evidence = entry["evidence"] + [e for e in task.evidence() if e not in entry["evidence"]]
        alive = [(h.payload, h.weight) for h in result.hypotheses if h.weight > 0 and h.verified is not False]
        entry["hypotheses"] = [(p, w) for p, w in alive if all(_safe_run(p, a, b) == c for (a, b), c in evidence)]
        entry["evidence"] = evidence
        memory["uses"][task.symbol] = memory["uses"].get(task.symbol, 0) + 1

    def best(self, mind: Mind, symbol: str) -> Tuple:
        return max(self._memory(mind)["concepts"][symbol]["hypotheses"], key=lambda pw: pw[1])[0]

    def ambiguity(self, mind: Mind, symbol: str) -> float:
        """Entropy (bits) of the concept's surviving hypotheses' behaviour over the question pool."""
        hyps = self._memory(mind)["concepts"][symbol]["hypotheses"]
        classes: Dict[Tuple, float] = {}
        for p, w in hyps:
            fp = tuple(_safe_run(p, a, b) for a, b in QUESTION_POOL)
            classes[fp] = classes.get(fp, 0.0) + w
        total = sum(classes.values())
        return -sum(w / total * math.log2(w / total) for w in classes.values() if w > 0)

    def study(self, mind: Mind, symbol: str, teacher: Callable[[Fraction, Fraction], Fraction], budget: int,
              strategy: str = "curious", seed: int = 0) -> int:
        """Self-study with a teacher. strategy='curious' asks the question whose answer is most uncertain
        under the surviving hypotheses (expected information gain); 'random' asks any pool question.
        Returns the number of questions asked."""
        entry = self._memory(mind)["concepts"][symbol]
        rng = random.Random(seed)
        asked = 0
        while asked < budget and self.ambiguity(mind, symbol) > 0:
            seen = {q for q, _ in entry["evidence"]}
            pool = [q for q in QUESTION_POOL if q not in seen]
            if not pool:
                break
            if strategy == "curious":
                def gain(q) -> float:
                    split: Dict[Any, float] = {}
                    for p, w in entry["hypotheses"]:
                        split[_safe_run(p, *q)] = split.get(_safe_run(p, *q), 0.0) + w
                    total = sum(split.values())
                    return -sum(w / total * math.log2(w / total) for w in split.values() if w > 0)
                q = max(pool, key=gain)
            else:
                q = rng.choice(pool)
            answer = teacher(*q)
            entry["evidence"].append((q, answer))
            entry["hypotheses"] = [(p, w) for p, w in entry["hypotheses"] if _safe_run(p, *q) == answer]
            asked += 1
        return asked

    def sleep(self, mind: Mind, teacher: Optional[Callable] = None, budget: int = 10) -> Dict[str, Any]:
        """Self-loop: (1) curiosity -- study the concepts whose version space is most ambiguous,
        weighted by how often they are used; (2) consolidate laws; (3) rethink -- re-test every
        law on freshly imagined values outside the original probe range, retracting any that fail."""
        memory = self._memory(mind)
        report: Dict[str, Any] = {"asked": {}, "laws": {}, "retracted": []}
        order = sorted(memory["concepts"], key=lambda s: -self.ambiguity(mind, s) * (1 + memory["uses"].get(s, 0)))
        for symbol in order:
            if teacher is not None and budget > 0 and self.ambiguity(mind, symbol) > 0:
                asked = self.study(mind, symbol, teacher, budget)
                budget -= asked
                report["asked"][symbol] = asked
            key = (symbol, f"concept:{show(self.best(mind, symbol))}")
            if key not in memory["laws"]:
                memory["laws"][key], _ = self.consolidate(mind, symbol, max_size=7, probes=16)
            kept = [law for law in memory["laws"][key] if self._holds(law, self.best(mind, symbol))]
            report["retracted"] += [str(law) for law in memory["laws"][key] if law not in kept]
            memory["laws"][key] = kept
            report["laws"][symbol] = len(kept)
        return report

    @staticmethod
    def _holds(law: Law, prog: Tuple, trials: int = 40, seed: int = 99) -> bool:
        """Test a law on imagined values beyond the discovery probes (x, y up to 30; z rational)."""
        rng = random.Random(seed)

        def ev(term, env):
            if isinstance(term, str):
                return env[term]
            op, left, right = term
            lv, rv = ev(left, env), ev(right, env)
            if lv is None or rv is None:
                return None
            return lv + rv if op == "add" else _safe_run(prog, lv, rv)

        for _ in range(trials):
            env = {"?x": Fraction(rng.randint(0, 30)), "?y": Fraction(rng.randint(0, 30)),
                   "?z": Fraction(rng.randint(-60, 60), rng.randint(1, 7))}
            if ev(law.lhs, env) != ev(law.rhs, env):
                return False
        return True

    def consolidate(self, mind: Mind, symbol: str, max_size: int = 7, probes: int = 24, seed: int = 0) -> Tuple[List[Law], int]:
        """Sleep: discover laws of a known concept by testing (QuickSpec) over x, y natural and z rational."""
        prog = self.best(mind, symbol)
        rng = random.Random(seed)
        envs = [{"?x": Fraction(rng.randint(0, 5)), "?y": Fraction(rng.randint(0, 5)),
                 "?z": Fraction(rng.randint(-6, 6), rng.randint(1, 3))} for _ in range(probes)]
        counter = Counter(10 ** 9)

        def ev(term, env) -> Any:
            if isinstance(term, str):
                return env[term]
            op, left, right = term
            lv, rv = ev(left, env), ev(right, env)
            if lv is None or rv is None:
                return None
            if op == "add":
                counter.spend()
                return lv + rv
            return run(prog, lv, rv, counter)

        leaves = ["?x", "?y", "?z"]
        table: Dict[int, List] = {1: list(leaves)}
        for s in range(3, max_size + 1, 2):
            table[s] = [(op, l, r) for op in ("add", "f") for s1 in range(1, s - 1, 2)
                        for l in table[s1] for r in table[s - 1 - s1]]
        universe = [t for s in sorted(table) for t in table[s]]
        laws = discover_laws(universe, ev, envs, law_rules)
        key = f"concept:{show(prog)}"
        self._memory(mind)["laws"][(symbol, key)] = laws
        return laws, counter.used


def _safe_run(prog: Tuple, a: Fraction, b: Fraction) -> Optional[Fraction]:
    try:
        return run(prog, a, b, Counter(2000))
    except Exhausted:
        return None


def _term(term) -> str:
    op, payload, kids = term
    if op == "const":
        return str(payload[1])
    if op == "add":
        return f"({_term(kids[0])} + {_term(kids[1])})"
    return f"f({_term(kids[0])}, {_term(kids[1])})"
