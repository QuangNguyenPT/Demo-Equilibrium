"""Concept construction: new basis features built from primitives when the language runs out.

HRM-X v0.5 fits a hypothesis *family* exactly, but every family was written down in advance
(`concepts.FAMILIES`). When no family explains the evidence, BOTTOM wins and the mind abstains --
correctly, since ADR-012, but it can never learn the missing concept. This module is the step
from "select among written families" to "build a new one":

    primitives  P = {x, y, 1, 2, +, -, *, /, ifgt(a, b, c, d)} plus every stored library concept
    feature     g = a postfix expression over P, found bottom-up and kept only if it is new
    family      m(x, y) = a g(x, y) + sum_{s in S} b_s s(x, y),   S subset of {x, y, 1}
    fit         the same exact solve over Q as every other family (concepts.fit), so a feature
                buys its linear coefficients for free and only its *skeleton* is searched

Three reductions keep the search small, and each is checked by a test:

  1. observational equivalence -- two expressions with equal values on the probe points are one
     feature (the smallest is kept);
  2. canonical class modulo the base span span{1, x, y, xy} -- g and 3g - x + 2 fit exactly the
     same data under the linear closure, so only one of them is a new feature, and a feature
     inside the span (x - y, xy + 1) is not new at all;
  3. a float prefilter -- a feature whose full family {g, x, y, 1} cannot fit the evidence in
     least squares cannot fit it exactly either, so only survivors pay for an exact solve.

The primitive set is still given. What is no longer given is the family: `max`, `min`, `|x-y|`,
`x^2 + y` and `relu(x - y)` are not in the list above; `max` is `ifgt(x, y, x, y)` (5 tokens) and
the other three are affine images of features found the same way. That is the claim this module
makes, and `experiments/x_construct.py` measures it.

Stored concepts enter the primitive set as binary tokens (DreamCoder-style compression): once
`max` is learned, `max(x, y) * y` is 5 tokens instead of 9, so concepts built on concepts come
within a budget that could not reach them from scratch.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hrm_x.concepts import ConceptConfig, Family, Fitted, fit

Q = Fraction

# ---------- the primitive alphabet ----------
X, Y, ONE, TWO = 0, 1, 2, 3
ADD, SUB, MUL, DIV, IFGT = 10, 11, 12, 13, 14
LIB = 32                       # token LIB + k calls stored concept k as a binary operator
LEAVES = (X, Y, ONE, TWO)
BINARY = (ADD, SUB, MUL, DIV)
NAMES = {X: "x", Y: "y", ONE: "1", TWO: "2", ADD: "+", SUB: "-", MUL: "*", DIV: "/", IFGT: "ifgt"}
MAX_VALUE = 1e12               # a probe value beyond this is treated as undefined (overflow guard)

# Probe points: dyadic rationals, so +, -, * are exact in float64; no zero coordinate, so x / y and
# y / x are defined everywhere; negative values so that sign-sensitive features are told apart.
PROBES: Tuple[Tuple[Q, Q], ...] = tuple((Q(a), Q(b)) for a, b in [
    (1, 2), (2, 3), (3, 1), (5, 2), (2, 5), (4, 4), (-1, 3), (3, -2), (Q(1, 2), 3), (6, 1),
    (7, 7), (3, Q(1, 2)), (-2, -5), (-3, 4)])

Expr = Tuple[int, ...]


def arity(tok: int) -> int:
    if tok in LEAVES:
        return 0
    if tok == IFGT:
        return 4
    return 2                   # BINARY and LIB tokens


def evaluate(expr: Expr, x: Q, y: Q, library: Sequence["Callable[[Q, Q], Optional[Q]]"] = ()) -> Optional[Q]:
    """Exact value of a postfix expression at (x, y); None where it is undefined (division by 0)."""
    stack: List[Q] = []
    for tok in expr:
        if tok == X:
            stack.append(x)
        elif tok == Y:
            stack.append(y)
        elif tok == ONE:
            stack.append(Q(1))
        elif tok == TWO:
            stack.append(Q(2))
        elif tok == IFGT:
            d, c, b, a = stack.pop(), stack.pop(), stack.pop(), stack.pop()
            stack.append(c if a > b else d)
        else:
            b, a = stack.pop(), stack.pop()
            if tok == ADD:
                stack.append(a + b)
            elif tok == SUB:
                stack.append(a - b)
            elif tok == MUL:
                stack.append(a * b)
            elif tok == DIV:
                if b == 0:
                    return None
                stack.append(a / b)
            else:
                v = library[tok - LIB](a, b)
                if v is None:
                    return None
                stack.append(v)
    assert len(stack) == 1, expr
    return stack[0]


def render(expr: Expr, lib_names: Sequence[str] = ()) -> str:
    """Infix text of a postfix expression (the form the writer and the tests read)."""
    stack: List[str] = []
    for tok in expr:
        if tok in LEAVES:
            stack.append(NAMES[tok])
        elif tok == IFGT:
            d, c, b, a = stack.pop(), stack.pop(), stack.pop(), stack.pop()
            stack.append(f"ifgt({a}, {b}, {c}, {d})")
        elif tok >= LIB:
            b, a = stack.pop(), stack.pop()
            name = lib_names[tok - LIB] if tok - LIB < len(lib_names) else f"lib{tok - LIB}"
            stack.append(f"{name}({a}, {b})")
        else:
            b, a = stack.pop(), stack.pop()
            stack.append(f"({a} {NAMES[tok]} {b})")
    return stack[0]


def expr_bits(expr: Expr, n_library: int = 0) -> float:
    """Description length: each token is one choice among the current alphabet."""
    return len(expr) * math.log2(len(LEAVES) + len(BINARY) + 1 + n_library)


# ---------- bottom-up enumeration with observational equivalence ----------

@dataclass(frozen=True)
class Feature:
    """A constructed basis function: its smallest expression and its values on the probes."""

    expr: Expr
    values: Tuple[float, ...]
    bits: float


def _float_values(values: np.ndarray) -> Optional[np.ndarray]:
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > MAX_VALUE):
        return None
    return values


def _key(values: np.ndarray) -> Tuple[float, ...]:
    """Observational-equivalence key: values at 1e-9 resolution (rint(v * 1e9), the native key too)."""
    return tuple((np.rint(np.asarray(values) * 1e9) + 0.0).tolist())


def enumerate_bank(max_size: int, library: Sequence[Tuple[Expr, Callable]] = ()) -> Dict[int, List[Tuple[Expr, np.ndarray]]]:
    """All observationally distinct expressions by size (smallest representative kept).

    `library` holds (expr, fn) of stored concepts; each becomes a binary token whose probe values
    are computed by its exact function, so a library call costs one token however large its body.
    """
    xs = np.array([float(p[0]) for p in PROBES])
    ys = np.array([float(p[1]) for p in PROBES])
    bank: Dict[int, List[Tuple[Expr, np.ndarray]]] = {s: [] for s in range(1, max_size + 1)}
    if not any(fn is not None for _, fn in library):
        from hrm_x import native                      # C++ twin, bit-identical (tests/test_x_native.py)
        if native.enabled():
            for expr, v in native.enumerate_bank(max_size, xs, ys):
                bank[len(expr)].append((expr, v))
            return bank
    seen: Dict[Tuple[float, ...], Expr] = {}

    def add(expr: Expr, v: Optional[np.ndarray]) -> None:
        if v is None:
            return
        k = _key(v)
        if k not in seen:
            seen[k] = expr
            bank[len(expr)].append((expr, v))

    for tok, v in ((X, xs), (Y, ys), (ONE, np.ones_like(xs)), (TWO, 2 * np.ones_like(xs))):
        add((tok,), v)
    lib_tables = [(k, _lib_table(fn)) for k, (_, fn) in enumerate(library) if fn is not None]
    for size in range(3, max_size + 1):
        for s1 in range(1, size - 1):
            s2 = size - 1 - s1
            for e1, v1 in bank[s1]:
                for e2, v2 in bank[s2]:
                    with np.errstate(all="ignore"):
                        add(e1 + e2 + (ADD,), _float_values(v1 + v2))
                        add(e1 + e2 + (SUB,), _float_values(v1 - v2))
                        add(e1 + e2 + (MUL,), _float_values(v1 * v2))
                        add(e1 + e2 + (DIV,), _float_values(v1 / v2) if np.all(v2 != 0) else None)
                    for k, table in lib_tables:
                        add(e1 + e2 + (LIB + k,), table(v1, v2))
        if size >= 5:
            for s1 in range(1, size - 3):
                for s2 in range(1, size - 2 - s1):
                    for s3 in range(1, size - 1 - s1 - s2):
                        s4 = size - 1 - s1 - s2 - s3
                        if s4 < 1:
                            continue
                        for ea, va in bank[s1]:
                            for eb, vb in bank[s2]:
                                gt = va > vb
                                if gt.all() or (~gt).all():
                                    continue          # a constant branch is just one of c, d
                                for ec, vc in bank[s3]:
                                    for ed, vd in bank[s4]:
                                        add(ea + eb + ec + ed + (IFGT,), np.where(gt, vc, vd))
    return bank


def _lib_table(fn: Callable[[Q, Q], Optional[Q]]) -> Callable[[np.ndarray, np.ndarray], Optional[np.ndarray]]:
    """Float evaluation of a stored concept on value vectors (exact per element, then float)."""
    def table(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
        out = []
        for u, v in zip(a.tolist(), b.tolist()):
            r = fn(Q(u).limit_denominator(1 << 20), Q(v).limit_denominator(1 << 20))
            if r is None:
                return None
            out.append(float(r))
        return _float_values(np.array(out))
    return table


def _base_projector() -> np.ndarray:
    """Orthogonal projector onto span{1, x, y, xy} over the probe points (float)."""
    xs = np.array([float(p[0]) for p in PROBES])
    ys = np.array([float(p[1]) for p in PROBES])
    B = np.stack([np.ones_like(xs), xs, ys, xs * ys], axis=1)
    return B @ np.linalg.pinv(B)


def canonical_features(bank: Dict[int, List[Tuple[Expr, np.ndarray]]], n_library: int = 0) -> List[Feature]:
    """One feature per class modulo span{1, x, y, xy}; members of the span are dropped.

    The residual r = (I - P) v is normalised by its largest-magnitude entry (sign included), so g
    and a g + (anything in the span) share a key. Classes are visited smallest-first, so the kept
    representative is the shortest expression of its class.
    """
    P = _base_projector()
    items = [(expr, v) for size in sorted(bank) for expr, v in bank[size]]
    if not items:
        return []
    V = np.stack([v for _, v in items])
    R = V - V @ P.T                                     # residuals modulo the base span, all at once
    scale = np.max(np.abs(R), axis=1)
    inside = scale < 1e-7 * np.maximum(1.0, np.max(np.abs(V), axis=1))
    pivot = R[np.arange(len(R)), np.argmax(np.abs(R), axis=1)]
    out: List[Feature] = []
    seen = set()
    for i, (expr, v) in enumerate(items):
        if inside[i]:
            continue                                    # inside the base span: not a new feature
        k = _key(R[i] / pivot[i])
        if k in seen:
            continue
        seen.add(k)
        out.append(Feature(expr, tuple(v.tolist()), expr_bits(expr, n_library)))
    return out


@lru_cache(maxsize=8)
def _cached_features(max_size: int) -> Tuple[Feature, ...]:
    return tuple(canonical_features(enumerate_bank(max_size)))


def features(max_size: int, library: Sequence[Tuple[Expr, Callable]] = ()) -> List[Feature]:
    """Canonical constructed features up to `max_size` tokens (cached when the library is empty)."""
    if not any(fn is not None for _, fn in library):
        return list(_cached_features(max_size))
    return canonical_features(enumerate_bank(max_size, library), len(library))


# ---------- fitting a constructed feature to evidence ----------

EXTRAS: Tuple[Tuple[str, ...], ...] = tuple(s for r in range(0, 4) for s in itertools.combinations(("x", "y", "1"), r))
_BASIS = {"x": lambda x, y: x, "y": lambda x, y: y, "1": lambda x, y: Q(1)}


def prefilter(g_values: Sequence[Optional[float]], xs: Sequence[float], ys: Sequence[float],
              zs: Sequence[float], tol: float = 1e-8) -> bool:
    """Could z = a g + b x + c y + d hold exactly? (Necessary condition, float least squares.)"""
    if any(v is None for v in g_values):
        return False
    A = np.array([[g, x, y, 1.0] for g, x, y in zip(g_values, xs, ys)])
    z = np.array(zs, dtype=float)
    sol, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(np.max(np.abs(A @ sol - z))) <= tol * max(1.0, float(np.max(np.abs(z))))


@dataclass
class ConstructConfig:
    """How far construction may go, and and what it may use."""

    sizes: Tuple[int, ...] = (3, 5, 7)     # token budget per propose level 1, 2, 3
    margin_bits: float = 10.0              # prune: families that cannot come within this of BOTTOM
    max_readings: int = 12                 # readings handed to the controller per level
    use_library: bool = True               # stored concepts are primitives (compression)


def _is_monomial(values: Sequence[float]) -> bool:
    """g equals c x^a y^b on the probes (then the bare family {g} duplicates a base monomial)."""
    xs = [float(p[0]) for p in PROBES]
    ys = [float(p[1]) for p in PROBES]
    for a in range(-3, 4):
        for b in range(-3, 4):
            m = [x ** a * y ** b for x, y in zip(xs, ys)]
            ratios = {round(v / w, 9) for v, w in zip(values, m) if w != 0}
            if len(ratios) == 1:
                return True
    return False


def families_of(feature: Feature, fn: Callable[[Q, Q], Optional[Q]], label: str) -> List[Family]:
    """The linear-closure families of one feature: g plus every subset of {x, y, 1}."""
    mono = _is_monomial(feature.values)
    out = []
    for extra in EXTRAS:
        if not extra and mono:
            continue
        feats = (fn,) + tuple(_BASIS[e] for e in extra)
        name = f"construct[{label}]" + ("+" + "+".join(extra) if extra else "")
        out.append(Family(name, feats, bits=feature.bits + math.log2(len(EXTRAS)) + len(extra),
                          laws=f"linear closure of the constructed feature {label}"))
    return out


@dataclass
class ConstructionReport:
    """What one construction pass examined, kept, and paid."""

    fitted: List[Fitted] = field(default_factory=list)
    examined: int = 0
    prefilter_passed: int = 0
    ops: int = 0


def construct(evidence: Sequence[Tuple[Tuple[Q, Q], Q]], cfg: ConceptConfig, feats: Sequence[Feature],
              fn_of: Callable[[Feature], Callable[[Q, Q], Optional[Q]]], label_of: Callable[[Feature], str],
              margin_bits: float = 10.0, log2_symmetry: float = 3.0) -> ConstructionReport:
    """Fit every feature of the level: float prefilter, then an exact solve for the survivors.

    The search is exhaustive within the level, so its result does not depend on the order of
    `feats`. (An anytime stop with a sound bound was tried and removed: the sum of the remaining
    features' weight ceilings never fell below the best reading, so it never fired -- ADR-013.)
    """
    rep = ConstructionReport()
    n = len(evidence)
    xs = [float(p[0]) for (p, _) in evidence]
    ys = [float(p[1]) for (p, _) in evidence]
    zs = [float(z) for _, z in evidence]
    for feat in feats:
        rep.examined += 1
        fn = fn_of(feat)
        g = [fn(p[0], p[1]) for p, _ in evidence]
        rep.ops += n * len(feat.expr)
        if not prefilter([None if v is None else float(v) for v in g], xs, ys, zs):
            continue
        rep.prefilter_passed += 1
        # A family with k parameters has at most n - k surplus rows; if even that cannot bring it
        # within `margin_bits` of BOTTOM it can never carry answer mass, so it is not fitted.
        fams = [fam for fam in families_of(feat, fn, label_of(feat))
                if -fam.bits + max(0, n - len(fam.features)) * cfg.log2_answer_space + log2_symmetry
                > -cfg.bits_bottom - margin_bits]
        if not fams:
            continue
        bel = fit(list(evidence), cfg, fams)
        rep.ops += bel.ops
        rep.fitted.extend(bel.fitted)
    rep.fitted.sort(key=lambda f: -f.log2w)
    return rep
