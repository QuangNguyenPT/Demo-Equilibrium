"""Typed concept induction for binary operators (doc 03, sections 1-2).

Part of the operator inference over structured situations core: the L-level half of the
hypothesis loop, where a structured situation (x, y) is read through a hypothesis language and
the answer is decided or the operator is asked about. A hypothesis is a *family* of maps
m(x, y) = sum_j p_j psi_j(x, y) that is the full solution set of a law set (Cauchy-type
functional equations), with rational parameters p fitted EXACTLY from the evidence. Three
things are computed per family:
  - consistency: the evidence lies in the family (exact solve over Q);
  - evidence weight: Bayes factor |Y|^(n - rank) (Prop. 4: fitted parameters buy no evidence);
  - determinacy: a query is answered by the family only if psi(query) is in the row space of the
    evidence matrix (Theorem 3); otherwise the family has no opinion at that query.
A reserved hypothesis BOTTOM ("outside my language") keeps mass 2^-bits_bottom (Def. 3).
Nothing here is domain-specific: the same families apply to any pair of rational quantities.

Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from hrm_x.qlinalg import in_row_space, rank, solve

Q = Fraction
Feature = Callable[[Q, Q], Optional[Q]]
Pair = Tuple[Q, Q]
Evidence = List[Tuple[Pair, Q]]


def _ipow(b: Q, e: Q) -> Optional[Q]:
    """b**e exactly when e is an integer (and b != 0 for e < 0); None otherwise (undefined)."""
    if e.denominator != 1 or abs(e) > 64:
        return None
    if b == 0 and e < 0:
        return None
    return b ** int(e)


@dataclass(frozen=True)
class Family:
    name: str
    features: Tuple[Feature, ...]
    bits: float                      # description length of the discrete choice (Occam prior)
    laws: str                        # the functional equations whose solution set this is

    def row(self, x: Q, y: Q) -> Optional[List[Q]]:
        out = []
        for f in self.features:
            v = f(x, y)
            if v is None:
                return None
            out.append(Q(v))
        return out


def build_families() -> List[Family]:
    """The domain-independent hypothesis language (doc 03, Definition 2)."""
    fam: List[Family] = []
    basis = {"xy": lambda x, y: x * y, "x": lambda x, y: x, "y": lambda x, y: y, "1": lambda x, y: Q(1)}
    supports = [s for r in range(1, 5) for s in itertools.combinations(["xy", "x", "y", "1"], r)]
    for s in supports:                                                     # 15 bi-affine supports
        fam.append(Family("biaffine{" + ",".join(s) + "}", tuple(basis[k] for k in s),
                          bits=math.log2(len(supports)) + len(s), laws="affine in each argument (Jensen x2)"))
    mono = [(a, b) for a in range(-3, 4) for b in range(-3, 4) if (a, b) not in {(0, 0), (1, 0), (0, 1), (1, 1)}]
    for a, b in mono:                                                      # 45 monomials c x^a y^b
        fam.append(Family(f"monomial x^{a} y^{b}",
                          (lambda x, y, a=a, b=b: (None if (p := _ipow(x, Q(a))) is None or (r := _ipow(y, Q(b))) is None else p * r),),
                          bits=math.log2(len(mono)) + 1, laws="multiplicative in each argument (Cauchy P x2)"))
    for a in (1, 2):                                                       # c x^(a y)
        fam.append(Family(f"exp-power x^({a}y)", (lambda x, y, a=a: _ipow(x, a * y),),
                          bits=1 + 1, laws="exponential in y, power in x"))
    for base in (2, 3, 10):                                                # c b^y and c b^x
        fam.append(Family(f"exp {base}^y", (lambda x, y, b=base: _ipow(Q(b), y),), bits=math.log2(6) + 1,
                          laws="exponential in y, constant in x"))
        fam.append(Family(f"exp {base}^x", (lambda x, y, b=base: _ipow(Q(b), x),), bits=math.log2(6) + 1,
                          laws="exponential in x, constant in y"))
    return fam


FAMILIES = build_families()
PROBES: List[Pair] = [(Q(a), Q(b)) for a, b in [(1, 2), (2, 3), (3, 5), (5, 2), (3, 1)]]


@dataclass
class Fitted:
    family: Family
    params: List[Q]
    A: List[List[Q]]                 # evidence feature matrix
    rank: int
    surplus: int
    log2w: float
    symmetric: bool

    seen_x: frozenset = frozenset()
    seen_y: frozenset = frozenset()
    enforce_coverage: bool = True
    min_surplus: int = 0

    def tested(self) -> bool:
        """The reading has predicted at least `min_surplus` evidence rows it was not fitted to."""
        return self.surplus >= self.min_surplus

    def covered(self, x: Q, y: Q) -> bool:
        """Open-world coverage: moving an argument to an unseen value is supported only if that
        argument varied in the evidence, or the reading is symmetric and the other one varied.
        (Any g(x) h(y) with h fixed on the seen values fits evidence in which y never varied.)"""
        if not self.enforce_coverage:
            return True
        vx, vy = len(self.seen_x) > 1, len(self.seen_y) > 1
        ok_x = x in self.seen_x or vx or (self.symmetric and vy)
        ok_y = y in self.seen_y or vy or (self.symmetric and vx)
        return ok_x and ok_y

    def value(self, x: Q, y: Q, coverage: bool = True) -> Optional[Q]:
        """Value at (x, y) if the evidence determines it under this family (Theorem 3) and the query is
        covered by the evidence's variation (open-world rule), else None. coverage=False is the
        'imagined' prediction used only to value questions (what would this reading predict there?)."""
        if coverage and not (self.covered(x, y) and self.tested()):
            return None
        r = self.family.row(x, y)
        if r is None or not in_row_space(self.A, r):
            return None
        return sum((a * b for a, b in zip(r, self.params)), Q(0))


@dataclass
class ConceptConfig:
    log2_answer_space: float = 10.0  # |Y| = 2^10 effective answer space (Prop. 4)
    bits_bottom: float = 4.0         # prior mass of "outside my language" (open world); inf = closed world
    symmetry_bits: float = 3.0       # prior preference for hypotheses symmetric in x, y that use both
    wrong_penalty: float = 4.0       # L: answer only if (1+L) q - L > 0, i.e. q > L/(1+L)
    coverage: bool = True            # open-world coverage rule (ablation: False = closed world)
    min_surplus: int = 0             # answer only from readings with >= this many predicted rows (E14)


@dataclass
class Belief:
    fitted: List[Fitted]
    log2w_bottom: float
    ops: int = 0

    def posterior(self) -> Tuple[List[float], float]:
        ws = [f.log2w for f in self.fitted] + [self.log2w_bottom]
        m = max(ws)
        z = [2.0 ** (w - m) for w in ws]
        s = sum(z)
        return [v / s for v in z[:-1]], z[-1] / s

    def answer_distribution(self, x: Q, y: Q, coverage: bool = True) -> Tuple[Dict[Q, float], float, float]:
        """q(v) over determined answers; also mass that is undetermined and the BOTTOM mass."""
        post, p_bot = self.posterior()
        q: Dict[Q, float] = {}
        undetermined = 0.0
        for f, p in zip(self.fitted, post):
            v = f.value(x, y, coverage)
            self.ops += len(f.A) + 1
            if v is None:
                undetermined += p
            else:
                q[v] = q.get(v, 0.0) + p
        return q, undetermined, p_bot


def rival_margin(bel: Belief, x: Q, y: Q, answer: Q) -> float:
    """log2(mass behind `answer` / mass of the best reading predicting a different value at (x, y)).

    ADR-011 proposal A. Readings are compared on what they would say at the query
    (coverage off), so a rival refused only by the coverage rule still counts: it fits the
    evidence exactly and disagrees. inf when no reading disagrees; -inf when none agrees.
    Computed from the log2 weights directly, so a rival 2000 bits behind does not underflow.
    """
    by_value: Dict[Q, List[float]] = {}
    for f in bel.fitted:
        v = f.value(x, y, coverage=False)
        if v is not None:
            by_value.setdefault(v, []).append(f.log2w)

    def log2_sum(ws: List[float]) -> float:
        m = max(ws)
        return m + math.log2(sum(2.0 ** (w - m) for w in ws))

    support = by_value.pop(answer, None)
    if not by_value:
        return math.inf
    rival = max(log2_sum(ws) for ws in by_value.values())
    return log2_sum(support) - rival if support else -math.inf


def _symmetric(f: Family, params: List[Q], A: List[List[Q]]) -> bool:
    """The fitted instance is symmetric in (x, y) and depends on both, judged on shared probes."""
    vals = []
    for a, b in PROBES:
        ra, rb = f.row(a, b), f.row(b, a)
        if ra is None or rb is None or not in_row_space(A, ra) or not in_row_space(A, rb):
            return False
        va = sum((u * v for u, v in zip(ra, params)), Q(0))
        vb = sum((u * v for u, v in zip(rb, params)), Q(0))
        if va != vb:
            return False
        vals.append(va)
    return len(set(vals)) > 1       # constants are symmetric but ignore both arguments: no bonus


def param_bits(params: Sequence[Q]) -> float:
    """Description length of exact rational parameters: sign + log2(1+|num|) + log2(den) bits each."""
    return sum(1.0 + math.log2(1 + abs(p.numerator)) + math.log2(p.denominator) for p in params)


def fit(evidence: Evidence, cfg: ConceptConfig, families: Sequence[Family] = FAMILIES) -> Belief:
    fitted: List[Fitted] = []
    ops = 0
    n = len(evidence)
    for fam in families:
        rows = [fam.row(x, y) for (x, y), _ in evidence]
        ops += n * len(fam.features)
        if any(r is None for r in rows):
            continue                                   # undefined on evidence: not a reading
        A = [list(r) for r in rows]                    # type: ignore[arg-type]
        b = [z for _, z in evidence]
        p = solve(A, b)
        ops += n * len(fam.features) ** 2
        if p is None:
            continue
        r = rank(A)
        sym = _symmetric(fam, p, A)
        # MDL: L(h) = L(family choice) + L(fitted rational parameters); L(D | h) is paid only for
        # the r points used to fit, the n - r surplus points are predicted (Prop. 4).
        log2w = -fam.bits - param_bits(p) + (n - r) * cfg.log2_answer_space + (cfg.symmetry_bits if sym else 0.0)
        fitted.append(Fitted(fam, p, A, r, n - r, log2w, sym, frozenset(x for (x, _), _ in evidence), frozenset(y for (_, y), _ in evidence), cfg.coverage, cfg.min_surplus))
    return Belief(fitted, -cfg.bits_bottom, ops)


@dataclass
class Decision:
    status: str                      # "answer" | "abstain"
    answer: Optional[Q]
    confidence: float
    reason: str
    questions: List[Tuple[Pair, Q]] = field(default_factory=list)
    ops: int = 0                     # object-level: fits, evaluations, asking cost
    trace: List[str] = field(default_factory=list)
    top: List[Tuple[str, float]] = field(default_factory=list)
    meta_ops: int = 0                # metareasoning: VOI simulations
    winner: Optional["Fitted"] = None  # highest-posterior reading that produced the answer


QUESTION_POOL: List[Pair] = [(Q(a), Q(b)) for a in range(1, 6) for b in range(1, 6)] + \
                           [(Q(1, 2), Q(3)), (Q(3), Q(1, 2)), (Q(5, 2), Q(2))]


def _entropy(dist: Dict[object, float]) -> float:
    s = sum(dist.values())
    return -sum(p / s * math.log2(p / s) for p in dist.values() if p > 0) if s > 0 else 0.0


def _winner(bel: "Belief", query: Pair, ans: Q) -> Optional[Fitted]:
    post, _ = bel.posterior()
    cands = [(p, f) for f, p in zip(bel.fitted, post) if f.value(*query) == ans]
    return max(cands, key=lambda t: t[0])[1] if cands else None


def utility(q: Dict[Q, float], L: float) -> float:
    """U = max(0, max_y (1+L) q(y) - L): +1 right, -L wrong, 0 abstain (doc 03, section 8)."""
    return max(0.0, (1.0 + L) * max(q.values(), default=0.0) - L)


def value_of_question(ev: Evidence, query: Pair, pq: Pair, bel: Belief, cfg: ConceptConfig,
                      families: Sequence[Family] = FAMILIES) -> Tuple[float, int]:
    """E[U(query) after hearing the teacher at pq] - U(query) now, under the current belief.
    Each fitted reading predicts the teacher's answer at pq (or has no opinion); the outcome branches
    are weighted by posterior mass; readings with no opinion / BOTTOM are spread as an 'unknown' branch
    whose posterior is left unchanged (a conservative approximation)."""
    post, p_bot = bel.posterior()
    q_now, _, _ = bel.answer_distribution(*query)
    ops = bel.ops
    branches: Dict[Q, float] = {}
    z = sum(post) or 1.0                  # branch weights conditioned on "the concept is in my language":
    for f, p in zip(bel.fitted, post):    # optimistic about BOTTOM, so questions are valued as exploration
        v = f.value(*pq, coverage=False)
        if v is not None:
            branches[v] = branches.get(v, 0.0) + p / z
    rest = max(0.0, 1.0 - sum(branches.values()))
    exp_u = rest * utility(q_now, cfg.wrong_penalty)
    for v, p in branches.items():
        b2 = fit(ev + [(pq, v)], cfg, families)
        q2, _, _ = b2.answer_distribution(*query)
        ops += b2.ops
        exp_u += p * utility(q2, cfg.wrong_penalty)
    return exp_u - utility(q_now, cfg.wrong_penalty), ops


def decide(evidence: Evidence, query: Pair, cfg: ConceptConfig,
           teacher: Optional[Callable[[Q, Q], Q]] = None, max_questions: int = 4,
           lam: float = 2e-5, ask_cost: float = 2000.0, policy: str = "voc",
           families: Sequence[Family] = FAMILIES) -> Decision:
    """H-level loop for one query: fit (L-level, exact) -> answer | ask | abstain.
    policy 'voc': ask the question with the largest VOI while VOI > lam * ask_cost (Prop. 7);
    'fixed': always ask max_questions questions (fixed compute); 'never': never ask."""
    ev = list(evidence)
    asked: List[Tuple[Pair, Q]] = []
    ops = 0
    ops_meta = 0
    trace: List[str] = []
    threshold = cfg.wrong_penalty / (1.0 + cfg.wrong_penalty)
    for _ in range(max_questions + 1):
        bel = fit(ev, cfg, families)
        q, undet, p_bot = bel.answer_distribution(*query)
        ops += bel.ops
        post, _ = bel.posterior()
        top = sorted(((f.family.name, p) for f, p in zip(bel.fitted, post)), key=lambda t: -t[1])[:3]
        best = max(q.items(), key=lambda kv: kv[1]) if q else (None, 0.0)
        trace.append(f"n={len(ev)} best={best[0]} q={best[1]:.3f} undetermined={undet:.3f} bottom={p_bot:.3f}")
        must_ask = policy == "fixed" and teacher is not None and len(asked) < max_questions
        if best[1] > threshold and not must_ask:
            return Decision("answer", best[0], best[1], "determined by the dominant reading", asked, ops, trace, top, ops_meta, _winner(bel, query, best[0]))
        if teacher is None or policy == "never" or len(asked) >= max_questions:
            break
        seen = {pq for pq, _ in ev}
        best_q, best_v = None, -1.0
        scored = []
        for pq in QUESTION_POOL:
            if pq in seen:
                continue
            dq, u, pb = bel.answer_distribution(*pq, coverage=False)   # imagined answers: readings must disagree
            dist = dict(dq)
            if u + pb > 0:
                dist["?"] = u + pb
            h = _entropy(dist)
            if h > 1e-9:
                scored.append((h, pq))
        meta_ops = bel.ops
        for _, pq in sorted(scored, key=lambda t: -t[0])[:5]:  # VOI only for the 5 most uncertain questions
            voi, o = value_of_question(ev, query, pq, bel, cfg, families)
            meta_ops += o
            if voi > best_v:
                best_q, best_v = pq, voi
        ops_meta += meta_ops
        if best_q is None or (policy == "voc" and best_v <= lam * ask_cost):
            trace.append(f"stop asking: best VOI {best_v:.4f} <= lam*cost {lam * ask_cost:.4f}")
            break
        a = teacher(*best_q)
        asked.append((best_q, a))
        ev.append((best_q, a))
        ops += int(ask_cost)
        trace.append(f"ask {best_q[0]},{best_q[1]} -> {a} (VOI {best_v:.3f})")
    bel = fit(ev, cfg, families)
    q, undet, p_bot = bel.answer_distribution(*query)
    ops += bel.ops
    best = max(q.items(), key=lambda kv: kv[1]) if q else (None, 0.0)
    post, _ = bel.posterior()
    top = sorted(((f.family.name, p) for f, p in zip(bel.fitted, post)), key=lambda t: -t[1])[:3]
    if best[1] > threshold:
        return Decision("answer", best[0], best[1], "determined by the dominant reading", asked, ops, trace, top, ops_meta, _winner(bel, query, best[0]))
    if not q:
        reason = ("no reading in my language fits the evidence" if not bel.fitted else
                  "the evidence does not determine this input (outside the span of what I have seen)")
    elif p_bot > 0.5:
        reason = "the evidence is better explained by 'outside my language' than by any reading"
    else:
        reason = "readings disagree: " + ", ".join(f"{v}:{p:.2f}" for v, p in sorted(q.items(), key=lambda t: -t[1])[:3])
    return Decision("abstain", None, best[1], reason, asked, ops, trace, top, ops_meta)
