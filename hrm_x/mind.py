"""Mind: the cognitive loop. It keeps a belief over hypotheses and spends computation only where it pays.

State (spec v0.4 §14): hypotheses h with posterior mass p_h, each possibly derived (answer y_h)
and verified. The probability that answer y is correct is
    q(y) = sum_h p_h r_h [y_h = y],   r_h = 1 if verified else rho
(rho = learned pass rate of verification). Answering y earns +1 if right and -L if wrong,
abstaining earns 0, so the value of the current state is
    U = max(0, max_y (1 + L) q(y) - L).
Each cycle it takes the action with the largest
    VOC(a) = E[U after a] - U - lambda * cost(a)
(myopic value of computation, Russell & Wefald 1991) and answers when no action has VOC > 0.
Actions: propose (induce/parse hypotheses), derive (L-level inference), verify (independent
check), ask (a question with the largest value), abstract (domain laws). Easy tasks end after
derive + verify; ambiguity makes asking valuable; missing evidence ends in "I don't know".
Two timescales: H = these decisions; L = the saturation/execution inside derive().
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Protocol, Tuple

UNKNOWN = "undefined"
# The reserved reading "the truth is outside my language" (spec v0.5 Def. 3). A hypothesis with this
# answer keeps its share of the normalizing mass -- it is a real alternative, unlike UNKNOWN, which
# means "this reading has no value here" and leaves the mass -- but predicts no value, is never
# verified and never becomes the answer. Without it the BOTTOM prior cancelled in normalization.
OUTSIDE = "outside my language"


def _is_value(a: Any) -> bool:
    """An answer the mind could give: not missing, not undefined, not the outside-my-language mass."""
    return a is not None and a != UNKNOWN and a != OUTSIDE


DEFAULT_LAM = 2e-5


@dataclass
class Hyp:
    key: str
    weight: float
    payload: Any
    answer: Any = None
    derived: bool = False
    verified: Optional[bool] = None
    note: str = ""
    culprits: FrozenSet[str] = frozenset()
    spread: float = 0.0              # OUTSIDE hypotheses only: probability of any one specific answer


@dataclass
class Step:
    cycle: int
    action: str
    target: str
    voc: float
    ops: int
    note: str = ""


@dataclass
class Result:
    status: str                      # "answer" | "abstain"
    answer: Any
    confidence: float
    reason: str
    trace: List[Step]
    ops: int
    hypotheses: List[Hyp]

    @property
    def cycles(self) -> int:
        return len(self.trace)


class Domain(Protocol):
    name: str

    def propose(self, task: Any, level: int, mind: "Mind") -> Optional[Tuple[List[Hyp], int]]: ...
    def propose_cost(self, task: Any, level: int) -> float: ...
    def derive(self, task: Any, hyp: Hyp, mind: "Mind") -> int: ...
    def derive_cost(self, task: Any, hyp: Hyp, mind: "Mind") -> float: ...
    def verify(self, task: Any, hyp: Hyp, mind: "Mind") -> int: ...
    def questions(self, task: Any, hyps: List[Hyp], mind: "Mind") -> List[Tuple[Any, List[Any], List[Any], int]]: ...
    def ask(self, task: Any, question: Any, hyps: List[Hyp], mind: "Mind") -> int: ...
    def remember(self, task: Any, result: Result, mind: "Mind") -> None: ...


@dataclass
class Patience:
    """5-HT-like: the price of one inference op, in utility units. Learns instead of being fixed.

    Spec v0.4 §14 left lambda "[U] no learning rule proposed; left fixed on purpose", and
    Modulators had to carry that as a caveat. This is the rule.

    The economic reading: lambda is the exchange rate between utility and ops. A controller
    that acts when VOC(a) = E[U after a] - U - lambda*cost(a) > 0 stops paying exactly when
    the marginal value of an op falls to its price, so the *correct* price is the marginal
    value density, not a constant chosen by hand. The mind observes that density directly:
    over an episode it spent `ops` and ended holding utility `utility`, so

        d = utility / ops

    is a sample of value-per-op. Lambda is driven to that sample by log-space dual ascent,

        lam <- lam * exp(eta * (log d - log lam)),

    whose fixed point is lam = d. Multiplicative because lambda and d are both positive and
    the update must be scale-free; log-space because the interesting range spans decades
    (2e-5 is the default and d can be 1e-1). `shrink` keeps the estimate anchored to `base`,
    so a single unrepresentative episode cannot drag the price somewhere absurd, and
    `bound` keeps it positive and finite.

    Under volatility the old reliability evidence is stale, so the price is scaled up
    (1 + coupling * volatility): after the world changes, being patient is how you waste the
    most compute on a model you no longer trust. That is the NE-high / 5-HT-low direction,
    and it is the one coupling here that is a prediction rather than a tautology -- it is
    tested in tests/test_x_patience.py.

    Off by default, and now measured rather than merely unproven. `experiments/x_patience.py`
    judged this rule against a fixed lambda swept over this class's own reachable range
    [1e-7, 1e-1] on held-out arithmetic and novel grid streams; the learned price reached
    -0.90 and -0.36 accuracy against the best fixed price, so the fixed constant stays.

    The cause is structural, not a tuning miss. The fixed point above is *average* value
    density U/ops, while the VOC rule needs *marginal* gain/cost. Where the two coincide the
    rule is fine -- `test_the_price_is_a_live_control_variable_and_learning_it_beats_both_extremes`
    builds a domain whose density is 1.6e-4, right at break-even. Where an episode is cheap
    they differ by orders of magnitude: arithmetic costs ~6 ops at U~1, so the rule climbs
    toward 0.167, four decades above the band (lambda <= 1e-4) where the mind still acts. It
    then freezes, because `update` cannot take a density sample from a zero-op episode, which
    makes being priced out an absorbing state rather than an oscillation.

    So do not simply turn this on. A replacement needs marginal-value telemetry that does not
    exist yet, and therefore a decision record of its own. Two tests in test_x_patience.py
    characterize the defect on purpose; if you change `update`, expect them to fail.
    """

    lam: float = DEFAULT_LAM
    base: Optional[float] = None          # None = inherit Mind.control.lam, the one source of truth
    eta: float = 0.15
    shrink: float = 0.9
    coupling: float = 1.0
    bound: Tuple[float, float] = (1e-7, 1e-1)
    density: float = 0.0
    seen: float = 0.0

    def prior(self) -> float:
        """The value lambda returns to. `None` base means "whatever the Controller says"."""
        return DEFAULT_LAM if self.base is None else self.base

    def price(self, volatility: float = 0.0) -> float:
        """The lambda the loop should use now: the learned price, raised while the world is unstable."""
        return min(self.bound[1], self.lam * (1.0 + self.coupling * max(0.0, volatility)))

    def update(self, ops: int, utility: float) -> float:
        """Fold one finished episode into the price. Returns the new lambda.

        There is no special case for a worthless episode. A density of 0 enters the EMA like
        any other sample, drags the density down, and the ascent toward the lower target does
        the work at a bounded step. (An earlier version decayed lambda directly by
        `1 - shrink`; with shrink=0.99 that moved the price 99% on one bad episode, which is
        the anchoring this class is supposed to provide.)
        """
        if ops <= 0:
            return self.lam                      # no ops, no density sample
        d = max(utility, 0.0) / ops
        self.density = d if self.seen == 0.0 else self.shrink * self.density + (1.0 - self.shrink) * d
        self.seen += 1.0
        target = max(self.density, self.bound[0])
        self.lam = min(self.bound[1], max(self.bound[0],
                                         self.lam * math.exp(self.eta * (math.log(target) - math.log(self.lam)))))
        return self.lam

    def reset(self) -> None:
        self.lam, self.density, self.seen = self.prior(), 0.0, 0.0


@dataclass
class Controller:
    """Knobs of the metareasoning; `lam` prices one inference op in utility units."""

    lam: float = DEFAULT_LAM
    wrong_penalty: float = 1.0
    verify_ops: float = 50.0
    ask_ops: float = 2000.0
    max_cycles: int = 40
    use_verifier: bool = True
    use_questions: bool = True
    learn_patience: bool = False       # opt in; measured worse than the constant (see Patience)
    ask_horizon: int = 3               # questions a plan may commit to when OUTSIDE holds the mass (v0.6)


@dataclass
class Modulators:
    """Global control signals, each with exactly one computational role (Doya 2002; Yu & Dayan 2005).

    reliability (ACh-like, expected uncertainty): the learned rate rho at which unverified
        derivations turn out right -- the Beta mean in Mind.stats.
    prediction error (DA-like): outcome minus expectation; it is what updates reliability.
    volatility (NE-like, unexpected uncertainty): how far recent surprise, -log P(outcome | rho),
        exceeds the surprise the mind itself expects, the entropy H(rho) (its expected uncertainty).
        High volatility means the world changed, so old reliability evidence is discounted before
        the new outcome is counted.
    patience (5-HT-like) is the price of computation. It is a fixed constant in
        `Controller.lam` unless `Controller.learn_patience` is set, in which case
        `Patience` drives it from the value-per-op the mind actually observes. This
        modulator's job is to supply the volatility that scales that price.
    """

    enabled: bool = True
    trace: Optional[float] = None       # recent observed surprise (EMA)
    volatility: float = 0.0
    rate: float = 0.3
    forget: float = 0.6

    def surprise(self, p_outcome: float, rho: float) -> float:
        """Update with one outcome whose predicted probability was p_outcome; returns volatility in [0, 1]."""
        s = -math.log(max(p_outcome, 1e-6))
        expected = -(rho * math.log(max(rho, 1e-6)) + (1 - rho) * math.log(max(1 - rho, 1e-6)))
        self.trace = expected if self.trace is None else self.trace
        self.trace += self.rate * (s - self.trace)
        excess = (self.trace - expected) / max(expected, 0.05)
        self.volatility = min(1.0, max(0.0, excess)) if self.enabled else 0.0
        return self.volatility


@dataclass
class Mind:
    """Holds semantic memory (per-domain knowledge), episodic memory and the metareasoning statistics."""

    control: Controller = field(default_factory=Controller)
    knowledge: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    episodes: List[Tuple[str, Any, Result]] = field(default_factory=list)
    stats: Dict[str, List[float]] = field(default_factory=dict)
    modulators: Modulators = field(default_factory=Modulators)
    patience: Patience = field(default_factory=Patience)

    def __post_init__(self) -> None:
        # One source of truth for the price's prior. `Controller.lam` is it: the value the mind
        # starts from and returns to. Patience inherits it, and starts at it, so enabling
        # learning moves the price away from the published constant rather than ignoring it.
        # (Two independent knobs -- Controller.lam and Patience.base -- were a silent footgun:
        # setting one while learning was off did nothing at all.)
        if self.patience.base is None:
            self.patience.base = self.control.lam
        if self.patience.lam == DEFAULT_LAM and self.control.lam != DEFAULT_LAM:
            self.patience.lam = self.control.lam

    def price_of_computation(self) -> float:
        """The lambda this episode should use: learned when enabled, the fixed constant otherwise.

        Read once per cycle so that a volatility spike raised mid-episode takes effect on the
        next decision instead of being frozen at the value the episode started with.
        """
        if self.control.learn_patience:
            return self.patience.price(self.modulators.volatility)
        return self.control.lam

    # ---------- learned metareasoning statistics (Beta means) ----------

    def rate(self, key: str, prior: Tuple[float, float] = (4.0, 1.0)) -> float:
        a, b = self.stats.get(key, list(prior))
        return a / (a + b)

    def observe(self, key: str, success: bool, prior: Tuple[float, float] = (4.0, 1.0)) -> None:
        s = self.stats.setdefault(key, list(prior))
        if key.endswith(":verify-pass"):
            p = s[0] / (s[0] + s[1])
            v = self.modulators.surprise(p if success else 1.0 - p, p)
            keep = 1.0 - self.modulators.forget * v          # discount stale evidence toward the prior
            s[0] = prior[0] + (s[0] - prior[0]) * keep
            s[1] = prior[1] + (s[1] - prior[1]) * keep
        s[0 if success else 1] += 1.0

    def feedback(self, domain: str, correct: bool) -> None:
        """External outcome after answering (a teacher, the world): the DA-like prediction-error signal."""
        self.observe(f"{domain}:verify-pass", correct)

    def sleep(self, domain: Domain, teacher: Any = None, budget: int = 10) -> Dict[str, Any]:
        """The self-loop between tasks (study / rethink): the domain spends a question budget where
        curiosity says knowledge is most uncertain, consolidates laws, and re-tests them."""
        fn = getattr(domain, "sleep", None)
        return fn(self, teacher, budget) if fn is not None else {}

    def save(self, path: str) -> None:
        """Persist semantic memory, learned statistics and modulators (a pickle: load only your own files)."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"knowledge": self.knowledge, "stats": self.stats, "modulators": self.modulators,
                         "control": self.control, "patience": self.patience}, f)

    @staticmethod
    def load(path: str) -> "Mind":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        return Mind(control=state["control"], knowledge=state["knowledge"], stats=state["stats"],
                    modulators=state["modulators"],
                    patience=state.get("patience", Patience()))


    # ---------- belief ----------

    def _belief(self, hyps: List[Hyp], rho: float) -> Tuple[Dict[Any, float], float]:
        """q(y) and the normalizing mass. Readings under which the question is undefined are left out of
        the mass: a question that is asked presupposes it has an answer (an explicit pragmatic
        assumption; if no reading answers, the mind abstains)."""
        alive = [h for h in hyps if h.weight > 0 and h.verified is not False
                 and not (h.derived and h.answer == UNKNOWN)]
        mass = sum(h.weight for h in alive)
        q: Dict[Any, float] = {}
        if mass <= 0:
            return q, 0.0
        for h in alive:
            if h.derived and _is_value(h.answer):   # "undefined for me" and "outside my language" are not answers
                r = 1.0 if h.verified else rho
                q[h.answer] = q.get(h.answer, 0.0) + h.weight / mass * r
        return q, mass

    @staticmethod
    def _predict(domain: Domain, task: Any, hyp: Hyp) -> Tuple[Any, int]:
        """(cheap prediction of hyp's answer or None, ops spent the first time it is asked)."""
        predict = getattr(domain, "predict", None)
        return predict(task, hyp) if predict is not None else (None, 0)

    def _utility(self, q: Dict[Any, float]) -> float:
        best = max(q.values(), default=0.0)
        penalty = self.control.wrong_penalty
        return max(0.0, (1.0 + penalty) * best - penalty)

    def _value_of_question(self, alive: List[Hyp], at_q: List[Any], goals: List[Any], rho: float,
                           horizon: int = 1) -> float:
        """E[U after hearing the true answer at q] - U, using each hypothesis's goal prediction.

        A reading predicts one outcome at q. An OUTSIDE hypothesis predicts none in particular: it
        joins every predicted outcome with likelihood `spread` and keeps the rest for "something
        else", so hearing a reading's prediction confirmed re-weights it against OUTSIDE by
        1 / spread -- the same Bayes factor the domain's own posterior applies to a surplus row.
        """
        def value(group: List[Tuple[Hyp, Any, float]]) -> float:
            mass = sum(w for h, g, w in group if g is not None and g != UNKNOWN)
            if mass <= 0:
                return 0.0
            q: Dict[Any, float] = {}
            for h, g, w in group:
                if _is_value(g):
                    q[g] = q.get(g, 0.0) + w / mass * (1.0 if h.verified else rho)
            return self._utility(q)

        total = sum(h.weight for h in alive)
        outcomes: Dict[Any, List[Tuple[Hyp, Any, float]]] = {}
        outside = []
        for h, g, a in zip(alive, goals, at_q):
            if a == OUTSIDE:
                outside.append((h, g))
            else:
                outcomes.setdefault(UNKNOWN if a is None else a, []).append((h, g, h.weight))
        predicted = [k for k in outcomes if k != UNKNOWN]
        for k in predicted:
            outcomes[k] += [(h, g, h.weight * h.spread ** horizon) for h, g in outside]
        if outside:
            outcomes[OUTSIDE] = [(h, g, h.weight * max(0.0, 1.0 - len(predicted) * h.spread ** horizon))
                                 for h, g in outside]
        after = sum(sum(w for _, _, w in group) / total * value(group) for group in outcomes.values())
        return after - value([(h, g, h.weight) for h, g in zip(alive, goals)])

    # ---------- the loop ----------

    def solve(self, domain: Domain, task: Any) -> Result:
        c = self.control
        hyps: List[Hyp] = []
        trace: List[Step] = []
        ops, level, exhausted = 0, 0, False
        rho_key = f"{domain.name}:verify-pass"
        for cycle in range(1, c.max_cycles + 1):
            lam = self.price_of_computation()
            rho = self.rate(rho_key)
            q, mass = self._belief(hyps, rho)
            u_now = self._utility(q)
            options: List[Tuple[float, str, Any, str]] = []

            readings = sum(h.weight for h in hyps if h.weight > 0 and h.answer != OUTSIDE
                           and h.verified is not False and not (h.derived and h.answer == UNKNOWN))
            outside = sum(h.weight for h in hyps if h.weight > 0 and h.answer == OUTSIDE)
            outside_share = outside / mass if mass > 0 else 0.0
            if not exhausted and (readings <= 0 or outside_share > 0.5):
                # A deeper level is worth proposing when nothing explains the evidence, or when the
                # best explanation is "outside my language" (v0.6: the next level may *build* one).
                # Only a domain that reports OUTSIDE can reach the second case.
                p_found = self.rate(f"{domain.name}:propose@{level}", prior=(1.0, 1.0))
                gain = p_found if readings <= 0 else p_found * outside_share
                options.append((gain - lam * domain.propose_cost(task, level), "propose", level, ""))

            alive = [h for h in hyps if h.weight > 0 and h.verified is not False]
            # Derive-group macro-action: hypotheses the domain cheaply predicts to give the same answer
            # are derived together. A one-step lookahead over single hypotheses cannot see that two
            # 50% hypotheses agreeing on the answer make answering worthwhile.
            groups: Dict[Any, List[Hyp]] = {}
            underived = [h for h in alive if not h.derived]
            for h in underived:
                guess, spent = self._predict(domain, task, h) if len(alive) > 1 else (None, 0)
                ops += spent
                groups.setdefault(("?", h.key) if guess is None else guess, []).append(h)
            # Groups predicted undefined leave the answering mass once derived, so the other groups'
            # shares are taken over what remains (answerability conditioning, as in _belief).
            unknown = sum(m.weight for m in groups.get(UNKNOWN, []))
            answering = mass - unknown
            for guess, members in groups.items():
                weight = sum(m.weight for m in members)
                if guess == UNKNOWN:
                    trial = {y: p * mass / answering for y, p in q.items()} if answering > 0 else {}
                elif answering > 0:
                    rescale = mass / answering
                    trial = {y: p * rescale for y, p in q.items()}
                    trial[guess] = trial.get(guess, 0.0) + weight / answering * rho
                else:
                    continue
                gain = self._utility(trial) - u_now
                cost = sum(domain.derive_cost(task, m, self) for m in members)
                options.append((gain - lam * cost, "derive", members, ""))
            for h in alive:
                p_h = h.weight / mass if mass > 0 else 0.0
                if h.derived and c.use_verifier and h.verified is None and _is_value(h.answer):
                    passed = dict(q)
                    passed[h.answer] += p_h * (1.0 - rho)
                    rest = [x for x in alive if x is not h]
                    rest_mass = sum(x.weight for x in rest)
                    failed = {}
                    for x in rest:
                        if x.derived and x.answer is not None and x.answer != OUTSIDE:
                            failed[x.answer] = failed.get(x.answer, 0.0) + x.weight / max(rest_mass, 1e-12) * (1.0 if x.verified else rho)
                    gain = rho * self._utility(passed) + (1 - rho) * self._utility(failed) - u_now
                    options.append((gain - lam * c.verify_ops, "verify", h, ""))

            if c.use_questions and len(alive) > 1:
                has_outside = any(h.answer == OUTSIDE for h in alive)
                horizon = c.ask_horizon if has_outside else 1
                for question, at_q, goals, cost in domain.questions(task, alive, self):
                    # Myopic VOI undervalues a question when one confirmation cannot lift a reading
                    # past OUTSIDE but h agreeing answers can (each multiplies the ratio by 1/spread).
                    # Value the best h-question plan, priced at h questions, and take its first step.
                    best = max(((self._value_of_question(alive, at_q, goals, rho, h) - lam * h * (c.ask_ops + cost)), h)
                               for h in range(1, horizon + 1))
                    options.append((best[0], "ask", question, f"voi(h={best[1]})"))

            if not options:
                break
            voc, action, target, note = max(options, key=lambda o: o[0])
            if voc <= 0:
                break
            if action == "propose":
                got = domain.propose(task, level, self)
                if got is None:
                    exhausted = True
                    trace.append(Step(cycle, "propose", f"level {level}", voc, 0, "search space exhausted"))
                    continue
                new, spent = got
                self.observe(f"{domain.name}:propose@{level}", bool(new), prior=(1.0, 1.0))
                hyps.extend(new)
                ops += spent
                trace.append(Step(cycle, "propose", f"level {level}", voc, spent, f"{len(new)} hypotheses"))
                level += 1
            elif action == "derive":
                spent = sum(domain.derive(task, h, self) for h in target)
                ops += spent
                trace.append(Step(cycle, "derive", " | ".join(h.key for h in target), voc, spent,
                                  " | ".join(h.note for h in target)))
            elif action == "verify":
                spent = domain.verify(task, target, self)
                ops += spent
                self.observe(rho_key, bool(target.verified))
                trace.append(Step(cycle, "verify", target.key, voc, spent,
                                  "pass" if target.verified else f"FAIL {target.note}"))
            else:
                spent = domain.ask(task, target, alive, self)
                ops += spent
                trace.append(Step(cycle, "ask", repr(target), voc, spent, note))

        q, mass = self._belief(hyps, self.rate(rho_key))
        if q and self._utility(q) > 0:
            answer = max(q, key=q.get)
            result = Result("answer", answer, q[answer], "best-supported answer", trace, ops, hyps)
        else:
            result = Result("abstain", None, max(q.values(), default=0.0), self._why_not(hyps, q),
                            trace, ops, hyps)
        if c.learn_patience:
            # The sample that drives the price: utility actually held at the end, per op spent.
            self.patience.update(ops, self._utility(q))
        domain.remember(task, result, self)
        self.episodes.append((domain.name, task, result))
        return result

    @staticmethod
    def _why_not(hyps: List[Hyp], q: Dict[Any, float]) -> str:
        if not hyps:
            return "no hypothesis explains the evidence"
        refuted = [h for h in hyps if h.verified is False or h.weight == 0]
        if refuted and len(refuted) == len(hyps):
            culprits = sorted(set().union(*(h.culprits for h in refuted)))
            return "every reading is inconsistent" + (f"; culprits: {', '.join(culprits)}" if culprits else "")
        if len(q) > 1:
            return "hypotheses disagree: " + ", ".join(f"{k}:{v:.2f}" for k, v in sorted(q.items(), key=str))
        if any(h.weight > 0 and not h.derived for h in hyps):
            return "not derived: the expected cost of deriving exceeds the value of an answer"
        if any(h.weight > 0 and h.answer == UNKNOWN for h in hyps):
            return "my concept is not defined for this input (" + \
                "; ".join(h.note for h in hyps if h.weight > 0 and h.answer == UNKNOWN)[:160] + ")"
        notes = sorted({h.note for h in hyps if h.note})
        return "; ".join(notes) or "confidence below threshold"
