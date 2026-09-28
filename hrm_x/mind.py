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
class Controller:
    """Knobs of the metareasoning; `lam` prices one inference op in utility units."""

    lam: float = 2e-5
    wrong_penalty: float = 1.0
    verify_ops: float = 50.0
    ask_ops: float = 2000.0
    max_cycles: int = 40
    use_verifier: bool = True
    use_questions: bool = True


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
    patience (5-HT-like) is the price of computation Controller.lam; fixed here, since nothing
        measured so far needs it to adapt.
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
                         "control": self.control}, f)

    @staticmethod
    def load(path: str) -> "Mind":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        return Mind(control=state["control"], knowledge=state["knowledge"], stats=state["stats"],
                    modulators=state["modulators"])


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
            if h.derived and h.answer is not None and h.answer != UNKNOWN:   # "undefined for me" is not an answer
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

    def _value_of_question(self, alive: List[Hyp], at_q: List[Any], goals: List[Any], rho: float) -> float:
        """E[U after hearing the true answer at q] - U, using each hypothesis's goal prediction."""
        def value(group: List[Tuple[Hyp, Any]]) -> float:
            mass = sum(h.weight for h, g in group if g is not None and g != UNKNOWN)
            if mass <= 0:
                return 0.0
            q: Dict[Any, float] = {}
            for h, g in group:
                if g is not None and g != UNKNOWN:
                    q[g] = q.get(g, 0.0) + h.weight / mass * (1.0 if h.verified else rho)
            return self._utility(q)

        pairs = list(zip(alive, goals))
        total = sum(h.weight for h in alive)
        after = 0.0
        outcomes: Dict[Any, List[Tuple[Hyp, Any]]] = {}
        for (h, g), a in zip(pairs, at_q):
            outcomes.setdefault(UNKNOWN if a is None else a, []).append((h, g))
        for group in outcomes.values():
            after += sum(h.weight for h, _ in group) / total * value(group)
        return after - value(pairs)

    # ---------- the loop ----------

    def solve(self, domain: Domain, task: Any) -> Result:
        c = self.control
        hyps: List[Hyp] = []
        trace: List[Step] = []
        ops, level, exhausted = 0, 0, False
        rho_key = f"{domain.name}:verify-pass"
        for cycle in range(1, c.max_cycles + 1):
            rho = self.rate(rho_key)
            q, mass = self._belief(hyps, rho)
            u_now = self._utility(q)
            options: List[Tuple[float, str, Any, str]] = []

            if mass <= 0 and not exhausted:
                p_found = self.rate(f"{domain.name}:propose@{level}", prior=(1.0, 1.0))
                options.append((p_found - c.lam * domain.propose_cost(task, level), "propose", level, ""))

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
                options.append((gain - c.lam * cost, "derive", members, ""))
            for h in alive:
                p_h = h.weight / mass if mass > 0 else 0.0
                if h.derived and c.use_verifier and h.verified is None and h.answer is not None \
                        and h.answer != UNKNOWN:
                    passed = dict(q)
                    passed[h.answer] += p_h * (1.0 - rho)
                    rest = [x for x in alive if x is not h]
                    rest_mass = sum(x.weight for x in rest)
                    failed = {}
                    for x in rest:
                        if x.derived and x.answer is not None:
                            failed[x.answer] = failed.get(x.answer, 0.0) + x.weight / max(rest_mass, 1e-12) * (1.0 if x.verified else rho)
                    gain = rho * self._utility(passed) + (1 - rho) * self._utility(failed) - u_now
                    options.append((gain - c.lam * c.verify_ops, "verify", h, ""))

            if c.use_questions and len(alive) > 1:
                for question, at_q, goals, cost in domain.questions(task, alive, self):
                    voi = self._value_of_question(alive, at_q, goals, rho)
                    options.append((voi - c.lam * (c.ask_ops + cost), "ask", question, f"voi={voi:.3f}"))

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
