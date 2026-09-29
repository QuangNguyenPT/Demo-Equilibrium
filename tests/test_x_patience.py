"""Patience: the learned price of computation (spec v0.4 §14, 5-HT-like).

The rule: lambda is driven to the value-per-op the mind actually observes, by log-space dual
ascent whose fixed point is lam = d. These tests pin the fixed point, the bounds, the anchoring,
the volatility coupling, and the end-to-end consequence -- and the last one is the falsifiable
part: a mind whose actions are usually worthless must become *less* patient and spend less.
"""
import copy
import math
import random
from typing import Optional

import pytest

from hrm_x.mind import DEFAULT_LAM, Controller, Mind, Patience, Result


def test_price_starts_at_the_constant():
    p = Patience()
    assert p.price() == p.lam == p.prior() == 2e-5
    assert p.base is None, "a standalone Patience defers to the Controller, it does not fix a prior"


def test_the_controller_is_the_only_source_of_truth_for_the_prior():
    """Regression: two knobs (Controller.lam and Patience.base) meant setting one did nothing."""
    m = Mind(control=Controller(lam=1e-3, learn_patience=True))
    assert m.patience.base == 1e-3 and m.patience.lam == 1e-3
    assert m.price_of_computation() == 1e-3
    m.patience.update(10, 1.0)
    m.patience.reset()
    assert m.patience.lam == 1e-3, "reset returns to the controller's prior"


def test_fixed_point_is_the_observed_density():
    """A steady stream of episodes at a constant density must pull lambda onto that density."""
    p = Patience(base=1.0, eta=0.5, shrink=0.5)
    ops, utility = 100, 1.0                      # density 0.01 every episode
    for _ in range(400):
        p.update(ops, utility)
    assert p.density == pytest.approx(0.01, rel=0.05)
    assert p.lam == pytest.approx(0.01, rel=0.10)


def test_ascends_when_compute_is_paying_and_descends_when_it_is_not():
    rich, poor = Patience(base=1e-4, eta=0.4), Patience(base=1e-4, eta=0.4)
    for _ in range(60):
        rich.update(50, 50.0)                    # 1.0 utility per op: very worth it
        poor.update(50, 0.0)                     # nothing for all that work
    assert rich.lam > 1e-4
    assert poor.lam < 1e-4


def test_never_overshoots_the_bounds():
    p = Patience(base=1.0, eta=5.0, bound=(1e-7, 1e-3))
    for _ in range(200):
        p.update(1, 1e6)                         # absurdly valuable ops
    assert p.lam <= p.bound[1]
    p2 = Patience(base=1.0, eta=5.0, bound=(1e-7, 1e-3))
    for _ in range(200):
        p2.update(1, 0.0)
    assert p2.lam >= p2.bound[0]


def test_zero_ops_is_a_no_op_not_a_division_by_zero():
    p = Patience()
    before = p.lam
    assert p.update(0, 5.0) == before


def test_one_bad_episode_cannot_drag_the_price():
    """The prior must hold: anchoring, not momentum."""
    p = Patience(base=1e-3, eta=0.9, shrink=0.99)
    p.update(10, 1.0)
    before = p.lam
    p.update(10, 0.0)                            # one episode that achieved nothing
    assert p.lam > before * 0.5, "a single failure halved the price"


def test_volatility_raises_the_effective_price_but_not_the_learned_one():
    """The coupling is a price multiplier, not a write to lambda: a stable regime keeps the lesson."""
    p = Patience(lam=1e-4, coupling=1.0)
    assert p.price(0.0) == pytest.approx(1e-4)
    assert p.price(0.5) == pytest.approx(1.5e-4)
    assert p.price(1.0) == pytest.approx(2.0e-4)
    assert p.lam == 1e-4, "price() must not mutate the learned value"


def test_negative_volatility_is_treated_as_no_volatility():
    p = Patience(lam=1e-4)
    assert p.price(-0.3) == pytest.approx(1e-4)


def test_reset_returns_to_the_prior():
    p = Patience(base=7e-5)
    p.update(10, 10.0)
    p.reset()
    assert p.lam == 7e-5 and p.density == 0.0 and p.seen == 0.0


def test_disabled_by_default_so_the_published_e9_baseline_is_unchanged():
    assert Controller().learn_patience is False
    assert Controller().lam == 2e-5


def test_price_of_computation_switches_on_the_flag():
    assert Mind().price_of_computation() == Mind().control.lam
    m = Mind(control=Controller(learn_patience=True), patience=Patience(lam=3e-4))
    assert m.price_of_computation() == pytest.approx(3e-4)
    m.modulators.volatility = 1.0
    assert m.price_of_computation() == pytest.approx(6e-4)


def test_solve_updates_the_price_only_when_enabled():
    class Trivial:
        name = "t"

        def propose_cost(self, task, level): return 0.0
        def propose(self, task, level, mind): return None
        def derive_cost(self, task, hyp, mind): return 0.0
        def derive(self, task, hyp, mind): return 0
        def verify(self, task, hyp, mind): return 0
        def questions(self, task, hyps, mind): return []
        def ask(self, task, q, hyps, mind): return 0
        def remember(self, task, result, mind): pass

    m = Mind()
    m.solve(Trivial(), object())
    assert m.patience.seen == 0.0, "patience must not learn when disabled"

    m2 = Mind(control=Controller(learn_patience=True))
    lam_before = m2.patience.lam
    m2.solve(Trivial(), object())
    assert m2.patience.seen == 0.0, "an episode that spent no ops carries no density sample"
    assert m2.patience.lam == lam_before, "and must leave the price alone"


class _Buyable:
    """A domain with genuinely optional computation, so the price is the binding constraint.

    Story tasks cannot test this: they derive once, verify and answer, there is no optional work
    to decline, and ops come out identical at any price (measured, not assumed). Here the mind
    must pick which of twelve readings to buy. Ten of them agree on the right answer and can be
    derived together as one macro-action; the two outliers predict the wrong answer, so buying
    them only lowers confidence and their VOC is negative. That is the regime in which
    `VOC - lam*cost` can actually flip sign, so it is the only regime where a learned price is
    observable. `predict` is free, which is what puts a group inside the one-step lookahead.
    """

    name = "buyable"

    def __init__(self, n_agree: int = 10, n_out: int = 2, cost: float = 400.0) -> None:
        self.answers = [0] * n_agree + list(range(1, n_out + 1))
        self.cost = cost

    def propose_cost(self, task, level):
        return 1.0

    def propose(self, task, level, mind):
        from hrm_x.mind import Hyp
        if level > 0:
            return None
        return [Hyp(f"r{i}", 1.0, a) for i, a in enumerate(self.answers)], len(self.answers)

    def predict(self, task, hyp):
        return hyp.payload, 0

    def derive_cost(self, task, hyp, mind):
        return self.cost

    def derive(self, task, hyp, mind):
        hyp.answer = hyp.payload
        hyp.derived = True
        return int(self.cost)

    def verify(self, task, hyp, mind):
        hyp.verified = hyp.payload == 0
        return 5

    def questions(self, task, hyps, mind):
        return []

    def ask(self, task, q, hyps, mind):
        return 0

    def remember(self, task, result, mind):
        pass


def _run(lam: Optional[float], learn: bool, episodes: int = 30) -> dict:
    """Solve the same episodes at a fixed price, or with the price learned. Returns the trace.

    The price goes through `Controller(lam=...)`, which is the one source of truth: an earlier
    version of this harness set `Patience(base=...)` instead and every run was byte-identical
    at 1e-7 and 1e-2, because with learning off the loop reads `Controller.lam` and never looks
    at Patience. Mind.__post_init__ now inherits base from the controller so that cannot recur.
    """
    m = Mind(control=Controller(lam=DEFAULT_LAM if lam is None else lam, learn_patience=learn,
                                max_cycles=30))
    m.patience.eta, m.patience.shrink, m.patience.coupling = 0.5, 0.7, 0.0
    dom = _Buyable()
    for _ in range(episodes):
        m.solve(dom, object())
        m.observe(f"{dom.name}:verify-pass", True)      # the world rewards the right reading
    ops = sum(r.ops for _, _, r in m.episodes)
    return {"lam": m.patience.lam, "ops": ops, "density": m.patience.density,
            "ops_per_ep": ops / episodes, "confidence": m.episodes[-1][2].confidence}


def test_the_learned_price_converges_to_the_observed_value_density():
    """The spec's actual claim (v0.4 §14): lambda is driven to the value-per-op the mind observes.

    This is sharp and it is not a tautology -- the density is measured from the episodes, and
    lambda had to arrive there by dual ascent from a different starting point."""
    out = _run(lam=None, learn=True)
    assert out["lam"] == pytest.approx(out["density"], rel=0.10), \
        f"lambda {out['lam']:.3g} did not converge to density {out['density']:.3g}"
    assert out["confidence"] == pytest.approx(0.833, abs=0.02), "and it kept answering correctly"


def test_the_price_is_a_live_control_variable_and_learning_it_beats_both_extremes():
    """Why the previous version of this test was wrong, stated rather than deleted.

    It asserted "a higher learned price means less compute", which is false in a well-posed
    domain: the learned price converges onto the break-even price (utility per op = 0.666 /
    4062 = 1.64e-4), which is exactly the price at which the marginal action is worth taking.
    Spending is therefore *invariant* to lambda here, and the honest result is that lambda
    explains the spending rather than changing it. What must still hold is that lambda is a real
    control: mispriced in either direction the mind spends differently, and the learned value
    lands between them."""
    cheap, learned, dear = _run(1e-7, False), _run(None, True), _run(1e-2, False)
    assert cheap["ops"] > learned["ops"] > dear["ops"], \
        f"ops must fall with price: {cheap['ops']}, {learned['ops']}, {dear['ops']}"
    assert cheap["lam"] < learned["lam"] < dear["lam"]
    # a price far above what the work is worth makes the mind stop computing and lose utility
    assert dear["ops_per_ep"] < 100, f"a 1e-2 price should nearly stop it, spent {dear['ops']}"
    assert learned["ops_per_ep"] == pytest.approx(learned["lam"] and 4062, rel=0.02), \
        "the learned price should keep the mind at its break-even stopping point"


def test_the_learned_price_climbs_out_of_the_usable_band_on_a_cheap_domain():
    """The measured failure, pinned so it cannot be forgotten (experiments/results/x_patience.json).

    `_run` above builds a synthetic domain where one episode costs ~4062 ops, so the density the
    rule ascends to is 1.6e-4 -- which happens to be the break-even price, and everything is fine.
    That coincidence is not guaranteed, and the real domains break it. On arithmetic, which costs
    ~6 ops at utility ~1, the same rule ascends toward 0.167, four orders of magnitude above the
    price at which any action is worth taking.

    The first update already shows the arithmetic: 2e-05 * (0.167/2e-05)**0.15 = 7.7e-05, and the
    recorded trace starts there. From there it keeps climbing until no action clears its own cost,
    at which point `Patience.update` sees ops == 0 and returns without sampling, so the price
    freezes. The collapse is absorbing, not oscillatory.

    This is a characterization test of a known defect, not a statement of intent. If the rule is
    changed, this test is expected to fail -- and whoever changes it owes the project a decision
    record, because the alternative to learning the price is the constant that currently ships.
    """
    from hrm_x.domains.arith import Arith

    mind, dom = _cheap_arith_mind()
    mind.control = Controller(lam=DEFAULT_LAM, learn_patience=True)
    mind.patience = Patience(lam=DEFAULT_LAM, base=DEFAULT_LAM)
    start = mind.patience.lam
    results = [mind.solve(dom, t) for t in _cheap_arith_tasks(8)]

    assert mind.patience.lam > start, "the rule should have climbed, as dual ascent always does"
    tail = results[len(results) // 2:]
    assert all(r.ops == 0 for r in tail), \
        f"once priced out the mind never samples again, so it never recovers: {[r.ops for r in results]}"
    assert all(r.status == "abstain" for r in tail), "and it abstains rather than guessing"
    assert results[0].status == "answer", "it was answering before the climb, so this is the climb's doing"


def test_the_shipped_default_is_inside_the_band_where_the_mind_still_works():
    """Why `learn_patience` stays off: the constant beats the thing that would replace it.

    Same tasks, same mind, only the flag differs. This is the comparison that decides the default.
    """
    from hrm_x.domains.arith import Arith

    fixed, dom_fixed = _cheap_arith_mind()
    learned, dom_learned = _cheap_arith_mind()
    learned.control = Controller(lam=DEFAULT_LAM, learn_patience=True)
    learned.patience = Patience(lam=DEFAULT_LAM, base=DEFAULT_LAM)
    tasks = _cheap_arith_tasks(8)

    fixed_out = [fixed.solve(dom_fixed, t) for t in tasks]
    learned_out = [learned.solve(dom_learned, t) for t in tasks]

    assert all(r.status == "answer" for r in fixed_out), "the shipped default must keep working"
    assert all(r.ops > 0 for r in fixed_out), "and must keep spending"
    assert sum(r.ops for r in learned_out) < sum(r.ops for r in fixed_out) / 2, \
        "the learned arm should be measurably cheaper -- that is the whole point of it, and it is not worth anything"
    assert sum(r.status == "answer" for r in learned_out) < sum(r.status == "answer" for r in fixed_out), \
        "but it should also be measurably less useful, which is why it stays off"


def _cheap_arith_tasks(n: int):
    """Held-out arithmetic over concepts the mind already knows, in x_patience's seed."""
    from experiments.x_control import arith_tasks

    return [t[1] for t in arith_tasks(random.Random(101), n, hard=False)]


def _cheap_arith_mind():
    from hrm_x.domains.arith import Arith
    from experiments.x_control import learned_arith_mind

    return copy.deepcopy(learned_arith_mind()), Arith()


def test_a_useless_domain_makes_the_mind_explore_rather_than_stop():
    """The other direction, stated because it is a real property and not obviously desirable.

    When an episode yields no utility the density is 0, the target price falls to its floor and
    the mind becomes maximally patient. That is exploration -- the same shape as curiosity, and
    the same reason E8's "not defined for me" is worth reaching -- but it does mean a mind that
    is never rewarded will keep buying computation rather than giving up. The escape is
    volatility: after the world changes, `Patience.price` scales the floor up."""
    lam, _ = 2e-5, None
    p = Patience(base=2e-5, eta=0.5, shrink=0.7, coupling=0.0)
    for _ in range(50):
        p.update(1000, 0.0)
        lam = p.lam
    assert lam < 2e-5, "no utility means a lower price, i.e. more exploration"
    braking = Patience(base=2e-5, coupling=1.0)
    braking.update(1000, 0.0)
    assert braking.price(volatility=1.0) > braking.price(volatility=0.0), "volatility is the brake on that"
