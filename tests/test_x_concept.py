"""The v0.5 concept route: exact inference, determinacy, coverage, and its measured limits.

These tests pin behaviour that has no v0.4 equivalent, and several of them pin
*negative* results on purpose. `test_a_coincidental_symmetric_fit_is_still_confidently_wrong`
and `test_one_question_destroys_the_coverage_guard` exist so that a future change
which claims to have closed the wrong-answer channel has to fail here rather
than quietly pass.

Theorem, proposition and definition numbers refer to the v0.5 delivery,
`docs/03_mathematical_foundations.md`.

Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
"""
from fractions import Fraction as F

import numpy as np
import pytest

from hrm_x.concepts import FAMILIES, ConceptConfig, fit
from hrm_x.consolidation import MacroLibrary, consolidate
from hrm_x.domains.concept import Concept, ConceptTask
from hrm_x.mind import Controller, Mind
from hrm_x.qlinalg import in_row_space, rank, rref, solve

ROT90 = np.array([[0, -1], [1, 0]])


def _grid():
    return np.array([[0, 0, 6, 6],
                     [0, 6, 6, 6],
                     [0, 6, 6, 6]], dtype=np.int64)


# ---------- qlinalg: the determinacy calculus is exact over Q ----------


def test_rref_and_rank_are_exact_over_rationals():
    R, pivots = rref([[F(1, 3), F(2, 3)], [F(1, 3), F(1, 6)]])
    assert pivots == [0, 1]
    assert R[0] == [F(1), F(0)]
    assert R[1] == [F(0), F(1)]
    assert rank([[F(1, 3), F(2, 3)], [F(1, 3), F(1, 6)]]) == 2


def test_rank_counts_independent_rows_not_rows_in_the_list():
    assert rank([[F(1), F(1)], [F(2), F(2)], [F(1), F(-1)]]) == 2
    assert rank([[F(1), F(1)], [F(2), F(2)]]) == 1


def test_solve_refuses_an_inconsistent_system():
    # 2p = 1 and 2p = 3 cannot both hold.
    assert solve([[F(2)], [F(2)]], [F(1), F(3)]) is None
    assert solve([[F(2)]], [F(1)]) == [F(1, 2)]


def test_in_row_space_is_exactly_the_determinacy_test():
    A = [[F(1), F(1)]]                       # the span is the diagonal, {(t, t)}
    assert in_row_space(A, [F(2), F(2)])
    assert not in_row_space(A, [F(1), F(0)])
    assert in_row_space([], [F(0), F(0)])    # the empty row space is {0}
    assert not in_row_space([], [F(1)])


# ---------- concepts: Prop. 4, Def. 3, and the coverage rule ----------


def test_a_query_outside_the_evidence_span_has_no_opinion():
    """Theorem 3. With one worked example the family `alpha x + beta y` is
    rank 1, so the span of the evidence is the single direction (2, 3). The
    query (4, 6) lies in it and is answerable; (1, 1) does not, and the reading
    must say nothing rather than extrapolate.

    Coverage is switched off so that what is under test is the row-space test
    and not the variation rule, which has its own test below.
    """
    reading = [f for f in fit([((F(2), F(3)), F(5))], ConceptConfig()).fitted
               if f.family.name == "biaffine{x,y}"][0]
    assert reading.rank == 1 and reading.surplus == 0
    assert reading.value(F(4), F(6), coverage=False) == F(10)
    assert reading.value(F(1), F(1), coverage=False) is None


def test_fitted_parameters_buy_no_evidence():
    """Prop. 4: each row the family *predicts* rather than fits is worth exactly
    log2|Y| bits, and nothing else changes."""
    cfg = ConceptConfig()
    base = [((F(2), F(3)), F(5))]
    one = base + [((F(4), F(6)), F(10))]
    two = one + [((F(6), F(9)), F(15))]

    def log2w(evidence):
        f = [x for x in fit(evidence, cfg).fitted if x.family.name == "biaffine{x,y}"][0]
        return f.log2w, f.rank, f.surplus, f.params

    w0, r0, s0, p0 = log2w(base)
    w1, r1, s1, p1 = log2w(one)
    w2, r2, s2, p2 = log2w(two)
    assert (r0, s0, r1, s1, r2, s2) == (1, 0, 1, 1, 1, 2)
    assert p1 == p0 and p2 == p0, "the parameters are the same in all three"
    assert w1 - w0 == pytest.approx(cfg.log2_answer_space)
    assert w2 - w0 == pytest.approx(2 * cfg.log2_answer_space)


def test_the_reserved_bottom_hypothesis_carries_its_prior_share():
    """Def. 3. `a^2 + b` on three rows is not explained by anything in the
    language, so the posterior collapses onto the reserved hypothesis. Its share
    has to grow as its prior worsens, and vanish in the closed world."""
    rows = [((F(2), F(3)), F(7)), ((F(4), F(1)), F(17)), ((F(3), F(2)), F(11))]

    def bottom(bits: float) -> float:
        return fit(rows, ConceptConfig(bits_bottom=bits)).posterior()[1]

    confident, cautious, closed = bottom(1.0), bottom(8.0), bottom(float("inf"))
    assert confident > cautious > 0.9
    assert closed == 0.0


def test_the_coverage_rule_refuses_an_argument_that_never_varied():
    """The open-world rule. With `y` pinned at 3 the reading `alpha x + gamma`
    fits both rows, but it is not symmetric, so it loses its entitlement at an
    unseen `y`: any `g(x) h(y)` with `h` fixed on the seen values fits evidence
    in which `y` never moved, and that is exactly what the rule excludes."""
    reading = [f for f in fit([((F(2), F(3)), F(7)), ((F(4), F(3)), F(19))],
                              ConceptConfig()).fitted
               if f.family.name == "biaffine{x,1}"][0]
    assert reading.value(F(2), F(3)) == F(7) and reading.value(F(4), F(3)) == F(19)
    assert reading.symmetric is False
    assert reading.covered(F(5), F(3)) is True        # x varied, so x is fine
    assert reading.covered(F(5), F(7)) is False       # y never varied, not symmetric
    assert reading.value(F(5), F(7)) is None
    # Switch the rule off and the same reading answers at (5, 7): 6*5 - 5.
    assert reading.value(F(5), F(7), coverage=False) == F(25)


def test_one_question_destroys_the_coverage_guard():
    """Measured mechanism, pinned on purpose.

    The coverage rule asks "did this argument ever take a different value?". One
    question about a *new* y makes it take a second value, and the guard stops
    firing -- for every reading, not only the wrong ones. So the controller's own
    act of asking re-opens the coincidental-fit channel the rule exists to close.

    `experiments/x_v05.py` measures the consequence: 5 confidently wrong answers
    on 6 out-of-language queries, every one of them after an `ask`.
    """
    square_add = lambda a, b: a * a + b
    task = ConceptTask.from_op(square_add, [(F(2), F(3)), (F(4), F(3))], (F(5), F(7)),
                               symbol="*", oracle=square_add)
    domain, mind = Concept(), Mind(Controller())
    before, _ = domain.propose(task, 0, mind)
    asymmetric_before = [h for h in before if not h.payload.symmetric]
    assert asymmetric_before, "expected at least one asymmetric reading"
    assert all(not h.payload.covered(F(5), F(7)) for h in asymmetric_before)

    domain.ask(task, (F(1), F(1)), before, mind)
    after, _ = domain.propose(task, 1, mind)
    asymmetric_after = [h for h in after if not h.payload.symmetric]
    assert asymmetric_after, "the ask left no asymmetric reading to re-open the channel"
    assert any(h.payload.covered(F(5), F(7)) for h in asymmetric_after)


# ---------- the domain, end to end through the real mind ----------


def _solve(op, evidence, query, **kw):
    task = ConceptTask.from_op(op, evidence, query, symbol="*", oracle=op)
    return Mind(Controller(**kw)).solve(Concept(), task)


def test_the_mind_answers_multiplication_from_two_examples_and_verifies_it():
    """2x3=6 and 3x4=12 leave exactly one bi-affine reading with a surplus
    row, the query is in the span, and the answer is derived and then
    independently re-derived."""
    result = _solve(lambda a, b: a * b, [(F(2), F(3)), (F(3), F(4))], (F(10), F(7)))
    assert result.status == "answer" and result.answer == F(70)
    verified = [h for h in result.hypotheses if h.verified]
    assert verified, "no reading passed the dual check"
    assert all(h.answer == F(70) for h in verified)


def test_verification_reproduces_the_answer_by_the_dual_solve():
    """`verify` reads the number off the right-hand side through
    `A^T lambda = r`, not off the fitted parameters, so the two computations are
    independent and their agreement is exact over the rationals."""
    ab1 = lambda a, b: a * b + 1
    task = ConceptTask.from_op(ab1, [(F(2), F(3)), (F(4), F(1))], (F(5), F(2)),
                               symbol="+1", oracle=ab1)
    domain, mind = Concept(), Mind(Controller())
    got, spent = domain.propose(task, 0, mind)
    assert got and spent > 0
    hyp = [h for h in got if h.payload.family.name == "biaffine{xy,1}"][0]
    domain.derive(task, hyp, mind)
    assert hyp.answer == F(11)

    dual = solve([list(col) for col in zip(*hyp.payload.A)],
                 list(hyp.payload.family.row(F(5), F(2))))
    assert dual is not None
    assert sum((lam * z for lam, z in zip(dual, [z for _, z in task.evidence()])),
               F(0)) == F(11)

    assert domain.verify(task, hyp, mind) > 0
    assert hyp.verified is True and hyp.note == ""


def test_verification_catches_a_reading_whose_parameters_were_corrupted():
    """The dual is only worth computing if it can fail. Move the parameters and
    re-derive: the primal now says one number and the dual another."""
    add = lambda a, b: a + b
    task = ConceptTask.from_op(add, [(F(1), F(1)), (F(2), F(3))], (F(5), F(2)),
                               symbol="+", oracle=add)
    domain, mind = Concept(), Mind(Controller())
    hyp = [h for h in domain.propose(task, 0, mind)[0]
           if h.payload.family.name == "biaffine{x,y}"][0]
    domain.derive(task, hyp, mind)
    assert hyp.answer == F(7)
    assert domain.verify(task, hyp, mind) > 0 and hyp.verified is True

    hyp.payload.params = [F(10), F(1)]          # the primal now computes 52
    domain.derive(task, hyp, mind)
    assert hyp.answer == F(52)
    domain.verify(task, hyp, mind)
    assert hyp.verified is False
    assert "dual says" in hyp.note


def test_a_coincidental_symmetric_fit_is_still_confidently_wrong():
    """The measured limit, pinned on purpose.

    `a^2 + b` is outside the hypothesis language. On this evidence the full
    bi-affine family fits both rows, the mind buys one question, the coverage
    guard stops applying, the dual confirms the result, and the answer is 47
    where the truth is 32. Doc 13 calls this the coincidental-fit channel (Q1):
    the open-world posterior removes errors caused by a *missing* hypothesis, not
    errors caused by a *simpler wrong* one that happens to fit.

    `experiments/x_v05.py` measures it at 5 wrong answers out of 6 out-of-language
    queries, against 0 wrong and 4 abstentions for the v0.4 route. If a later
    change claims to close the channel, this test is what has to fail.
    """
    square_add = lambda a, b: a * a + b
    result = _solve(square_add, [(F(2), F(3)), (F(4), F(3))], (F(5), F(7)))
    assert result.status == "answer"
    assert result.answer == F(47)
    assert result.answer != square_add(F(5), F(7))
    assert any(s.action == "ask" for s in result.trace)


def test_the_language_is_exhausted_rather_than_reshuffled():
    """A second `propose` on the same task must report exhaustion, or the
    controller keeps proposing forever."""
    mul = lambda a, b: a * b
    task = ConceptTask.from_op(mul, [(F(2), F(3))], (F(4), F(6)), symbol="*", oracle=mul)
    domain, mind = Concept(), Mind(Controller())
    assert domain.propose(task, 0, mind)[0]
    assert domain.propose(task, 1, mind)[0] is None


def test_propose_charges_a_fit_and_derive_charges_only_a_lookup():
    """The asymmetry that is the point of the route: the expensive part moved
    into `propose`, and deriving is now a lookup."""
    mul = lambda a, b: a * b
    task = ConceptTask.from_op(mul, [(F(2), F(3)), (F(3), F(4))], (F(4), F(6)),
                               symbol="*", oracle=mul)
    domain, mind = Concept(), Mind(Controller())
    assert domain.propose_cost(task, 0) > 0
    hyp = domain.propose(task, 0, mind)[0][0]
    assert 0 < domain.derive_cost(task, hyp, mind) < domain.propose_cost(task, 0)
    assert domain.derive(task, hyp, mind) == len(hyp.payload.A) + 1
    assert hyp.derived and hyp.answer == F(24)


def test_a_learned_concept_survives_json_with_its_exact_parameters():
    """`remember` keeps the version space, and `Mind.save` writes JSON -- a
    `Fraction` is not a JSON scalar, so the parameters are stored as strings and
    come back losslessly."""
    import json

    mul = lambda a, b: a * b
    task = ConceptTask.from_op(mul, [(F(2), F(3)), (F(3), F(4))], (F(4), F(6)),
                               symbol="*", oracle=mul)
    mind = Mind(Controller())
    mind.solve(Concept(), task)
    stored = mind.knowledge["concepts"]["concepts"]["*"]
    assert stored["readings"] and stored["evidence"]
    assert all(isinstance(p, str) for r in stored["readings"] for p in r["params"])
    assert {"2", "3", "6"} <= {cell for row in stored["evidence"] for cell in row}
    assert json.loads(json.dumps(mind.knowledge)) == mind.knowledge


# ---------- consolidation: the MDL admission rule ----------


def _multiplicative_reading():
    """The fitted `x*y` instance, taken from an actual fit rather than written
    out, so the test cannot drift from the representation the rule reads."""
    belief = fit([((F(2), F(3)), F(6)), ((F(3), F(4)), F(12)), ((F(1), F(5)), F(5))],
                 ConceptConfig())
    for f in belief.fitted:
        if f.family.name == "biaffine{xy}" and f.surplus >= 1:
            return f
    raise AssertionError("multiplication is not in the belief")


def test_a_macro_that_cannot_pay_for_itself_is_rejected():
    """One solved task cannot amortise its own description: the macro costs
    more bits than re-describing that one task."""
    lib = MacroLibrary()
    report = consolidate(lib, [("mul", _multiplicative_reading())], [], ConceptConfig())
    assert report and all(not r["added"] for r in report)
    assert len(lib.macros) == 0
    assert all(r["gain_bits"] < 0 for r in report)


def test_repeated_behaviour_classes_pay_for_the_macro():
    """The same reading solved on three tasks amortises its declaration, which
    is the whole point of a library -- and admitting it changes the hypothesis
    language, not just its description."""
    lib = MacroLibrary()
    report = consolidate(lib, [("mul-1", _multiplicative_reading()),
                               ("mul-2", _multiplicative_reading()),
                               ("mul-3", _multiplicative_reading())],
                         [], ConceptConfig())
    assert [r for r in report if r["added"]], report
    assert len(lib.macros) == 1
    assert lib.macros[0].fn(F(6), F(7)) == F(42)
    assert lib.families() != list(FAMILIES)


def test_a_rejected_library_leaves_the_hypothesis_language_untouched():
    lib = MacroLibrary()
    consolidate(lib, [("mul", _multiplicative_reading())], [], ConceptConfig())
    assert lib.families() == list(FAMILIES)


# ---------- the phasor algebra (Prop. 8) ----------


def test_binding_round_trips_and_characters_are_homomorphisms():
    from hrm_x.algebra import ReferenceAlgebra

    alg = ReferenceAlgebra(d=256, seed=7)
    role, filler = alg.random_codes(1)[0], alg.random_codes(1)[0]
    assert np.abs(alg.unbind(role, alg.bind(role, filler)) - filler).max() < 1e-12
    omega = alg.random_codes(1)[0]
    assert np.abs(alg.character(omega, 2.0) - alg.character(omega, 1.0) ** 2).max() < 1e-12
    assert np.abs(alg.character(omega, 6.0)
                  - alg.character(alg.scale(omega, 3.0), 2.0)).max() < 1e-12


def test_cleanup_recovers_the_bound_filler_at_the_dimension_the_bound_asks_for():
    """Prop. 8: K distinct slots each bind one of M fillers; unbinding the
    queried slot and cleaning up has to return that filler. At d = 1024,
    K = 8, M = 100 the bound is under 1%, and on a sample of 100 it never fires."""
    from hrm_x.algebra import ReferenceAlgebra, cleanup_error_bound

    d, k, m = 1024, 8, 100
    alg = ReferenceAlgebra(d=d, seed=3)
    fillers, slots = alg.random_codes(m), alg.random_codes(k)
    errors = 0
    for i in range(m):
        s = alg.superpose([alg.bind(slots[j], f) for j, f in enumerate(fillers[i:i + k])])
        errors += alg.cleanup(fillers, alg.unbind(slots[0], s)) != i
    assert cleanup_error_bound(d, k, m) < 0.01
    assert cleanup_error_bound(d, 1, m) == 0.0
    assert errors == 0


# ---------- the grid operator layer (Theorem 6) ----------


def test_d4_is_closed_and_its_composition_is_exact():
    """Theorem 6 after projection: composition of estimated operators is exact
    for any length, so the error bound is zero rather than merely small."""
    from hrm_x.operators import D4, D4_NAMES, apply_global

    assert len(D4) == len(D4_NAMES) == 8
    for i in range(8):
        for j in range(8):
            product = apply_global(D4[i], {}, apply_global(D4[j], {}, _grid()))
            assert np.array_equal(product, apply_global(D4[i] @ D4[j], {}, _grid()))


def test_the_global_layer_reproduces_a_demonstrated_rotation():
    from hrm_x.operators import D4_NAMES, apply_global, solve_grid

    demo_in = _grid()
    demo_out = apply_global(ROT90, {}, demo_in)
    test = np.array([[6, 0, 0, 0],
                     [0, 0, 6, 6],
                     [0, 6, 6, 6]], dtype=np.int64)
    got = solve_grid([(demo_in, demo_out)], test)
    assert got.status == "answer" and got.grid is not None
    assert D4_NAMES[2] == "rot90"
    assert np.array_equal(got.grid, apply_global(ROT90, {}, test))


# ---------- the situation model ----------


def _taught_memory():
    """Two stated entities and one transfer, each learned from a single worked
    example: a frame's coefficients are an exact solve of one equation per
    observed entity."""
    from hrm_x.situation import Memory, teach

    mem = Memory()
    state = teach(mem, {"Ann": F(0), "Ben": F(0)}, "Ann has 5 pears .",
                  {"Ann": F(5), "Ben": F(0)})
    assert (state.alpha, state.arity) == (F(1), 1)
    teach(mem, {"Ann": F(0), "Ben": F(0)}, "Ben has 4 pears .", {"Ann": F(0), "Ben": F(4)})
    transfer = teach(mem, {"Ann": F(5), "Ben": F(4)}, "Ann gives 2 pears to Ben .",
                     {"Ann": F(3), "Ben": F(6)})
    assert (transfer.alpha, transfer.beta) == (F(-1), F(1))
    return mem


_STORY = ["Ann has 5 pears .", "Ben has 4 pears .", "Ann gives 2 pears to Ben ."]


def test_one_taught_frame_per_event_answers_the_story_and_conserves_the_total():
    from hrm_x.situation import answer

    mem = _taught_memory()
    ann = answer(mem, _STORY, "How many pears does Ann have ?")
    ben = answer(mem, _STORY, "How many pears does Ben have ?")
    assert (ann.status, ann.answer, ann.verified) == ("answer", F(3), True)
    assert (ben.status, ben.answer, ben.verified) == ("answer", F(6), True)
    assert ann.answer + ben.answer == F(9)      # the conservation check, independently


def test_an_untaught_frame_abstains_rather_than_guessing():
    from hrm_x.situation import answer

    lost = answer(_taught_memory(), _STORY + ["Ben sells 4 pears to Ann ."],
                  "How many pears does Ann have ?")
    assert lost.status == "abstain" and lost.answer is None and not lost.verified
    assert lost.unknown_frames and "never taught" in lost.reason


def test_a_story_with_no_stated_initial_quantity_is_not_answered():
    """The situation model refuses to invent a starting state, which is the same
    discipline ADR-009 credited to `reading_defect` on the v0.4 route."""
    from hrm_x.situation import Memory, answer, teach

    mem = Memory()
    teach(mem, {"Ann": F(5), "Ben": F(0)}, "Ann gives 2 pears to Ben .",
          {"Ann": F(3), "Ben": F(2)})
    got = answer(mem, ["Ann gives 2 pears to Ben ."], "How many pears does Ann have ?")
    assert got.status == "abstain"
    assert "initial quantity" in got.reason
