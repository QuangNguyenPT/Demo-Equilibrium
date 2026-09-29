"""How the learned DSL is allowed to touch search (tests for the arith side of hrm_x/library.py).

`tests/test_x_library.py` covers the compression itself. These cover the two ways it could quietly
damage the domain, both of which were real:

* a library that is merely *present* must not change the search. Adding a constructor multiplies the
  programs at every size, so enumerating in the extended language at the same bound cost 9.4x more
  (39,511 vs 4,218 ops). A present library is a description facility until it is asked to be a search
  facility.
* reducing the size bound must be paid for, not assumed, and by default it is not, because on the
  realistic arith corpus it lost at every bound from 4 to 10.
"""
from fractions import Fraction

import pytest

from hrm_x.domains.arith import (ONE, X, Y, ZERO, Arith, ArithTask, programs, programs_with,
                                 show)
from hrm_x.library import Library, abstract_all, expand, grow
from hrm_x.mind import Controller, Mind

TAGS = ("add", "iter")
VARS = ("x", "y", "a")
STEP = ("add", ("a",), X)
EXAMPLES = [(Fraction(2), Fraction(3), Fraction(6)),
            (Fraction(3), Fraction(2), Fraction(6)),
            (Fraction(4), Fraction(3), Fraction(12))]


@pytest.fixture(scope="module")
def mined():
    corpus = [("iter", Y, STEP, ZERO), ("iter", X, STEP, ZERO),
              ("iter", ("add", X, Y), STEP, ZERO), ("iter", ("add", X, ONE), STEP, ZERO),
              ("iter", Y, STEP, ONE), ("iter", Y, STEP, ("c", Fraction(2)))]
    return grow(corpus, tag="g", arity_vars=VARS, tags=TAGS)


def solve_ops(arith: Arith) -> tuple:
    mind = Mind(control=Controller(max_cycles=8))
    task = ArithTask("×", list(EXAMPLES), "10 × 3 + 1", oracle=lambda a, b: a * b)
    result = mind.solve(arith, task)
    return (result.status, sum(s.ops for s in result.trace if s.action == "propose"),
            any(h.verified for h in result.hypotheses), result.answer)


def test_the_extended_language_reaches_the_compressed_program_one_node_earlier(mined):
    """The capability the library exists for, stated positively: a program too large for the base
    language at bound n is expressible at n - credit once the step is named."""
    lib, _ = mined
    assert lib.credit == 1.0, "one step per program, one node saved"
    small = ("iter", Y, ("prim", "g1", ("a",)), ZERO)
    assert len(collect(small, programs_with(4, lib.prims))) == 0, "not in the base language at 4"
    assert small in collect(small, programs_with(5, lib.prims)), "reachable once named"


def collect(target, stream):
    return [p for p, _ in stream if p == target]


def test_a_present_library_leaves_the_default_search_byte_identical(mined):
    lib, _ = mined
    plain = Arith(max_size=(6, 7))
    with_lib = Arith(max_size=(6, 7), library=Library())
    with_lib.library = lib
    assert with_lib.bound(0) == plain.bound(0) == 6
    assert solve_ops(with_lib) == solve_ops(plain), "a present library must not change the search"


def test_the_extended_enumeration_is_larger_at_the_same_bound(mined):
    """Why the default has to be off: this is the measurement, pinned so it cannot be forgotten."""
    lib, _ = mined
    assert len(list(programs_with(6, lib.prims))) > len(list(programs(6)))


def test_reducing_the_bound_is_opt_in_and_only_ever_by_the_measured_credit(mined):
    lib, _ = mined
    off = Arith(max_size=(6, 7), library=Library())
    on = Arith(max_size=(6, 7), library=Library(), use_library_bound=True)
    off.library = on.library = lib
    assert off.bound(0) == 6
    assert on.bound(0) == 6 - int(round(lib.credit)) == 5


def test_the_reduced_bound_still_finds_the_concept_and_can_cost_more(mined):
    """The honest comparison, kept as a test so the negative result is not quietly deleted."""
    lib, _ = mined
    plain = Arith(max_size=(6, 7))
    applied = Arith(max_size=(6, 7), library=Library(), use_library_bound=True)
    applied.library = lib
    status_a, ops_a, ok_a, ans_a = solve_ops(plain)
    status_b, ops_b, ok_b, ans_b = solve_ops(applied)
    assert (status_a, ok_a, ans_a) == (status_b, ok_b, ans_b) == ("answer", True, Fraction(31))
    assert ops_b > ops_a, "on this corpus the reduced bound is more expensive, and that is the finding"


def test_the_trade_reverses_when_the_repeated_structure_is_a_large_share(mined):
    """The regime the feature is for. Six nested steps save 11 nodes, which buys far more bound than
    the new constructor costs."""
    six = ("add", ("add", ("add", ("add", ("add", ("add", ("a",), X), X), X), X), X), X)
    dense = [("add", six, ("iter", Y, ONE, ZERO)), ("add", six, ("iter", X, ONE, ZERO))]
    lib, log = grow(dense, tag="h", arity_vars=VARS, tags=TAGS)
    assert lib.credit >= 10, f"expected a large saving, got {lib.credit}"
    assert lib.prims[0].saving > 1
    for bound in (6, 7, 8):
        base = len(list(programs_with(bound)))
        reduced = len(list(programs_with(max(2, bound - int(round(lib.credit))), lib.prims)))
        assert reduced < base, f"at bound {bound}: {reduced} should beat {base}"


def test_the_credit_is_measured_on_the_corpus_not_derived_from_primitive_sizes(mined):
    lib, _ = mined
    assert lib.prims[0].saving == 1
    assert lib.credit == 1.0, "one step per program, so one node per program"


def test_mine_uses_the_surviving_version_space_and_not_just_the_winner(mined):
    """`Arith.mine` mines every hypothesis still standing for every concept, because a structure
    worth naming recurs across concepts and the losers are where the evidence is."""
    arith = Arith(max_size=(4, 5))
    mind = Mind()
    # Six surviving hypotheses, not one: a step seen three times does not pay for its declaration
    # (3*1 - 4 < 0), so the test has to use a version space big enough for the claim to hold.
    arith._memory(mind)["concepts"]["×"] = {
        "hypotheses": [(("iter", Y, STEP, ZERO), 0.30), (("iter", X, STEP, ZERO), 0.20),
                       (("iter", ("add", X, Y), STEP, ZERO), 0.15),
                       (("iter", ("add", X, ONE), STEP, ZERO), 0.15),
                       (("iter", Y, STEP, ONE), 0.10), (("iter", Y, STEP, ("c", Fraction(2))), 0.10)],
        "evidence": []}
    log = arith.mine(mind)
    assert log, "mining an empty memory is a no-op, mining a real one is not"
    assert arith.library.prims[0].body == STEP
    assert arith.mine(Mind()) == [], "and nothing to mine from nothing"


def test_a_library_never_reaches_the_evaluator_through_a_hypothesis(mined):
    """`propose` stores the expanded program, so laws, the verifier, the notation rule and semantic
    memory all keep working on the base language."""
    lib, _ = mined
    arith = Arith(max_size=(6, 7), library=Library(), use_library_bound=True)
    arith.library = lib
    mind = Mind(control=Controller(max_cycles=8))
    task = ArithTask("×", list(EXAMPLES), "10 × 3 + 1", oracle=lambda a, b: a * b)
    result = mind.solve(arith, task)
    for hyp in result.hypotheses:
        assert expand(hyp.payload, lib.prims) == hyp.payload, "payload must already be base-language"
        assert not (isinstance(hyp.payload, tuple) and hyp.payload
                    and hyp.payload[0] == "prim")
    arith.sleep(mind, lambda a, b: a * b, budget=1)     # laws must still consolidate
