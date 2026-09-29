"""The library must be a naming device and nothing else (tests for hrm_x/library.py).

The properties asserted here are the ones a primitive could plausibly violate: that abstraction is
reversible, that it does not change what a program computes, that a primitive actually pays for
itself, and that mining the encoded corpus finds structure mining the raw one cannot.
"""
from fractions import Fraction

import pytest

from hrm_x.library import (Library, Primitive, abstract_all, abstract_once, apply, arg_slots,
                           candidates, expand, grow, leaves, occurrences, show, size, tags_of)
from hrm_x.domains.arith import ONE, X, Y, Counter, Exhausted, programs, run

ARITH_VARS = ("x", "y", "a")
ARITH_TAGS = ("add", "iter")
STEP = ("add", ("a",), X)                  # the loop step of repeated addition
D = ("c", Fraction(0))                     # the initial value
TWO = ("c", Fraction(2))


def multiply_suite() -> list:
    """Distinct programs that all contain the same step, the way a real corpus of known concepts
    would. Duplicated programs would inflate the count without adding evidence, so these differ."""
    return [
        ("iter", Y, STEP, D),                       # x * y
        ("iter", X, STEP, D),                       # y * x
        ("iter", ("add", X, Y), STEP, D),           # x * (x + y)
        ("iter", Y, STEP, ONE),                      # offset by one
        ("iter", Y, STEP, TWO),                      # offset by two
        ("iter", Y, ("add", ("a",), X), ONE),        # same step, different start
    ]


def test_size_leaves_and_occurrences_describe_the_language():
    assert size(X) == 1 and size(("c", Fraction(3))) == 1
    assert size(("add", X, Y)) == 3
    assert leaves(X) == frozenset({"x"})
    assert leaves(("c", Fraction(3))) == frozenset(), "a constant is not a variable"
    assert leaves(("add", X, Y)) == frozenset({"x", "y"})
    assert leaves(("iter", Y, STEP, ONE)) == frozenset({"a", "x", "y"})
    p = ("add", STEP, STEP)
    assert occurrences(p)[STEP] == 2 and occurrences(p)[p] == 1


def test_arg_slots_requires_each_hole_to_occur_exactly_once():
    assert arg_slots(STEP, ("a",)) == (0,)
    assert arg_slots(STEP, ("a", "x")) == (0, 1)
    assert arg_slots(("add", X, X), ("x",)) is None, "a repeated variable is not one argument"
    assert arg_slots(("add", X, Y), ("a",)) is None


def test_binding_the_accumulator_is_what_makes_a_step_worth_naming():
    """The accounting that the first version of this file got wrong. `add(a, x)` applied as
    `step(a, x)` saves nothing: the application plus two one-node arguments is exactly the body.
    Capturing x and binding only a gives a body of 3 and an application of 2, so one node per use."""
    both = Primitive("t", STEP, ("a", "x"))
    only_acc = Primitive("t", STEP, ("a",))
    assert both.saving == 0, "a name with trivial arguments is not a saving"
    assert only_acc.saving == 1, "capturing x is what pays"
    assert only_acc.gain(8) > 0 and both.gain(8) < 0


def test_a_primitive_is_a_name_and_expansion_is_exact():
    p = ("iter", Y, STEP, D)
    step = Primitive("g1", STEP, ("a",))
    named = abstract_once(p, step)
    assert named is not None and named[0] == "iter"
    assert expand(named, [step]) == p


def test_expansion_works_when_the_root_itself_is_a_primitive():
    step = Primitive("g1", STEP, ("a",))
    assert expand(apply(step, [("a",)]), [step]) == STEP, "the hole is filled with its own value"
    assert expand(apply(step, [ONE]), [step]) == ("add", ONE, X), "or with whatever it was given"


def test_abstraction_does_not_change_what_the_program_computes():
    """Checked by evaluation, not only by shape. The corpus includes a nested occurrence, since a
    step inside a step is the case a name has to get right and a flat case cannot show."""
    lib, _ = grow(multiply_suite(), tag="g")
    assert len(lib) >= 1
    corpus = multiply_suite() + [
        ("iter", Y, ("add", STEP, X), D),              # step nested one level deeper
        ("add", ("iter", Y, STEP, D), ("iter", Y, STEP, D)),
        ("iter", TWO, STEP, ("iter", Y, STEP, D)),      # two occurrences in one program
    ]
    probes = [Fraction(0), Fraction(1), Fraction(3), Fraction(7, 2)]
    checked = 0
    for prog in corpus:
        named = abstract_all(prog, lib.prims)
        assert named != prog, "every program in this corpus contains the step"
        assert expand(named, lib.prims) == prog
        for x in probes[:2]:
            for y in probes[:2]:
                try:
                    before = run(prog, x, y, Counter(4000))
                except Exhausted:
                    continue
                assert run(expand(named, lib.prims), x, y, Counter(4000)) == before, \
                    f"{show(prog)} changed value when named"
                checked += 1
    assert checked > 0, "the round-trip check never ran"


def test_a_primitive_that_cannot_pay_for_itself_is_rejected():
    lib, log = grow([("add", X, Y)], tag="g")
    assert len(lib) == 0, "a library that grows for free is exactly the complexity to avoid"
    assert log == []


def test_repetition_pays_and_the_step_is_learned():
    lib, log = grow(multiply_suite(), tag="g", tags=ARITH_TAGS)
    assert len(lib) >= 1
    prim = lib.prims[0]
    assert prim.body == STEP and prim.holes == ("a",), "the loop step is the reusable part"
    assert prim.saving == 1
    entry = log[0]
    assert entry["gain"] > 0
    assert entry["mdl_after"] < entry["mdl_before"], "a primitive must reduce description length"
    assert entry["uses"] >= entry["occurrences"]


def test_growing_shortens_the_programs_it_saw():
    """The honest comparison: total description length, library included, against no library at all.
    Comparing the library against itself in the *raw* language would show only the declaration cost,
    which is the cost and not the benefit."""
    corpus = multiply_suite()
    lib, _ = grow(corpus, tag="g")
    encoded = [abstract_all(p, lib.prims) for p in corpus]
    assert lib.mdl(encoded) < Library().mdl(corpus), "the library must pay for itself"
    assert lib.size_of(encoded[0]) < lib.size_of(corpus[0])
    assert lib.prims[0].uses == len(corpus)
    for original, named in zip(corpus, encoded):
        assert expand(named, lib.prims) == original


def test_min_count_suppresses_structure_seen_once():
    lib, _ = grow(multiply_suite(), tag="g", tags=ARITH_TAGS, min_count=99)
    assert len(lib) == 0


def test_min_gain_suppresses_a_primitive_that_only_just_pays():
    lib, _ = grow(multiply_suite(), tag="g", tags=ARITH_TAGS, min_gain=10 ** 6)
    assert len(lib) == 0


def test_later_passes_mine_the_encoded_corpus_not_the_raw_one():
    """After `step` is named, the residual structure is a different candidate. Mining the raw corpus
    every pass would keep re-proposing the same abstraction and never see anything else."""
    corpus = multiply_suite()
    one, log1 = grow(corpus, tag="g", tags=ARITH_TAGS, passes=1)
    many, log2 = grow(corpus, tag="g", tags=ARITH_TAGS, passes=4)
    assert len(log1) == 1
    assert len(log2) >= len(log1), "a longer run should not do less"
    for entry in log2:
        assert entry["mdl_after"] <= entry["mdl_before"], "every step must pay for itself"
    assert many.mdl(corpus) <= one.mdl(corpus)


def test_names_are_derived_from_the_library_not_a_counter():
    lib = Library()
    first = lib.add(Primitive(name=lib.name("g"), body=("add", X, Y), holes=("x",)))
    second = lib.add(Primitive(name=lib.name("g"), body=("add", Y, X), holes=("y",)))
    assert first.name != second.name
    assert lib.name("g") not in lib, "a fresh name must not collide"


def test_expanding_an_unknown_primitive_is_an_error_not_a_silent_wrong_answer():
    prim = Primitive("g1", ("add", X, Y), ("x",))
    p = apply(prim, [Y])
    with pytest.raises(KeyError):
        expand(p, [])
    with pytest.raises(ValueError):
        apply(prim, [Y, Y])


def test_candidates_skip_subtrees_that_mention_an_unbindable_variable():
    """A subtree mentioning a symbol outside the arity set cannot be given a hole, so it must not
    be proposed; otherwise expansion could never fill it in."""
    weird = ("add", X, ("weird", Y))
    assert candidates([weird] * 4, ARITH_VARS, ARITH_TAGS) == []
    assert tags_of(weird) == {"add", "x", "weird", "y"}
