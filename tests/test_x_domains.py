"""HRM-X domain behaviour: the capability claims of spec v0.4, each as a falsifiable check."""
from fractions import Fraction

import numpy as np
import pytest

from hrm_x.domains.arith import Arith, ArithTask, Counter, parse, programs, run
from hrm_x.domains.grid import GridTask, Grids, objects
from hrm_x.domains.story import Parser, Story, StoryTask
from hrm_x.induction import Law
from hrm_x.mind import Controller, Mind

TIMES = lambda a, b: a * b


def learned_mind():
    mind, dom = Mind(), Arith()
    result = mind.solve(dom, ArithTask.from_text(["2 × 2 = 4", "2 × 3 = 6"], "10 × 3.1", oracle=TIMES))
    return mind, dom, result


def plant_false_law(mind):
    """f(x, y) = add(x, y) filed under every surviving reading of × (laws are kept per reading)."""
    from hrm_x.domains.arith import show
    law = Law(("f", "?x", "?y"), ("add", "?x", "?y"), 0)
    for prog, _ in mind.knowledge["arith"]["concepts"]["×"]["hypotheses"]:
        mind.knowledge["arith"]["laws"][("×", f"concept:{show(prog)}")] = [law]


def test_two_examples_do_not_identify_multiplication_but_one_question_does():
    """2x2=4, 2x3=6 also fit y+y (simpler). Without a teacher the mind answers from the simpler
    reading; with one it asks the most informative question and gets 31 exactly."""
    alone = Mind().solve(Arith(), ArithTask.from_text(["2 × 2 = 4", "2 × 3 = 6"], "10 × 3.1"))
    assert alone.answer == Fraction(31, 5)
    mind, dom, result = learned_mind()
    assert result.answer == 31 and result.confidence == 1.0
    assert [s.action for s in result.trace].count("ask") == 1


def test_learned_concept_extrapolates_exactly_far_outside_the_evidence():
    mind, dom, _ = learned_mind()
    for q, truth in [("4812 × 37.25", Fraction(4812) * Fraction("37.25")), ("7 × 0.125", Fraction(7, 8))]:
        r = mind.solve(dom, ArithTask.from_text([], q))
        assert r.answer == truth and r.hypotheses[0].verified


def test_laws_turn_a_long_unfolding_into_a_short_derivation():
    mind, dom, _ = learned_mind()
    before = mind.solve(dom, ArithTask.from_text([], "1000 × 3.1"))
    laws, _ = dom.consolidate(mind, "×", max_size=7, probes=16)
    names = {str(l) for l in laws}
    assert "f(x, y) = f(y, x)" in names and "f(x, f(y, z)) = f(f(x, y), z)" in names
    assert "f(x, add(y, z)) = add(f(x, y), f(x, z))" in names
    after = mind.solve(dom, ArithTask.from_text([], "1000 × 3.1"))
    assert before.answer == after.answer == 3100
    assert after.ops < before.ops / 5


def test_residue_verifier_catches_a_false_law_and_blames_it():
    mind, dom, _ = learned_mind()
    plant_false_law(mind)
    r = mind.solve(dom, ArithTask.from_text([], "3 × 4"))
    bad = [h for h in r.hypotheses if h.verified is False]
    assert r.status == "abstain" and bad and any("f(x, y) = add(x, y)" in c for c in bad[0].culprits)


def test_program_language_and_parser():
    assert parse("2 × 3 + 4", "×") == ("add", ("op", ("num", 2), ("num", 3)), ("num", 4))
    mul = ("iter", ("x",), ("add", ("a",), ("y",)), ("c", Fraction(0)))
    assert run(mul, Fraction(3), Fraction("2.5"), Counter()) == Fraction("7.5")
    assert run(mul, Fraction("2.5"), Fraction(3), Counter()) is None      # a count must be natural
    assert mul in {p for p, _ in programs(6)}


@pytest.fixture(scope="module")
def story():
    parser = Parser()
    parser.train()
    return Story(parser)


def test_story_state_changes_are_derived(story):
    r = Mind().solve(story, StoryTask(["Alice has 4 apples.", "Bob has 2 apples.", "Alice gives Bob 3 apples.",
                                       "Bob buys 5 apples."], "How many apples does Bob have?"))
    assert r.answer == 10 and r.hypotheses[0].verified


def test_story_generalizes_to_unseen_names_items_and_magnitudes(story):
    r = Mind().solve(story, StoryTask(["Zoltan has 400 oranges.", "Quenby has 250 oranges.",
                                       "Quenby receives 120 oranges from Zoltan."],
                                      "How many oranges do Zoltan and Quenby have together?"))
    assert r.answer == 650


def test_role_swap_minimal_pair_changes_the_answer_for_the_right_reason(story):
    base = ["Alice has 5 apples.", "Bob has 5 apples."]
    to_bob = Mind().solve(story, StoryTask(base + ["Alice gives Bob 2 apples."], "How many apples does Bob have?"))
    to_alice = Mind().solve(story, StoryTask(base + ["Bob gives Alice 2 apples."], "How many apples does Bob have?"))
    assert (to_bob.answer, to_alice.answer) == (7, 3)


def test_contradiction_is_refused_with_the_responsible_sentences(story):
    r = Mind().solve(story, StoryTask(["Alice has 2 apples.", "Bob has 1 apples.", "Carol has 5 apples.",
                                       "Alice gives Bob 3 apples."], "How many apples does Bob have?"))
    assert r.status == "abstain" and "s0" in r.reason and "s3" in r.reason and "s2" not in r.reason


def test_missing_information_gives_i_dont_know(story):
    r = Mind().solve(story, StoryTask(["Alice has 2 apples.", "Alice gives Bob 1 apples."],
                                      "How many apples does Bob have?"))
    assert r.status == "abstain" and "unknown" in r.reason


def _scene(h, w, rects):
    g = np.zeros((h, w), int)
    for r, c, hh, ww, col in rects:
        g[r:r + hh, c:c + ww] = col
    return g


def test_objects_are_connected_components():
    g = _scene(5, 5, [(0, 0, 2, 2, 1), (3, 3, 1, 2, 1), (0, 4, 1, 1, 2)])
    assert sorted(o.size for o in objects(g)) == [1, 2, 4]


def test_grid_program_transfers_to_a_new_arrangement():
    demos = []
    for rects, big in [([(0, 0, 2, 2, 1), (4, 4, 3, 3, 2), (7, 0, 1, 2, 3)], 1),
                       ([(1, 1, 1, 3, 4), (5, 2, 2, 2, 1), (0, 6, 4, 2, 2)], 2),
                       ([(0, 0, 3, 3, 3), (6, 6, 2, 2, 1)], 0)]:
        x = _scene(9, 9, rects)
        y = x.copy()
        r, c, hh, ww, _ = rects[big]
        y[r:r + hh, c:c + ww] = 5
        demos.append((x, y))
    test = _scene(11, 11, [(0, 5, 2, 4, 2), (4, 0, 1, 1, 1), (6, 3, 3, 2, 4), (9, 9, 1, 1, 6)])
    truth = test.copy()
    truth[0:2, 5:9] = 5
    r = Mind().solve(Grids(), GridTask(demos, test))
    assert r.answer == tuple(map(tuple, truth))


def test_ambiguous_selector_on_the_test_input_is_not_verified():
    demos = [(_scene(6, 6, [(0, 0, 2, 2, 1), (4, 4, 1, 1, 2)]), None)]
    x, _ = demos[0]
    y = x.copy()
    y[0:2, 0:2] = 5
    tie = _scene(6, 6, [(0, 0, 2, 2, 1), (3, 3, 2, 2, 2)])           # two largest objects
    r = Mind().solve(Grids(), GridTask([(x, y)], tie))
    assert all(h.verified is not True for h in r.hypotheses if "largest" in h.key)


def test_easy_task_costs_less_than_hard_task():
    mind, dom, hard = learned_mind()
    easy = mind.solve(dom, ArithTask.from_text([], "6 × 7"))
    assert easy.answer == 42 and easy.cycles < hard.cycles and easy.ops < hard.ops / 10


def test_disabling_the_verifier_lets_a_false_law_through():
    mind, dom, _ = learned_mind()
    plant_false_law(mind)
    mind.control = Controller(use_verifier=False)
    r = mind.solve(dom, ArithTask.from_text([], "3 × 4"))
    assert r.status == "answer" and r.answer == 7                  # wrong, and confidently so
