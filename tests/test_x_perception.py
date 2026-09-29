"""What the held-out free-text gate actually found, pinned so it cannot be quietly forgotten.

Two things are being protected here, and they are not the same thing:

* The measurement itself. `experiments/x_perception.py` scores the parser against a hand-written
  reference reader, so the reference has to be correct and the held-out set has to be genuinely
  unseen. A bug in either would have produced a clean-looking result (it did, twice, during
  development) and these tests make it loud.
* The verifier's blind spot. `Story.verify` replays the parser's own reading, so a *complete but
  wrong* reading is verified. That is not a bug to be fixed here -- an independent reader is an
  architecture decision (ADR-009) -- but it is the reason the reasoner cannot be credited with the
  safe failure the story domain shows, and it needs a test or it will be rediscovered as a surprise.
"""
from fractions import Fraction

import pytest

from experiments.x_perception import (HELDOUT_ITEMS, HELDOUT_NAMES, REFERENCE, REFERENCE_SEEN,
                                      key, probes, reference_frame)
from hrm_x.domains.story import (Frame, Reading, Story, StoryTask, TEMPLATES, reading_defect)
from hrm_x.mind import UNKNOWN, Hyp, Mind

ALICE, BOB = "alice", "bob"
STATE = Frame("STATE", (ALICE,), None, None, Fraction(10), "apple")
CONSUME = Frame("CONSUME", (ALICE,), None, None, Fraction(3), "apple")
ACQUIRE = Frame("ACQUIRE", (ALICE,), None, None, Fraction(3), "apple")
ASK = Frame("Q_ONE", (ALICE,), None, None, None, "apple")


# ---------- the reference reader is trustworthy ----------

def test_the_reference_reader_agrees_with_every_hand_written_expectation():
    """Otherwise the experiment is scoring against a wrong answer key."""
    for group, sentence, changed, want in probes():
        got = reference_frame(sentence)
        assert got == want, f"{group}: {sentence!r} ({changed}) -> {got}, expected {want}"


def test_every_held_out_probe_is_genuinely_unseen():
    """The reader restricted to the trained-on grammar must fail on every held-out probe, which is
    what makes "held out" a property of the data rather than of the filename."""
    for group, sentence, _changed, _want in probes():
        if group.startswith("control"):
            assert reference_frame(sentence, seen=True) is not None, sentence
        else:
            assert reference_frame(sentence, seen=True) is None, f"{group}: {sentence!r} is seen"


def test_no_held_out_surface_form_appears_in_the_templates():
    """A second, independent guarantee: the held-out wording is not a TEMPLATES rewording."""
    template_words = {t for _, t in TEMPLATES}
    verbs = {w for pattern, schema, _ in REFERENCE for w in ("hands", "passes", "transfers", "gets",
                                                             "accepts", "collects", "gathers",
                                                             "gobbles", "holds")}
    for verb in verbs:
        assert not any(verb in t.split() for t in template_words), f"{verb} is in TEMPLATES"
    assert not any("altogether" in t or "in total" in t for t in template_words)


def test_the_reader_survives_the_spacing_and_capitalisation_the_domain_actually_uses():
    """It did not, during development, and the resulting silent zero looked like a perception
    success. `TEMPLATES` put a space before the final ".", and questions are capitalised."""
    assert reference_frame("How many oranges does Xavier have ?") is not None
    assert reference_frame("Zoltan hands 3 oranges to Priya .") is not None
    assert reference_frame("Priya is given 3 oranges by Zoltan .") is not None
    assert reference_frame("Zoltan has 3 oranges.") is not None, "no space before the period too"


def test_the_reader_covers_every_sentence_the_generator_can_produce():
    """`end_to_end` raises on an uncovered sentence; this makes that raise unreachable by accident."""
    from experiments.x_perception import held_out_story
    import random
    checked = 0
    for seed in range(60):
        rng = random.Random(seed)
        for _ in range(6):
            sentences, q, _truth, ok = held_out_story(rng, held_state=True, held_ask=True)
            if not ok:
                continue
            for sentence in sentences + [q]:
                assert reference_frame(sentence) is not None, sentence
                checked += 1
    assert checked > 100, f"only {checked} sentences checked"


def test_the_reference_reader_is_role_based_not_name_based():
    """It has to handle names it has never seen, or it would be memorising a list like the parser."""
    a, b = HELDOUT_NAMES[0], HELDOUT_NAMES[1]
    item = HELDOUT_ITEMS[0]
    frame = reference_frame(f"{a} hands 3 {item} to {b} .")
    assert frame is not None and frame.src == a.lower() and frame.dst == b.lower()
    assert frame.item == item[:-1], "the item is read by position and singularised, not listed"


# ---------- reading_defect is complete per schema ----------

@pytest.mark.parametrize("frame,expect", [
    (Frame("TRANSFER", (), "alice", "bob", Fraction(3), None), "no item"),
    (Frame("TRANSFER", (), "alice", "bob", None, "apple"), "without an amount"),
    (Frame("TRANSFER", (), "alice", "bob", Fraction(3), "apple"), None),
    (Frame("TRANSFER", (), None, "bob", Fraction(3), "apple"), "both parties"),
    (Frame("TRANSFER", (), "alice", None, Fraction(3), "apple"), "both parties"),
    (Frame("TRANSFER", (), None, None, Fraction(3), "apple"), "both parties"),
    (Frame("STATE", (ALICE,), None, None, None, "apple"), "no amount"),
    (Frame("STATE", (), None, None, Fraction(3), "apple"), "without an owner"),
    (Frame("ACQUIRE", (ALICE,), None, None, None, "apple"), "no amount"),
    (Frame("CONSUME", (ALICE,), None, None, None, "apple"), "no amount"),
    (Frame("Q_ONE", (), None, None, None, "apple"), "question without an owner"),
    (Frame("Q_SUM", (), None, None, None, "apple"), "question without an owner"),
    (Frame("Q_ONE", (ALICE,), None, None, None, None), "no item"),
])
def test_every_schema_reports_its_own_missing_slot(frame, expect):
    """This guard was found incomplete twice -- a missing amount reached `Fraction(None)`, then a
    question with no owner reached `owner[0]` -- so each schema is pinned slot by slot."""
    defect = reading_defect(Reading((STATE,), frame))
    if expect is None:
        assert defect is None
    else:
        assert defect is not None and expect in defect, f"{frame.schema}: {defect!r}"


def test_a_question_read_with_no_owner_never_reaches_the_workspace():
    """The crash the held-out split produced. "Alice has how many apples ?" is read as an ask with
    nobody as owner, and both `derive` (which indexes `owner[0]`) and `simulate` (which would sum
    over an empty tuple and answer 0) are wrong on it."""
    broken = Reading((STATE,), Frame("Q_ONE", (), None, None, None, "apple"))
    dom = Story.__new__(Story)                       # no parser needed: the frames are given
    hyp = Hyp("broken", 1.0, broken)
    dom.derive(StoryTask(["Alice has 10 apples."], "Alice has how many apples ?"), hyp, Mind())
    assert hyp.answer == UNKNOWN, hyp.answer
    assert "unknown" in (hyp.note or "")


def test_simulate_also_refuses_a_question_with_no_owner():
    """Without the guard `simulate` returns 0, which is a confident wrong answer rather than a crash."""
    from hrm_x.domains.story import simulate
    answer, problems = simulate(Reading((STATE,), Frame("Q_ONE", (), None, None, None, "apple")))
    assert answer is None and problems and "owner" in problems[0]


# ---------- the verifier cannot see a misreading ----------

def test_verification_confirms_a_complete_but_wrong_reading():
    """"Alice eats 3 apples" read as an acquire: internally consistent, derivable, and wrong.

    This is the case no structural check can catch -- every slot the frame needs is present, and it
    is the *parser* that got it wrong. `Story.verify` replays the same reading through a different
    representation, so it confirms it. The text says 7, the mind answers 13 and is confident.
    Pinned deliberately: this is why the safe failure rate in x_perception is a property of
    `reading_defect`, not of verification, and why free text needs an independent reader.
    """
    task = StoryTask(["Alice has 10 apples.", "Alice eats 3 apples."],
                     "How many apples does Alice have?")
    assert int(ACQUIRE.amount) == int(CONSUME.amount), "same magnitude, opposite event"
    dom = Story.__new__(Story)
    hyp = Hyp("acquired instead of consumed", 1.0, Reading((STATE, ACQUIRE), ASK))
    dom.derive(task, hyp, Mind())
    assert hyp.answer == Fraction(13), hyp.answer
    dom.verify(task, hyp, Mind())
    assert hyp.verified, "verification confirms it: it is replaying the reading it was given"
    assert hyp.answer != 7, "and the text says 7"


def test_the_same_derivation_is_correct_when_the_reading_is_right():
    """The control for the test above, so the pin is about the reading and not about the arithmetic."""
    task = StoryTask(["Alice has 10 apples.", "Alice eats 3 apples."],
                     "How many apples does Alice have?")
    dom = Story.__new__(Story)
    hyp = Hyp("consumed", 1.0, Reading((STATE, CONSUME), ASK))
    dom.derive(task, hyp, Mind())
    dom.verify(task, hyp, Mind())
    assert hyp.answer == Fraction(7) and hyp.verified
