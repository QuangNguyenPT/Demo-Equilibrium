"""HRM-X metacognition: version-space memory, curiosity, volatility (NE-like), sleep, writer, persistence."""
from fractions import Fraction

from hrm_x.domains.arith import Arith, ArithTask
from hrm_x.induction import Law
from hrm_x.mind import Mind, Modulators
from hrm_x.writer import explain

TIMES = lambda a, b: a * b


def ambiguous_mind():
    """Evidence 2x3=6, 3x5=15 leaves two readings of x (count by x, count by y) that agree on naturals."""
    mind, dom = Mind(), Arith()
    r = mind.solve(dom, ArithTask.from_text(["2 × 3 = 6", "3 × 5 = 15"], "4 × 2", oracle=TIMES))
    return mind, dom, r


def test_agreeing_hypotheses_are_derived_together_and_all_survivors_are_remembered():
    mind, dom, r = ambiguous_mind()
    assert r.answer == 8                                   # the derive-group action found the agreement
    assert len(mind.knowledge["arith"]["concepts"]["×"]["hypotheses"]) >= 2
    assert dom.ambiguity(mind, "×") > 0


def test_curiosity_resolves_the_concept_with_fewer_questions_than_random_study():
    curious_mind, dom, _ = ambiguous_mind()
    asked_curious = dom.study(curious_mind, "×", TIMES, budget=20, strategy="curious")
    assert dom.ambiguity(curious_mind, "×") == 0
    random_asks = []
    for seed in range(5):
        m, d, _ = ambiguous_mind()
        random_asks.append(d.study(m, "×", TIMES, budget=20, strategy="random", seed=seed))
    assert asked_curious < sum(random_asks) / len(random_asks)
    after = curious_mind.solve(dom, ArithTask.from_text([], "2.5 × 4"))       # fractional count, no teacher
    assert after.answer == 10 or after.status == "abstain"                  # never the wrong reading


def test_volatility_makes_reliability_track_a_changed_world_faster():
    def failures_until_distrust(mods: Modulators) -> int:
        mind = Mind(modulators=mods)
        for _ in range(100):
            mind.feedback("arith", True)
        n = 0
        while mind.rate("arith:verify-pass") > 0.8:
            mind.feedback("arith", False)
            n += 1
        return n
    assert failures_until_distrust(Modulators()) < failures_until_distrust(Modulators(enabled=False))


def test_sleep_studies_consolidates_and_retracts_a_false_law():
    mind, dom, _ = ambiguous_mind()
    report = mind.sleep(dom, teacher=TIMES, budget=5)
    assert report["asked"]["×"] >= 1 and report["laws"]["×"] > 0
    key = ("×", f"concept:{__import__('hrm_x.domains.arith', fromlist=['show']).show(dom.best(mind, '×'))}")
    mind.knowledge["arith"]["laws"][key].append(Law(("f", "?x", "?y"), ("add", "?x", "?y"), 0))
    again = mind.sleep(dom)
    assert "f(x, y) = add(x, y)" in again["retracted"]


def test_writer_reports_what_was_actually_done():
    mind, dom = Mind(), Arith()
    r = mind.solve(dom, ArithTask.from_text(["2 × 2 = 4", "2 × 3 = 6"], "10 × 3.1", oracle=TIMES))
    text = explain(r, "10 × 3.1")
    assert "-> 31" in text and "I asked" in text and "Checked independently: yes" in text


def test_saved_mind_uses_its_knowledge_after_reload(tmp_path):
    mind, dom = Mind(), Arith()
    mind.solve(dom, ArithTask.from_text(["2 × 2 = 4", "2 × 3 = 6"], "10 × 3.1", oracle=TIMES))
    mind.save(str(tmp_path / "mind.pt"))
    again = Mind.load(str(tmp_path / "mind.pt"))
    r = again.solve(Arith(), ArithTask.from_text([], "12 × 0.25"))
    assert r.answer == Fraction(3) and [s.action for s in r.trace][0] == "propose"
