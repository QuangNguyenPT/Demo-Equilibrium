"""Metacognition (self-model, curiosity, intent) and the self-study loop."""
import numpy as np
import pytest

import hrm.learning.sleep as sleep_module
from hrm.config import (LearningConfig, MetaConfig, ModelConfig, PerceptionConfig,
                        ReasoningConfig, SubstrateConfig)
from hrm.learning import GridTasks, Harness, HarnessLockError, SequenceTasks, StudyBudget
from hrm.metacognition import Curiosity, IntentHead, Message, SelfModel
from hrm.model import Model, ModelError

META = MetaConfig(min_attempts=2, patience=3)


def config(**learning) -> ModelConfig:
    return ModelConfig(substrate=SubstrateConfig(dim=512),
                       perception=PerceptionConfig(grid_max_shift=2),
                       reasoning=ReasoningConfig(max_ticks=150, restart_budget=2),
                       learning=LearningConfig(**learning), meta=META)


class Offsets(SequenceTasks):
    families = ("offset", "reverse")


class Composites(SequenceTasks):
    families = ("scale+reverse",)


class Gravity(GridTasks):
    families = ("gravity",)

    def __init__(self):
        super().__init__(size=6, max_shift=2)


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    path = tmp_path_factory.mktemp("bench") / "benchmark.json"
    return Harness.build(path, [Offsets()], per_family=1, decoys_per_domain=1, seed=3)


class TestSelfModel:
    def test_unexplored_learning_mastered(self):
        sm = SelfModel(META)
        assert sm.status("r") == "unexplored"
        sm.observe("r", "correct")
        sm.observe("r", "correct")
        assert sm.status("r") == "learning"
        for _ in range(8):
            sm.observe("r", "correct")
        assert sm.status("r") == "mastered"

    def test_no_progress_becomes_stuck(self):
        sm = SelfModel(META)
        for _ in range(META.patience):
            sm.observe("r", "abstained")
        assert sm.status("r") == "stuck"

    def test_round_trip_and_validation(self):
        sm = SelfModel(META)
        sm.observe("a", "wrong")
        other = SelfModel(META)
        other.load_dict(sm.to_dict())
        assert other.get("a").wrong == 1 and other.report()[0]["region"] == "a"
        with pytest.raises(ValueError):
            sm.observe("a", "maybe")


class TestCuriosity:
    def test_prefers_unexplored_skips_mastered_and_stuck(self):
        sm = SelfModel(META)
        for _ in range(10):
            sm.observe("mastered", "correct")
        for _ in range(META.patience):
            sm.observe("stuck", "abstained")
        cur = Curiosity(MetaConfig(explore=0.0), np.random.default_rng(0))
        assert cur.choose(sm, ["mastered", "stuck", "new"]) == "new"
        assert cur.choose(sm, ["mastered", "stuck"]) is None

    def test_learning_region_beats_flat_gap(self):
        sm = SelfModel(META)
        for outcome in ("abstained", "abstained", "correct", "correct"):
            sm.observe("improving", outcome)
        for _ in range(2):
            sm.observe("flat", "wrong")
        assert sm.status("improving") == "learning" and sm.status("flat") == "gap"
        cur = Curiosity(MetaConfig(explore=0.0), np.random.default_rng(0))
        assert cur.choose(sm, ["flat", "improving"]) == "improving"


class TestIntent:
    DEMOS = [([1, 2], [2, 3])]

    @pytest.mark.parametrize("msg,intent", [
        (Message(demos=DEMOS, query=[5]), "answer"),
        (Message(demos=DEMOS), "learn"),
        (Message(demos=DEMOS, query=[5], expected=[6]), "learn"),
        (Message(refers_to="t", verdict="wrong"), "feedback"),
        (Message(refers_to="t", expected=[6]), "feedback"),
        (Message(demos=DEMOS, query=[5], task_id="named"), "answer"),   # a name is not intent
    ])
    def test_prior_routes_structured_messages(self, msg, intent):
        probs = IntentHead().predict(msg)
        assert max(probs, key=probs.get) == intent and probs[intent] > 0.6

    def test_bare_text_is_uncertain(self):
        assert max(IntentHead().predict(Message(text="hmm?")).values()) < 0.6

    def test_learns_a_user_convention(self):
        head = IntentHead(lr=0.5)
        msg = Message(demos=self.DEMOS, query=[5], text="remember this")
        for _ in range(10):
            head.update(msg, "learn")
        probs = head.predict(msg)
        assert max(probs, key=probs.get) == "learn"
        with pytest.raises(ValueError):
            head.load_list([[0.0]])


class TestPerceive:
    def test_routes_answer_learn_feedback_and_asks(self, tmp_path):
        with Model.create(tmp_path / "m", config()) as m:
            ans = m.perceive(Message(demos=[([1, 4, 2], [3, 6, 4])], query=[7, 7], task_id="q1"))
            assert ans.intent == "answer" and ans.solution.answer == [9, 9]
            lesson = m.perceive(Message(demos=[([1, 2, 3], [3, 2, 1]), ([4, 0, 9], [9, 0, 4])]))
            assert lesson.intent == "learn" and len(m.knowledge) == 1
            fb = m.perceive(Message(refers_to="q1", verdict="correct"))
            assert fb.intent == "feedback" and fb.feedback.verdict == "correct"
            assert m.perceive(Message(text="?")).question is not None
            assert m.perceive(Message(query=[1])).question is not None

    def test_confirmed_intents_and_competence_persist(self, tmp_path):
        msg = Message(demos=[([1], [2])], query=[3], text="remember this")
        with Model.create(tmp_path / "m", config()) as m:
            m.solve([([1, 2], [2, 3])], [5], "t1")
            for _ in range(5):
                m.confirm_intent(msg, "learn")
            before = m.intent.predict(msg)
        with Model.open(tmp_path / "m") as m:
            assert m.intent.predict(msg) == before
            assert [r["region"] for r in m.self_report()] == ["sequence/affine"]


class TestSelfStudy:
    def test_refuses_without_an_intact_judge(self, tmp_path, benchmark):
        with Model.create(tmp_path / "a", config()) as m:
            with pytest.raises(ModelError):
                m.study(StudyBudget(max_tasks=1))
        with Model.create(tmp_path / "b", config(), benchmark=benchmark) as m:
            m.manifest["harness"]["lock"] = "0" * 64
            with pytest.raises(HarnessLockError):
                m.study(StudyBudget(max_tasks=1))

    def test_studies_gaps_and_sleeps(self, tmp_path, benchmark):
        with Model.create(tmp_path / "m", config(min_successes=1), benchmark=benchmark) as m:
            report = m.study(StudyBudget(max_tasks=6, sleep_every=3), generators=[Composites()])
            assert report.tasks == 6 and report.stopped == "task budget"
            assert len(report.cycles) == 2
            assert report.correct + report.wrong + report.abstained == 6
            assert m.self_model.get("sequence/scale+reverse").attempts == 6
            assert len(m.knowledge) + len(m.reasoner.priors) > 0     # it learned something

    def test_gives_up_on_unlearnable_region_by_itself(self, tmp_path, benchmark):
        with Model.create(tmp_path / "m", config(), benchmark=benchmark) as m:
            report = m.study(StudyBudget(max_tasks=20, sleep_every=50), generators=[Gravity()])
            assert report.stopped == "nothing learnable left"
            assert report.visited == {"grid/gravity": META.patience}
            assert report.correct == 0

    def test_repeated_discards_stop_for_human_review(self, tmp_path, benchmark, monkeypatch):
        monkeypatch.setattr(sleep_module, "gate", lambda *a: ["judge says no"])
        with Model.create(tmp_path / "m", config(min_successes=1), benchmark=benchmark) as m:
            report = m.study(StudyBudget(max_tasks=20, sleep_every=1, max_consecutive_discards=2),
                             generators=[Composites()])
            assert report.stopped.startswith("repeated discarded cycles")
            assert [c.status for c in report.cycles][-2:] == ["discard", "discard"]
