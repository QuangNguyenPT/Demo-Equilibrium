"""Sleep cycle (loop B): frozen harness, gate, promotion modes, graduation, rollback."""
import json

import numpy as np
import pytest

import hrm.learning.sleep as sleep_module
from hrm.config import (LearningConfig, ModelConfig, PerceptionConfig, ReasoningConfig,
                        SubstrateConfig)
from hrm.learning import (DECOY, GridTasks, Harness, HarnessLockError, HarnessResult,
                          SequenceTasks, gate)
from hrm.model import Model, ModelError
from hrm.perception.sequence import Affine, Reverse, SequenceProgram

COMPOSITE = SequenceProgram((Affine(2, 0), Reverse()))


def config(promotion: str = "auto") -> ModelConfig:
    return ModelConfig(substrate=SubstrateConfig(dim=512),
                       perception=PerceptionConfig(grid_max_shift=3),
                       reasoning=ReasoningConfig(max_ticks=200, restart_budget=2),
                       learning=LearningConfig(promotion=promotion))


class SmallSequenceTasks(SequenceTasks):
    families = ("offset", "reverse")


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    path = tmp_path_factory.mktemp("bench") / "benchmark.json"
    return Harness.build(path, [SmallSequenceTasks()], per_family=2, decoys_per_domain=2, seed=5)


def composite_task(seed):
    rng = np.random.default_rng(seed)
    xs = [[int(v) for v in rng.choice(np.arange(10), 5, replace=False)] for _ in range(4)]
    return [(x, COMPOSITE.apply(x)) for x in xs[:3]], xs[3]


def learn_composite(m: Model) -> int:
    """Two verified uses of the same composite: ready for consolidation."""
    first = m.solve(*composite_task(1), task_id="c1")
    second = m.solve(*composite_task(2), task_id="c2")
    assert first.status == "synthesized" and second.status == "fast"
    item = m.knowledge.items()[0]
    assert item.successes >= 2
    return item.id


def result(rate, fams, false_accepts=0) -> HarnessResult:
    return HarnessResult(rate, fams, false_accepts, 10, 0.0)


class TestGate:
    def test_passes_when_nothing_regresses(self):
        assert gate(result(0.5, {"a": 0.5}), result(0.5, {"a": 0.5}), 0.0) == []

    def test_any_false_accept_fails(self):
        assert "decoy" in gate(result(0.5, {}), result(0.9, {}, 1), 0.0)[0]

    def test_overall_drop_fails(self):
        assert gate(result(0.6, {}), result(0.5, {}), 0.0)

    def test_family_regression_fails_even_if_total_improves(self):
        reasons = gate(result(0.5, {"a": 1.0, "b": 0.0}), result(0.6, {"a": 0.5, "b": 0.7}), 0.1)
        assert reasons == ["a regressed 1.000 -> 0.500"]


class TestHarness:
    def test_generators_produce_ground_truth(self):
        rng = np.random.default_rng(0)
        for gen in (GridTasks(), SequenceTasks()):
            for family in gen.families:
                task = gen.generate(family, rng)
                assert task.family == family and len(task.demos) == gen.n_demos
            assert gen.decoy(rng).family == DECOY

    def test_model_scores_benchmark_and_abstains_on_decoys(self, tmp_path, benchmark):
        with Model.create(tmp_path / "m", config(), benchmark=benchmark) as m:
            res = m.require_harness().evaluate(m.reasoner, m.domains)
            assert res.false_accepts == 0 and res.solve_rate == 1.0
            assert set(res.per_family) == {"sequence/offset", "sequence/reverse"}
            assert m.knowledge.stats()["rows"] == 0           # evaluation never learns

    def test_edited_benchmark_breaks_the_lock_until_resigned(self, tmp_path, benchmark):
        with Model.create(tmp_path / "m", config(), benchmark=benchmark) as m:
            bench = tmp_path / "m" / "harness" / "benchmark.json"
            data = json.loads(bench.read_text())
            data["tasks"] = data["tasks"][:1]                 # an easier test: tampering
            bench.write_text(json.dumps(data))
            m.harness = Harness(bench)
            with pytest.raises(HarnessLockError):
                m.sleep()
            m.resign_harness()                                # human accepts the new judge
            assert m.sleep().status == "skip"

    def test_sleep_requires_a_benchmark(self, tmp_path):
        with Model.create(tmp_path / "m", config()) as m:
            with pytest.raises(ModelError, match="no frozen benchmark"):
                m.sleep()


class TestSleepCycle:
    def test_skip_when_nothing_is_ready(self, tmp_path, benchmark):
        with Model.create(tmp_path / "m", config(), benchmark=benchmark) as m:
            assert m.sleep().status == "skip" and m.generation == 0

    def test_keep_consolidates_graduates_and_persists(self, tmp_path, benchmark):
        with Model.create(tmp_path / "m", config(), benchmark=benchmark) as m:
            item = learn_composite(m)
            report = m.sleep()
            assert report.status == "keep" and m.generation == 1
            assert report.consolidated == [COMPOSITE.describe()]
            assert report.graduated == [item] and len(m.knowledge) == 0
            fresh = m.solve(*composite_task(3), task_id="c3")
            assert fresh.status == "fast" and not fresh.retrieved   # known by the core now
        with Model.open(tmp_path / "m") as m:
            assert m.generation == 1
            assert m.reasoner.priors.has("sequence", COMPOSITE.describe())
            ledger = (tmp_path / "m" / "ledger" / "results.tsv").read_text().splitlines()
            assert ledger[0].startswith("at\tgeneration") and "\tkeep\t" in ledger[-1]

    def test_staged_waits_for_human_promotion(self, tmp_path, benchmark):
        with Model.create(tmp_path / "m", config("staged"), benchmark=benchmark) as m:
            learn_composite(m)
            report = m.sleep()
            assert report.status == "staged" and report.candidate_generation == 1
            assert m.generation == 0 and not m.reasoner.priors.has("sequence",
                                                                    COMPOSITE.describe())
            graduated = m.promote(1)
            assert m.generation == 1 and graduated
            with pytest.raises(ModelError, match="not a staged candidate"):
                m.promote(1)

    def test_discard_restores_core_exactly(self, tmp_path, benchmark, monkeypatch):
        with Model.create(tmp_path / "m", config(), benchmark=benchmark) as m:
            learn_composite(m)
            before = m.reasoner.priors.to_list()
            monkeypatch.setattr(sleep_module, "gate", lambda *a: ["judge says no"])
            report = m.sleep()
            assert report.status == "discard" and report.reasons == ["judge says no"]
            assert m.reasoner.priors.to_list() == before and m.generation == 0
            assert len(m.knowledge) == 1                      # nothing graduated

    def test_integrity_problem_discards_without_touching_the_core(self, tmp_path, benchmark):
        with Model.create(tmp_path / "m", config(), benchmark=benchmark) as m:
            learn_composite(m)
            m.knowledge._keys[0] = np.float16(3.0)            # corrupt a key behind the API
            report = m.sleep()
            assert report.status == "discard" and report.baseline is None

    def test_rollback_restores_core_and_reactivates_knowledge(self, tmp_path, benchmark):
        with Model.create(tmp_path / "m", config(), benchmark=benchmark) as m:
            item = learn_composite(m)
            m.sleep()
            back, lost = m.rollback(0)
            assert m.generation == 0 and back == [item] and lost == []
            assert not m.reasoner.priors.has("sequence", COMPOSITE.describe())
            assert m.solve(*composite_task(4), task_id="c4").retrieved == [item]
