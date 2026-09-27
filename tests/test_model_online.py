"""Model bundle lifecycle and online learning (loop A): episodes, answers, feedback."""
import json

import numpy as np
import pytest

from hrm.config import ModelConfig, PerceptionConfig, ReasoningConfig, SubstrateConfig
from hrm.model import Model, ModelError
from hrm.perception.grid import ColorCycle, GridProgram, Translate

CFG = ModelConfig(substrate=SubstrateConfig(dim=512),
                  perception=PerceptionConfig(grid_max_shift=3),
                  reasoning=ReasoningConfig(max_ticks=300, restart_budget=3))
COMPOSITE = GridProgram((Translate(1, 0), ColorCycle(2)))
RNG = np.random.default_rng(21)


def grid_task(program=COMPOSITE, n=3, size=8):
    demos = []
    for _ in range(n + 1):
        g = np.zeros((size, size), dtype=np.int64)
        x, y = RNG.integers(1, size - 4, 2)
        g[y:y + 2, x:x + 2] = int(RNG.integers(1, 7))
        demos.append((g, program.apply(g)))
    return demos[:n], demos[n][0], demos[n][1]


@pytest.fixture
def model(tmp_path):
    with Model.create(tmp_path / "m", CFG) as m:
        yield m


class TestBundle:
    def test_create_writes_manifest_config_and_core(self, tmp_path):
        Model.create(tmp_path / "m", CFG).close()
        root = tmp_path / "m"
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["active_generation"] == 0
        assert (root / "core" / "gen-0000.json").exists()
        assert ModelConfig.load(root / "config.json") == CFG

    def test_create_refuses_to_overwrite(self, tmp_path):
        Model.create(tmp_path / "m", CFG).close()
        with pytest.raises(ModelError, match="already exists"):
            Model.create(tmp_path / "m", CFG)

    def test_open_missing_model(self, tmp_path):
        with pytest.raises(ModelError, match="no model"):
            Model.open(tmp_path / "nothing")

    def test_core_weights_round_trip(self, tmp_path):
        with Model.create(tmp_path / "m", CFG) as m:
            m.reasoner.priors.consolidate("grid", COMPOSITE, 0.5)
            m.reasoner.engine.resonator.params["gain"][:] = 0.42
            m.qhead.w[1, 3] = 7.0
            m._write_core(0)
        with Model.open(tmp_path / "m") as m:
            assert m.reasoner.priors.prior("grid", COMPOSITE) == 0.5
            assert np.allclose(m.reasoner.engine.resonator.params["gain"], 0.42)
            assert m.qhead.w[1, 3] == 7.0

    def test_corrupted_core_shape_rejected(self, tmp_path):
        Model.create(tmp_path / "m", CFG).close()
        path = tmp_path / "m" / "core" / "gen-0000.json"
        core = json.loads(path.read_text())
        core["qhead"] = [[0.0]]
        path.write_text(json.dumps(core))
        with pytest.raises(ValueError, match="qhead"):
            Model.open(tmp_path / "m")


class TestOnlineLearning:
    def test_synthesized_knowledge_is_kept_and_reused_after_reopen(self, tmp_path):
        with Model.create(tmp_path / "m", CFG) as m:
            demos, test_in, _ = grid_task()
            assert m.solve(demos, test_in, "t1").status == "synthesized"
            assert len(m.knowledge) == 1
            assert m.knowledge.episode("t1").domain == "grid"
        with Model.open(tmp_path / "m") as m:
            demos, test_in, expected = grid_task()
            sol = m.solve(demos, test_in, "t2")
            assert sol.status == "fast" and np.array_equal(sol.answer, expected)
            item = m.knowledge.items()[0]
            assert item.successes == 2                      # reuse is self-evidence
            assert m.knowledge.answer("t2").items == [item.id]

    def test_abstention_is_recorded_and_writes_nothing(self, model):
        demos = [(RNG.integers(0, 5, (6, 6)), RNG.integers(0, 5, (6, 6))) for _ in range(3)]
        sol = model.solve(demos, np.zeros((6, 6), dtype=np.int64), "noise")
        assert sol.status == "abstained" and len(model.knowledge) == 0
        assert model.knowledge.answer("noise").status == "abstained"

    def test_teach_keeps_explanation_once(self, model):
        demos, _, _ = grid_task(GridProgram((Translate(2, 0),)))
        sol = model.teach(demos, "lesson")
        assert sol.solved and model.knowledge.items()[0].provenance == "taught"
        again, _, _ = grid_task(GridProgram((Translate(2, 0),)))
        model.teach(again, "lesson2")
        assert len(model.knowledge) == 1                   # known: evidence, not a copy


class TestFeedback:
    def test_human_wrong_quarantines_source_knowledge(self, model):
        model.solve(*grid_task()[:2], task_id="t1")
        demos, test_in, _ = grid_task()
        assert model.solve(demos, test_in, "t2").status == "fast"
        result = model.feedback("t2", "wrong")
        assert result.quarantined and len(model.knowledge) == 0
        assert model.knowledge.answer("t2").verdict == "wrong"

    def test_human_correct_raises_confidence(self, model):
        model.solve(*grid_task()[:2], task_id="t1")
        demos, test_in, _ = grid_task()
        model.solve(demos, test_in, "t2")
        before = model.knowledge.items()[0].confidence
        model.feedback("t2", "correct")
        assert model.knowledge.items()[0].confidence > before

    def test_wrong_consolidated_prior_is_forgotten(self, model):
        model.reasoner.priors.consolidate("grid", COMPOSITE, 0.5)
        demos, test_in, _ = grid_task()
        assert model.solve(demos, test_in, "t1").status == "fast"
        assert model.feedback("t1", "wrong").forgotten_prior
        assert model.reasoner.priors.prior("grid", COMPOSITE) == 0.0

    def test_correction_disambiguates_and_is_learned(self, model):
        sol = model.solve([([1, 1], [2, 2])], [3, 5], "amb")
        assert sol.confidence < 0.5
        truth = [6, 10]                                     # the intended rule was "double"
        result = model.feedback("amb", "wrong", correction=truth)
        assert result.relearned.program.describe() == "affine(a=2, b=0)"
        assert any(i.provenance == "corrected" for i in model.knowledge.items())

    def test_feedback_validation(self, model):
        model.solve([([1, 2], [2, 3])], [5], "t")
        with pytest.raises(ValueError):
            model.feedback("t", "maybe")
        with pytest.raises(ValueError):
            model.feedback("t", "correct", correction=[1])
        with pytest.raises(KeyError):
            model.feedback("unknown-task", "wrong")
