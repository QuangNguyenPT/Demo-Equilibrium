"""Language section: program language parsing, inner speech, text perception, backends."""
import os

import numpy as np
import pytest

from hrm.config import (ConfigError, LanguageConfig, ModelConfig, PerceptionConfig,
                        ReasoningConfig, SubstrateConfig)
from hrm.language import (HashingLanguage, InnerSpeech, LanguageError, OllamaLanguage,
                          make_language, parse_task, rewrite)
from hrm.metacognition import IntentHead, Message, TextSignal
from hrm.model import Model
from hrm.perception import DomainRegistry, GridDomain, SequenceDomain
from hrm.perception.grid import GridProgram, Translate, ColorCycle
from hrm.perception.sequence import Affine, Reverse, SequenceProgram
from hrm.reasoning import Reasoner
from hrm.substrate import Codebook

CFG = ModelConfig(substrate=SubstrateConfig(dim=512),
                  perception=PerceptionConfig(grid_max_shift=2),
                  reasoning=ReasoningConfig(max_ticks=150, restart_budget=2))
CB = Codebook(CFG.substrate)
GRID = GridDomain(CB, CFG.perception)
SEQ = SequenceDomain(CB, CFG.perception)
COMPOSITE = SequenceProgram((Affine(2, 0), Reverse()))
DEMOS = [(x, COMPOSITE.apply(x)) for x in ([1, 2, 3], [4, 0, 7, 1], [5, 6])]


class Failing(HashingLanguage):
    def generate(self, prompt, max_new_tokens):
        raise LanguageError("backend down")


class TestProgramLanguage:
    def _round_trip(self, domain, program, sample):
        back = domain.parse_program(program.describe())
        assert back.describe() == program.describe()
        assert domain.equal(back.apply(sample), program.apply(sample))

    def test_every_grid_candidate_round_trips(self):
        g = np.zeros((6, 6), dtype=np.int64)
        g[1, 1:3], g[3, 2] = 1, 2
        demos = [(g, GridProgram((ColorCycle(2),)).apply(g))]
        vol = np.zeros((3, 5, 5), dtype=np.int64)
        vol[1, 1:3, 1] = 4
        for program in GRID.candidates(demos) + GRID.candidates([(vol, vol)]):
            self._round_trip(GRID, program, vol if program.describe().startswith("ortho") else g)
        self._round_trip(GRID, GridProgram((Translate(1, -2), ColorCycle(3))), g)

    def test_every_sequence_candidate_round_trips(self):
        for program in SEQ.candidates(DEMOS) + [COMPOSITE]:
            self._round_trip(SEQ, program, [3, 1, 4, 1, 5])

    @pytest.mark.parametrize("text", [
        "__import__('os').system('echo pwned')", "translate(dx=99999, dy=0)",
        "recolor({1: 30})", "affine(a=1, c=2)", "dihedral(r45)", "", "reverse(1)",
        "affine(a=True, b=0)", "translate(dx=1, dy=0) . teleport",
    ])
    def test_rejects_unsafe_or_malformed_text(self, text):
        for domain in (GRID, SEQ):
            with pytest.raises(ValueError):
                domain.parse_program(text)


class TestInnerSpeech:
    def test_parses_valid_lines_and_ignores_the_rest(self):
        lang = HashingLanguage(responder=lambda p: "Sure!\n1. affine(a=2, b=0) . reverse\n"
                                                   "- rotate the list\n`sort(desc)`")
        programs, said = InnerSpeech(lang).propose(SEQ, DEMOS)
        assert [p.describe() for p in programs] == ["affine(a=2, b=0) . reverse", "sort(desc)"]
        assert "Allowed operations" in InnerSpeech(lang).prompt(SEQ, DEMOS)

    def test_failing_backend_means_no_proposals(self):
        assert InnerSpeech(Failing()).propose(SEQ, DEMOS) == ([], "")

    def test_correct_proposal_is_answered_in_language_without_ticks(self):
        lang = HashingLanguage(responder=lambda p: "affine(a=2, b=0) . reverse")
        r = Reasoner(CFG, CB, DomainRegistry([GRID, SEQ]), speech=InnerSpeech(lang))
        sol = r.solve(DEMOS, [9, 8])
        assert sol.status == "language" and sol.ticks == 0 and sol.answer == [16, 18]
        assert sol.trace.of_kind("inner_speech")[0].detail["proposals"] == [
            "affine(a=2, b=0) . reverse"]

    def test_fluent_wrong_proposal_is_refused_by_verification(self):
        lang = HashingLanguage(responder=lambda p: "reverse")
        r = Reasoner(CFG, CB, DomainRegistry([GRID, SEQ]), speech=InnerSpeech(lang))
        sol = r.solve(DEMOS, [9, 8])
        assert sol.status == "synthesized" and sol.answer == [16, 18]


class TestTextPerception:
    def test_arrow_notation(self):
        demos, query = parse_task("1 2 3 -> 3 2 1 ; 4 5 -> 5 4 ; 7 8 9 -> ?")
        assert demos == [([1, 2, 3], [3, 2, 1]), ([4, 5], [5, 4])] and query == [7, 8, 9]
        grid_demos, grid_query = parse_task("0 1 / 1 0 -> 1 0 / 0 1\n2 0 / 0 2 -> ?")
        assert grid_demos[0][0].shape == (2, 2) and grid_query.tolist() == [[2, 0], [0, 2]]

    @pytest.mark.parametrize("text", ["hello there", "1 2 -> ", "1 2 / 3 -> 4", "-> 3",
                                      "1 -> ? ; 2 -> ?"])
    def test_not_the_notation(self, text):
        assert parse_task(text) is None

    def test_free_language_is_rewritten_then_validated(self):
        lang = HashingLanguage(responder=lambda p: "Here you go:\n1 2 -> 2 3 ; 5 -> 6 ; 9 -> ?")
        demos, query = rewrite(lang, "add one to every number; what about nine?")
        assert demos == [([1, 2], [2, 3]), ([5], [6])] and query == [9]
        assert rewrite(HashingLanguage(responder=lambda p: "I refuse"), "x") is None


class TestIntentText:
    def test_hashing_embeddings(self):
        lang = HashingLanguage()
        a, b = lang.embed("that answer was wrong"), lang.embed("wrong answer")
        assert np.isclose(np.linalg.norm(a), 1.0) and np.array_equal(a, lang.embed(
            "that answer was wrong"))
        assert a @ b > a @ lang.embed("please remember this rule")

    def test_text_signal_moves_intent(self):
        signal = TextSignal(HashingLanguage().embed)
        msg = Message(text="no, that was wrong")
        probs = IntentHead().predict(msg, signal(msg.text))
        assert max(probs, key=probs.get) == "feedback"


class TestModelLanguage:
    def test_text_task_answered_and_explained(self, tmp_path):
        with Model.create(tmp_path / "m", CFG) as m:
            resp = m.perceive(Message(text="1 2 3 -> 2 3 4 ; 5 6 -> 6 7 ; 10 11 -> ?",
                                      task_id="t1"))
            assert resp.intent == "answer" and resp.solution.answer == [11, 12]
            assert "affine(a=1, b=1)" in m.explain("t1")
            m.solve([([1, 2], [5, 1]), ([3, 4], [0, 9])], [5, 6], "noise")
            assert "did not answer" in m.explain("noise")

    def test_language_backend_from_config_thinks_inside_reasoning(self, tmp_path):
        lang = HashingLanguage(responder=lambda p: ("It doubles and reverses."
                                                    if "explain" in p else
                                                    "affine(a=2, b=0) . reverse"))
        with Model.create(tmp_path / "m", CFG, language=lang) as m:
            sol = m.solve(DEMOS, [9, 8], "t")
            assert sol.status == "language"
            assert m.explain("t").startswith("It doubles and reverses.")

    def test_factory_and_config(self):
        assert make_language(LanguageConfig()) is None
        assert isinstance(make_language(LanguageConfig(backend="hashing")), HashingLanguage)
        with pytest.raises(ConfigError):
            LanguageConfig(backend="transformers").validate()

    def test_unreachable_ollama_is_a_language_error(self):
        with pytest.raises(LanguageError):
            OllamaLanguage("any", endpoint="http://127.0.0.1:9", timeout=2).generate("hi", 4)


@pytest.mark.skipif(not os.environ.get("HRM_TEST_HF_MODEL"),
                    reason="set HRM_TEST_HF_MODEL to a local/HF model id to run")
def test_transformers_backend_end_to_end():
    from hrm.language import TransformersLanguage
    lang = TransformersLanguage(os.environ["HRM_TEST_HF_MODEL"])
    v = lang.embed("remember this rule")
    assert v.ndim == 1 and np.isclose(np.linalg.norm(v), 1.0, atol=1e-3)
    assert isinstance(lang.generate("Say hi.", 8), str)
