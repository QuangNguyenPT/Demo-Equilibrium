"""The model: one bundle directory on SSD, loaded section by section.

    <dir>/manifest.json          format, generations, the active generation, harness lock
    <dir>/config.json            ModelConfig (strict)
    <dir>/core/gen-NNNN.json     core weights of each generation (small, RAM-resident)
    <dir>/knowledge/             SSD knowledge: programs, episodes, answers, evidence
    <dir>/harness/benchmark.json frozen benchmark (the learning loop's judge)
    <dir>/ledger/results.tsv     one row per sleep cycle
    <dir>/metacognition.json     competence map + intent matrix (updated by experience
                                 and human confirmation, not by the gated core)

The core is loaded into RAM; knowledge stays on disk and is read inside the
reasoning pass. Every file update is atomic (write a temp file, then replace).
Humans keep the last word: they choose staged promotion, promote, roll back,
and are the only ones who can re-sign the harness lock.
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hrm.config import ModelConfig
from hrm.core import core_from_dict, core_to_dict
from hrm.knowledge import KnowledgeStore
from hrm.language import (InnerSpeech, LanguageSection, literal_explanation, make_language,
                          parse_task, rewrite)
from hrm.learning.generators import GridTasks, SequenceTasks, TaskGenerator
from hrm.learning.harness import Harness, HarnessResult
from hrm.learning.online import FeedbackResult, Learner
from hrm.learning.sleep import CycleReport, SleepCycle
from hrm.learning.study import SelfStudy, StudyBudget, StudyReport
from hrm.metacognition import IntentHead, Message, SelfModel, TextSignal
from hrm.perception import (DomainRegistry, Example, GridDomain, SequenceDomain,
                            Vector3DDomain)
from hrm.reasoning import QHead, Reasoner, Solution
from hrm.substrate import Codebook

MODEL_FORMAT = 1
_LEDGER_HEADER = "at\tgeneration\tstatus\tsolve_rate\tfalse_accepts\tactive_knowledge\tdescription\n"


@dataclass
class Response:
    """What `perceive` did with a message, or what it needs to know first."""
    intent: str                                   # answer | learn | feedback | unclear
    intent_confidence: float
    solution: Optional[Solution] = None
    feedback: Optional[FeedbackResult] = None
    question: Optional[str] = None                # asked back instead of guessing


class ModelError(RuntimeError):
    """The bundle is missing, malformed, or the request would corrupt it."""


class Model:
    def __init__(self, directory: Path, cfg: ModelConfig, manifest: Dict[str, Any],
                 language: Optional[LanguageSection] = None):
        self.dir = directory
        self.cfg = cfg
        self.manifest = manifest
        self.cb = Codebook(cfg.substrate)
        self.domains = DomainRegistry([GridDomain(self.cb, cfg.perception),
                                       SequenceDomain(self.cb, cfg.perception),
                                       Vector3DDomain(self.cb, cfg.perception)])
        self.knowledge = KnowledgeStore(directory / "knowledge", cfg.substrate, cfg.knowledge)
        self.language = language if language is not None else make_language(cfg.language)
        self.speech = (InnerSpeech(self.language, cfg.language.max_proposals,
                                   cfg.language.max_new_tokens) if self.language else None)
        self._text_signal = TextSignal(self.language.embed) if self.language else None
        self.qhead = QHead()
        self.reasoner = Reasoner(cfg, self.cb, self.domains, knowledge=self.knowledge,
                                 qhead=self.qhead, speech=self.speech)
        self.learner = Learner(self.knowledge, self.reasoner, self.domains)
        harness = manifest.get("harness")
        self.harness = Harness(directory / harness["benchmark"]) if harness else None
        self.self_model = SelfModel(cfg.meta)
        self.intent = IntentHead(cfg.meta.intent_lr)
        meta_path = directory / "metacognition.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.self_model.load_dict(meta["self_model"])
            self.intent.load_list(meta["intent"])

    # --- lifecycle -----------------------------------------------------------------
    @classmethod
    def create(cls, directory: str | Path, cfg: Optional[ModelConfig] = None,
               benchmark: Optional[str | Path] = None,
               language: Optional[LanguageSection] = None) -> "Model":
        """New bundle. `benchmark` is copied in and locked; without one, the sleep
        cycle is disabled because nothing independent could judge it."""
        path = Path(directory)
        if (path / "manifest.json").exists():
            raise ModelError(f"a model already exists at {path}")
        cfg = (cfg or ModelConfig()).validate()
        (path / "core").mkdir(parents=True, exist_ok=True)
        cfg.save(path / "config.json")
        manifest: Dict[str, Any] = {
            "format": MODEL_FORMAT, "active_generation": 0,
            "generations": [{"generation": 0, "status": "active", "created": time.time(),
                             "note": "initial weights"}]}
        if benchmark is not None:
            (path / "harness").mkdir(exist_ok=True)
            shutil.copyfile(benchmark, path / "harness" / "benchmark.json")
            manifest["harness"] = {"benchmark": "harness/benchmark.json",
                                   "lock": Harness(path / "harness" / "benchmark.json").lock()}
        model = cls(path, cfg, manifest, language)
        model._write_core(0)
        model._write_manifest()
        return model

    @classmethod
    def open(cls, directory: str | Path,
             language: Optional[LanguageSection] = None) -> "Model":
        path = Path(directory)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise ModelError(f"no model at {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != MODEL_FORMAT:
            raise ModelError(f"unsupported model format {manifest.get('format')}")
        model = cls(path, ModelConfig.load(path / "config.json"), manifest, language)
        model._load_core(manifest["active_generation"])
        return model

    def close(self) -> None:
        self.knowledge.close()

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def generation(self) -> int:
        return int(self.manifest["active_generation"])

    # --- interaction -----------------------------------------------------------------
    def solve(self, demos: Sequence[Example], test_input: Any, task_id: Optional[str] = None,
              observe: bool = True) -> Solution:
        """Answer a task; what is learned from it is kept (verified only)."""
        task_id = task_id or _new_task_id()
        sol = self.reasoner.solve(demos, test_input, task_id)
        self.learner.after_solve(sol, task_id, demos, test_input)
        if observe:
            family = sol.program.family if sol.solved else "unknown"
            self.self_model.observe(f"{sol.domain}/{family}",
                                    "solved" if sol.solved else "abstained")
            self.save_metacognition()
        return sol

    def teach(self, demos: Sequence[Example], task_id: Optional[str] = None) -> Solution:
        """Learn from demonstrations: find and keep a verified explanation."""
        return self.learner.teach(demos, task_id or _new_task_id())

    def feedback(self, task_id: str, verdict: str, correction: Any = None) -> FeedbackResult:
        """Human judgement of a past answer; outranks the model's own evidence."""
        result = self.learner.feedback(task_id, verdict, correction)
        answer = self.knowledge.answer(task_id)
        if answer.family is not None:
            domain = self.knowledge.episode(task_id).domain
            self.self_model.observe(f"{domain}/{answer.family}",
                                    "correct" if verdict == "correct" else "wrong")
            self.save_metacognition()
        return result

    def perceive(self, msg: Message) -> Response:
        """Infer what the sender wants, then do it, or ask when unsure.

        A task written as text is first read into structure: the arrow notation
        directly, free language through the language section (validated)."""
        if msg.text and not msg.demos:
            parsed = parse_task(msg.text)
            if parsed is None and self.language is not None:
                parsed = rewrite(self.language, msg.text)
            if parsed is not None:
                demos, query = parsed
                msg = replace(msg, demos=demos, query=msg.query if query is None else query)
        probs = self.intent.predict(msg, self._signal(msg))
        intent = max(probs, key=probs.get)
        conf = probs[intent]
        if conf < self.cfg.meta.intent_threshold:
            return Response("unclear", conf, question="Should I answer this, learn from it, or "
                                                      "take it as feedback on an earlier answer?")
        if intent == "answer":
            if not msg.demos or msg.query is None:
                return Response(intent, conf, question="To answer I need example pairs and "
                                                       "the input to solve.")
            return Response(intent, conf, solution=self.solve(msg.demos, msg.query, msg.task_id))
        if intent == "learn":
            demos = list(msg.demos or [])
            if msg.query is not None and msg.expected is not None:
                demos.append((msg.query, msg.expected))
            if not demos:
                return Response(intent, conf, question="To learn I need at least one example.")
            return Response(intent, conf, solution=self.teach(demos, msg.task_id))
        if msg.refers_to is None:
            return Response(intent, conf, question="Which earlier answer is this about?")
        verdict = msg.verdict or ("wrong" if msg.expected is not None else None)
        if verdict is None:
            return Response(intent, conf, question="Was my answer correct or wrong?")
        correction = msg.expected if verdict == "wrong" else None
        return Response(intent, conf, feedback=self.feedback(msg.refers_to, verdict, correction))

    def confirm_intent(self, msg: Message, intent: str) -> None:
        """Human: this is what that message meant. Trains the intent matrix."""
        self.intent.update(msg, intent, self._signal(msg))
        self.save_metacognition()

    def explain(self, task_id: str) -> str:
        """Say, in words, why the model answered a task the way it did."""
        answer = self.knowledge.answer(task_id)
        episode = self.knowledge.episode(task_id)
        if answer.program is None:
            return ("I did not find a rule that reproduces the examples, "
                    "so I did not answer rather than guess.")
        domain = self.domains.get(episode.domain)
        program = domain.parse_program(answer.program)
        answer_text = domain.render(domain.from_json(answer.answer))
        if self.speech is not None:
            return self.speech.explain(program, len(episode.demos), answer_text)
        return literal_explanation(program, len(episode.demos), answer_text)

    def _signal(self, msg: Message):
        return self._text_signal(msg.text) if self._text_signal is not None else None

    def self_report(self) -> List[Dict[str, Any]]:
        """What the model believes it knows: every region with its status."""
        return self.self_model.report()

    def study(self, budget: Optional[StudyBudget] = None,
              generators: Optional[Sequence[TaskGenerator]] = None) -> StudyReport:
        """Self-directed study of its own knowledge gaps (see hrm.learning.study)."""
        generators = generators or [GridTasks(max_shift=self.cfg.perception.grid_max_shift),
                                    SequenceTasks()]
        return SelfStudy(self, generators).run(budget or StudyBudget())

    def save_metacognition(self) -> None:
        _atomic_write(self.dir / "metacognition.json",
                      json.dumps({"self_model": self.self_model.to_dict(),
                                  "intent": self.intent.to_list()}, indent=1))

    # --- sleep, generations, human controls ----------------------------------------------
    def sleep(self) -> CycleReport:
        """One consolidation cycle behind the frozen benchmark gate."""
        return SleepCycle(self).run()

    @property
    def generations(self) -> List[Dict[str, Any]]:
        return list(self.manifest["generations"])

    def promote(self, generation: int) -> List[int]:
        """Human: make a staged candidate live; returns the knowledge it graduated."""
        if self._generation_entry(generation)["status"] != "candidate":
            raise ModelError(f"generation {generation} is not a staged candidate")
        self._activate(generation)
        items = [i for i in self.knowledge.items("active")
                 if self.reasoner.priors.has(i.domain, i.signature)]
        return SleepCycle(self).graduate(items)

    def rollback(self, generation: int) -> Tuple[List[int], List[int]]:
        """Human: return to an earlier core. Knowledge graduated after it becomes
        active again; returns (reactivated ids, ids whose key was already evicted)."""
        if self._generation_entry(generation)["status"] == "candidate":
            raise ModelError(f"generation {generation} is a staged candidate; use promote()")
        if generation == self.generation:
            return [], []
        self._activate(generation)
        return self.knowledge.reactivate_graduated(after_generation=generation)

    def resign_harness(self) -> str:
        """Human: accept the current benchmark and scorer as the judge."""
        harness = self.require_harness(check=False)
        self.manifest["harness"]["lock"] = harness.lock()
        self._write_manifest()
        return self.manifest["harness"]["lock"]

    def require_harness(self, check: bool = True) -> Harness:
        if self.harness is None:
            raise ModelError("this model has no frozen benchmark; the learning loop is disabled")
        if check:
            self.harness.check(self.manifest["harness"]["lock"])
        return self.harness

    def new_generation(self, note: str, result: HarnessResult, activate: bool) -> int:
        """Persist the in-RAM core as a new generation (live, or a staged candidate)."""
        generation = max(g["generation"] for g in self.manifest["generations"]) + 1
        self._write_core(generation)
        self.manifest["generations"].append({
            "generation": generation, "status": "candidate", "created": time.time(),
            "note": note, "gate": result.to_dict()})
        if activate:
            self._set_active(generation)
        self._write_manifest()
        return generation

    def append_ledger(self, generation: int, status: str, solve_rate: Optional[float],
                      false_accepts: Optional[int], description: str) -> None:
        path = self.dir / "ledger" / "results.tsv"
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            path.write_text(_LEDGER_HEADER, encoding="utf-8")
        row = [datetime.datetime.now().isoformat(timespec="seconds"), str(generation), status,
               "" if solve_rate is None else f"{solve_rate:.4f}",
               "" if false_accepts is None else str(false_accepts),
               str(len(self.knowledge)), description.replace("\t", " ").replace("\n", " ")]
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\t".join(row) + "\n")

    def _generation_entry(self, generation: int) -> Dict[str, Any]:
        for entry in self.manifest["generations"]:
            if entry["generation"] == generation:
                return entry
        raise ModelError(f"no generation {generation}")

    def _set_active(self, generation: int) -> None:
        for entry in self.manifest["generations"]:
            if entry["status"] == "active":
                entry["status"] = "retired"
        self._generation_entry(generation)["status"] = "active"
        self.manifest["active_generation"] = generation

    def _activate(self, generation: int) -> None:
        self._load_core(generation)
        self._set_active(generation)
        self._write_manifest()

    # --- persistence -----------------------------------------------------------------
    def _core_path(self, generation: int) -> Path:
        return self.dir / "core" / f"gen-{generation:04d}.json"

    def _write_core(self, generation: int) -> None:
        _atomic_write(self._core_path(generation),
                      json.dumps(core_to_dict(self.reasoner, self.qhead), indent=1))

    def _load_core(self, generation: int) -> None:
        path = self._core_path(generation)
        if not path.exists():
            raise ModelError(f"core weights for generation {generation} are missing")
        core_from_dict(json.loads(path.read_text(encoding="utf-8")), self.reasoner,
                       self.qhead, self.domains)

    def _write_manifest(self) -> None:
        _atomic_write(self.dir / "manifest.json", json.dumps(self.manifest, indent=1))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _new_task_id() -> str:
    return uuid.uuid4().hex[:12]
