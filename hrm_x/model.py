"""HRM-X v0.6 as one trainable, runnable model: config, train, checkpoint, answer, talk, evaluate.

    input text ──► language.parse ──► evidence ((x, y), z) per operator symbol
                   (numerals: DecimalReader, or the learned place-value reader)
                                   │
                                   ▼
    Mind (H level, value of computation) over the Concept domain (L level)
        propose level 0   exact fit: 68 written families + admitted stored concepts
        propose level k   construction depth k (3 / 5 / 7 tokens), stored concepts as primitives
        derive · verify (dual solve Aᵀλ = r) · ask (h-question horizon when OUTSIDE holds mass)
                                   │
                                   ▼
    answer / abstain (with reasons) ──► concept memory: candidate → admitted → (retired)
                                   │
                                   ▼
    language.explain ──► text

"Training" is experience: a curriculum of tasks is solved with a teacher, verified constructions
become concepts, and a second independent verification admits them. There are no gradient-trained
weights in the reasoning path; the only gradient-trained component is the optional numeral reader
(`learn_notation`), and it is accepted only after its snapped parameters reproduce every training
sum exactly. A checkpoint is the directory: config.json, mind.pkl, concepts.json (readable copy),
reader.json, train_log.jsonl. Training resumes from the step recorded in the log.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hrm_x.concept_memory import ConceptMemory
from hrm_x.concepts import ConceptConfig
from hrm_x.construct import ConstructConfig
from hrm_x.curriculum import Curriculum
from hrm_x.domains.concept import Concept, ConceptTask
from hrm_x.language import Conversation, DecimalReader, LearnedReader
from hrm_x.mind import Controller, Mind, Result

Q = Fraction


@dataclass
class ModelConfig:
    """Every number that changes behaviour; unknown keys are refused on load."""

    bits_bottom: float = 4.0
    log2_answer_space: float = 10.0
    answer_space: str = "empirical"
    construct_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    construct: bool = True
    use_memory: bool = True
    use_library_tokens: bool = True
    wrong_penalty: float = 1.0
    ask_horizon: int = 3
    use_questions: bool = True
    lam: float = 2e-5

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelConfig":
        known = set(ModelConfig.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return ModelConfig(**d)


class HRMX:
    """The whole model; one instance owns one Mind (semantic memory, statistics, modulators)."""

    def __init__(self, cfg: Optional[ModelConfig] = None, mind: Optional[Mind] = None,
                 reader: Any = None) -> None:
        self.cfg = cfg or ModelConfig()
        self.mind = mind or Mind(Controller(lam=self.cfg.lam, wrong_penalty=self.cfg.wrong_penalty,
                                            ask_horizon=self.cfg.ask_horizon,
                                            use_questions=self.cfg.use_questions))
        self.reader = reader or DecimalReader()
        self.domain = self.new_domain()
        self.chat = Conversation(self.mind, self.domain, self.reader)
        self.step = 0

    def new_domain(self) -> Concept:
        c = self.cfg
        construct = ConstructConfig(sizes=tuple(c.construct_sizes), use_library=c.use_library_tokens) \
            if c.construct else None
        return Concept(ConceptConfig(bits_bottom=c.bits_bottom, log2_answer_space=c.log2_answer_space,
                                     answer_space=c.answer_space),
                       construct=construct, use_memory=c.use_memory)

    # ----- inference -----
    def solve(self, task: ConceptTask) -> Result:
        return self.mind.solve(self.domain, task)

    def say(self, text: str) -> str:
        return self.chat.say(text)

    @property
    def memory(self) -> ConceptMemory:
        return ConceptMemory.of(self.mind)

    # ----- training -----
    def train(self, curriculum: Curriculum, max_tasks: Optional[int] = None, checkpoint: Optional[str] = None,
              every: int = 10, log: Any = print) -> Dict[str, Any]:
        """Solve curriculum tasks with the teacher; memory grows from verified constructions."""
        tally: Dict[str, Dict[str, int]] = {}
        start = time.time()
        for i, sample in enumerate(curriculum):
            if i < self.step:
                continue                                 # resume: tasks before the checkpoint are done
            if max_tasks is not None and i >= max_tasks:
                break
            result = self.solve(sample.task)
            outcome = ("abstain" if result.status != "answer" else
                       "correct" if result.answer == sample.truth else "wrong")
            if outcome != "abstain":
                self.mind.feedback(self.domain.name, outcome == "correct")
            t = tally.setdefault(sample.pool, {"correct": 0, "wrong": 0, "abstain": 0, "ops": 0})
            t[outcome] += 1
            t["ops"] += result.ops
            self.step = i + 1
            row = {"step": self.step, "pool": sample.pool, "op": sample.name, "outcome": outcome,
                   "ops": result.ops, "concepts": len(self.memory.concepts),
                   "admitted": len(self.memory.usable())}
            if checkpoint:
                Path(checkpoint).mkdir(parents=True, exist_ok=True)
                with open(Path(checkpoint) / "train_log.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                if self.step % every == 0:
                    self.save(checkpoint)
            if log is not None:
                log(f"[TRAIN] {self.step:4d} {sample.pool:13s} {sample.name:12s} {outcome:8s} "
                    f"ops {result.ops:7d} concepts {row['concepts']} admitted {row['admitted']}")
        if checkpoint:
            self.save(checkpoint)
        return {"tally": tally, "concepts": [self.memory.text(k) for k in range(len(self.memory.concepts))],
                "admitted": [self.memory.text(k) for k in self.memory.usable()],
                "seconds": round(time.time() - start, 1)}

    def learn_notation(self, sums: Sequence[tuple], seed: int = 0) -> bool:
        """Learn a numeral notation from written sums (needs torch); keep it only if verified exact."""
        reader = LearnedReader.learn(sums, seed)
        if reader is None:
            return False
        self.reader = reader
        self.chat.reader = reader
        return True

    # ----- checkpoints -----
    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")
        self.mind.save(str(d / "mind.pkl"))
        (d / "concepts.json").write_text(json.dumps(
            [{"index": k, "name": c.name, "expr": self.memory.text(k), "status": c.status, "tasks": c.tasks,
              "successes": c.successes, "failures": c.failures} for k, c in enumerate(self.memory.concepts)],
            indent=2), encoding="utf-8")
        reader = {"kind": "decimal"} if isinstance(self.reader, DecimalReader) else \
            {"kind": "learned", "theta": self.reader.theta, "base": self.reader.base}
        (d / "reader.json").write_text(json.dumps(reader), encoding="utf-8")
        (d / "state.json").write_text(json.dumps({"step": self.step}), encoding="utf-8")

    @staticmethod
    def load(directory: str) -> "HRMX":
        d = Path(directory)
        cfg = ModelConfig.from_dict(json.loads((d / "config.json").read_text(encoding="utf-8")))
        mind = Mind.load(str(d / "mind.pkl"))
        r = json.loads((d / "reader.json").read_text(encoding="utf-8")) if (d / "reader.json").exists() else {}
        reader = LearnedReader(r["theta"], r["base"]) if r.get("kind") == "learned" else DecimalReader()
        model = HRMX(cfg, mind, reader)
        if (d / "state.json").exists():
            model.step = json.loads((d / "state.json").read_text(encoding="utf-8"))["step"]
        return model

    def info(self) -> Dict[str, Any]:
        mem = self.memory
        return {"config": asdict(self.cfg), "step": self.step,
                "concepts": [{"name": c.name, "expr": mem.text(k), "status": c.status, "tasks": len(c.tasks)}
                             for k, c in enumerate(mem.concepts)]}
