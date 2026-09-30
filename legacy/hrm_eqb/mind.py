"""The running model: core (.hrmc) + knowledge (knowledge.pt) + hormones, answering or asking.

ask(task):
  1. hormones set the segment budget (serotonin) and routing gain (noradrenaline);
  2. a committee of augmented views solves the task; answers are de-augmented and tallied;
  3. leave-one-out over the demonstrations measures E_loo (demos the model cannot re-derive);
  4. unanimous, E_loo = 0 and confident -> Answer; otherwise -> Question about the query,
     with the committee's candidates and the information the answer would give (Q1).
learn(task, answer):
  the answered query joins the demonstrations; knowledge rows read for this task get a
  test-time update; the update is written to SSD only if every demonstration is then
  re-derived (E_loo = 0) and a fixed canary set does not get worse (spec §14, V1).
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from hrm_eqb import hrmc
from hrm_eqb import knowledge as kn
from hrm_eqb.curiosity import entropy_bits, tally
from hrm_eqb.device import select_device
from hrm_eqb.field import (IDENTITY, Augmentation, answers_equal, budget_chunks, collate, decode_answer,
                           encode_task, views)
from hrm_eqb.model import HRMEQB
from hrm_eqb.neuromodulation import Neuromodulators, drive
from hrm_eqb.tasks import FAMILIES, Task, generate
from hrm_eqb.train import EVAL_PAIRS

CORE, KNOWLEDGE, STATE = "model.hrmc", "knowledge.pt", "state.json"
HORMONE_BOX = ((0.05, 0.3, 0.2, 0.05), (0.95, 0.8, 0.9, 0.9))
HORMONE_RATE = 0.2
CONFIDENT = 0.5
CANARY_SEED, CANARY_SIZE = 777, 32


@dataclass
class Answer:
    grid: np.ndarray
    confidence: float
    votes: int
    views: int
    certified_error: float
    segments: int
    description: str
    kind: str = "answer"


@dataclass
class Question:
    query: np.ndarray
    candidates: List[Tuple[np.ndarray, int]]
    bits: float
    reasons: List[str]
    confidence: float
    description: str
    kind: str = "question"


@dataclass
class LearnReport:
    accepted: bool
    rows: int
    loo_before: int
    loo_after: int
    canary_before: int
    canary_after: int
    reason: str


@dataclass
class _Run:
    answers: List[np.ndarray] = field(default_factory=list)
    halts: List[List[float]] = field(default_factory=list)
    bounds: List[float] = field(default_factory=list)
    segments: int = 0
    description: str = ""


def leave_one_out(task: Task) -> List[Task]:
    """One task per demonstration: that demo becomes the query, the others stay demos."""
    if len(task.demos) < 2:
        return []
    return [Task(task.domain, task.family, task.demos[:j] + task.demos[j + 1:], x, y, task.description)
            for j, (x, y) in enumerate(task.demos)]


def canary_tasks() -> List[Task]:
    rng = np.random.default_rng(CANARY_SEED)
    fams = [(d, f) for d in ("grid", "sequence", "text") for f in FAMILIES[d]]
    return [generate(*fams[i % len(fams)], rng, max_side=6) for i in range(CANARY_SIZE)]


class Mind:
    """A model directory brought to life on one device."""

    def __init__(self, directory: Path, device: Optional[torch.device] = None):
        self.dir = Path(directory)
        self.device = device or select_device()
        self.model, self.header = hrmc.load(self.dir / CORE)
        self.kfile = kn.KnowledgeFile(self.dir / KNOWLEDGE)
        if self.kfile.schema != self.header["knowledge"]["schema"]:
            raise ValueError(f"knowledge schema {self.kfile.schema} does not match the core")
        self.model.memory.attach(self.kfile.fixed["keys_a"], self.kfile.fixed["keys_b"], self.kfile.tensor("values"))
        self.model.to(self.device).eval().requires_grad_(False)
        self.state = json.loads((self.dir / STATE).read_text(encoding="utf-8")) if (self.dir / STATE).exists() else {}
        self.hormones = Neuromodulators.create(*HORMONE_BOX, rate=HORMONE_RATE)
        if "hormones" in self.state:
            self.hormones.load_state(self.state["hormones"])
        self.rng = np.random.default_rng(self.state.get("episodes", 0))

    @staticmethod
    def create(directory: Path, model: HRMEQB, lineage: Optional[Dict] = None) -> str:
        """Write model.hrmc + knowledge.pt + state.json for a trained model; returns the core SHA-256."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sha = hrmc.save(model, directory / CORE, lineage)
        cfg, know = model.cfg, model.memory.knowledge()
        kn.create(directory / KNOWLEDGE, know["keys_a"], know["keys_b"], know["values"],
                  kn.schema_of(cfg.memory_side, cfg.key_dim, cfg.value_quaternions))
        state = {"generation": (lineage or {}).get("generation", 0), "core_sha256": sha, "episodes": 0,
                 "writes": 0, "hormones": Neuromodulators.create(*HORMONE_BOX, rate=HORMONE_RATE).state()}
        (directory / STATE).write_text(json.dumps(state, indent=2), encoding="utf-8")
        return sha

    def close(self) -> None:
        self.save_state()
        self.model.memory.attach(self.model.memory.keys_a, self.model.memory.keys_b,
                                 self.model.memory.values.clone())
        self.kfile.close()

    def save_state(self) -> None:
        self.state["hormones"] = self.hormones.state()
        (self.dir / STATE).write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    # ---------- solving ----------

    @torch.no_grad()
    def _solve(self, tasks: Sequence[Task], segments: int, describe: bool = False) -> List[_Run]:
        """Run each task (identity view) through the model, in attention-pair-budgeted batches."""
        encoded = [encode_task(t) for t in tasks]
        runs: List[_Run] = []
        for group in budget_chunks(encoded, EVAL_PAIRS):
            runs += self._solve_batch([encoded[i] for i in group], segments, describe)
        return runs

    def _solve_batch(self, encoded: List, segments: int, describe: bool) -> List[_Run]:
        batch = collate(encoded).to(self.device)
        outs = self.model(batch, segments=segments, route_gain=self.hormones.route_gain())
        ids = outs[-1].logits.argmax(-1).cpu().numpy()
        halts = torch.stack([torch.sigmoid(o.halt) for o in outs], dim=1).cpu().numpy()
        texts = [""] * len(encoded)
        if describe:
            answer_mask = torch.zeros_like(batch.valid)
            answer_mask.scatter_(1, batch.answer_pos, batch.answer_valid)
            memory = self.model.coda.memory([o.high for o in outs], batch.valid, answer_mask)
            texts = self.model.coda.describe(memory, self.model.coda.memory_mask(batch.valid))
        runs = []
        for i, enc in enumerate(encoded):
            runs.append(_Run([decode_answer(enc, ids[i, :enc.answer_pos.size])], [list(halts[i])],
                             [outs[-1].error_bound], len(outs), texts[i]))
        return runs

    def ask(self, task: Task, n_views: int = 4) -> Union[Answer, Question]:
        segments = min(self.model.cfg.max_segments, self.hormones.segment_budget())
        augs = views(task, n_views, self.rng)
        loo = leave_one_out(task)
        members = [a.apply(task) if task.kind == "grid" else task for a in augs]
        runs = self._solve(members + loo, segments, describe=True)
        answers = [_undo(a, r.answers[0], task) for a, r in zip(augs, runs[:len(augs)])]
        tallied = tally(answers)
        bits = entropy_bits([c for _, c in tallied])
        unexplained = sum(not answers_equal(r.answers[0], t.answer) for r, t in zip(runs[len(augs):], loo))
        halts = runs[0].halts[0]
        confidence = float(np.mean([r.halts[0][-1] for r in runs[:len(augs)]]))
        self._feel(None, halts[-1] - halts[0], bits, math.log2(max(len(augs), 2)), unexplained / max(len(loo), 1))
        self.state["episodes"] = self.state.get("episodes", 0) + 1
        reasons = []
        if bits > 0:
            reasons.append(f"the {len(augs)} views disagree ({bits:.2f} bits)")
        if unexplained:
            reasons.append(f"{unexplained} of {len(loo)} demonstrations cannot be re-derived")
        if confidence < CONFIDENT:
            reasons.append(f"low confidence ({confidence:.2f})")
        if reasons:
            return Question(task.query, tallied[:2], bits, reasons, confidence, runs[0].description)
        best, votes = tallied[0]
        return Answer(best, confidence, votes, len(augs), max(r.bounds[0] for r in runs[:len(augs)]),
                      runs[0].segments, runs[0].description)

    def _feel(self, reward_error: Optional[float], gain: float, bits: float, max_bits: float, unexplained: float) -> None:
        self.hormones.step(drive(reward_error, gain, bits, max_bits, unexplained, self.hormones.level))

    # ---------- learning (test-time, verified, written to SSD) ----------

    def _loss_and_errors(self, tasks: Sequence[Task], segments: int) -> Tuple[torch.Tensor, int]:
        """Mean answer loss and count of wrong tasks; callers keep `tasks` small (one task's demos)."""
        encoded = [encode_task(t) for t in tasks]
        batch = collate(encoded).to(self.device)
        outs = self.model(batch, segments=segments, route_gain=self.hormones.route_gain())
        logits = outs[-1].logits
        ce = F.cross_entropy(logits.transpose(1, 2), batch.target, reduction="none")
        loss = (ce * batch.answer_valid).sum() / batch.answer_valid.sum()
        wrong = ((logits.argmax(-1) != batch.target) & batch.answer_valid).any(dim=1)
        return loss, int(wrong.sum())

    def learn(self, task: Task, answer: np.ndarray, steps: int = 30, base_lr: float = 0.05) -> LearnReport:
        """Learn from a correct answer to `task` by a verified knowledge write."""
        taught = Task(task.domain, task.family, task.demos + [(task.query, np.asarray(answer))], task.query,
                      np.asarray(answer), task.description)
        loo = leave_one_out(taught)
        if not loo:
            return LearnReport(False, 0, 0, 0, 0, 0, "need at least two demonstrations to verify a write")
        segments = min(self.model.cfg.max_segments, self.hormones.segment_budget())
        canaries = canary_tasks()
        prior = self.ask(task) if task.demos else None
        predicted = prior.grid if isinstance(prior, Answer) else None
        with torch.no_grad():
            _, loo_before = self._loss_and_errors(loo, segments)
            _, canary_wrong_before = self._loss_and_errors(canaries, segments)
        mem = self.model.memory
        overlay = torch.zeros(mem.slots, 4 * mem.value_quaternions, device=self.device, requires_grad=True)
        mem.overlay = overlay
        opt = torch.optim.Adam([overlay], lr=self.hormones.learning_rate(base_lr))
        try:
            for _ in range(steps):
                loss, _ = self._loss_and_errors(loo, segments)
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                _, loo_after = self._loss_and_errors(loo, segments)
                _, canary_wrong_after = self._loss_and_errors(canaries, segments)
            rows = torch.nonzero(overlay.detach().abs().sum(dim=1) > 0).flatten().cpu().numpy()
            ok = loo_after == 0 and canary_wrong_after <= canary_wrong_before and rows.size > 0
            reason = "verified" if ok else (f"E_loo {loo_after} after the write" if loo_after else
                                            "the canary set got worse" if canary_wrong_after > canary_wrong_before
                                            else "no knowledge row was read")
            if ok:
                self._commit(rows, overlay.detach()[torch.as_tensor(rows, device=self.device)].cpu().numpy())
        finally:
            mem.overlay = None
        correct = predicted is not None and answers_equal(predicted, answer)
        confidence = prior.confidence if prior is not None else 0.5
        self._feel(float(correct) - confidence, 0.0, 0.0, 1.0, loo_before / len(loo))
        self.save_state()
        return LearnReport(ok, int(rows.size), loo_before, loo_after, canary_wrong_before, canary_wrong_after, reason)

    def _commit(self, rows: np.ndarray, delta: np.ndarray) -> None:
        maps = self.kfile.maps
        generation = int(self.state.get("generation", 0))
        self.kfile.write_rows(rows, {
            "values": np.asarray(maps["values"][rows]) + delta,
            "usage": np.asarray(maps["usage"][rows]) + 1.0,
            "confidence": np.minimum(1.0, np.asarray(maps["confidence"][rows]) + self.hormones.get("dopamine") * 0.1),
            "generation": np.full(rows.size, generation, dtype=np.int64),
        })
        self.state["writes"] = self.state.get("writes", 0) + 1


def _undo(aug: Augmentation, answer: np.ndarray, task: Task) -> np.ndarray:
    if task.kind != "grid" or aug == IDENTITY:
        return answer
    if answer.size == 0 or (answer < 0).any() or (answer > 9).any():
        return answer
    return aug.inverse(answer)


def task_from_json(data: Dict) -> Task:
    """{"domain": grid|sequence|text|sudoku, "train": [{"input","output"}], "test": [{"input"[, "output"]}]}.

    Text values may be strings; sequences may be flat lists.
    """
    demos = [(as_array(p["input"]), as_array(p["output"])) for p in data.get("train", [])]
    test = data["test"][0]
    return Task(data.get("domain", "grid"), data.get("family", "user"), demos, as_array(test["input"]),
                as_array(test["output"]) if "output" in test else None, data.get("description", ""))


def as_array(value) -> np.ndarray:
    """JSON value -> 2-D canvas: strings become one row of bytes, flat lists one row."""
    if isinstance(value, str):
        return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).astype(np.int64)[None, :]
    a = np.asarray(value, dtype=np.int64)
    return a[None, :] if a.ndim == 1 else a
