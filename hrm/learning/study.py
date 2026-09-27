"""Self-study (loop B driver): the model studies its own knowledge gaps.

The autoresearch pattern turned inward:
    choose    curiosity picks the region to study (learning progress, never tried)
    practice  a generated task with a known answer; the model solves it cold
    judge     the generator's answer is ground truth (external, deterministic)
    learn     wrong -> failure evidence + relearn from the correction;
              abstained -> study the task with its answer as one more example
    sleep     every `sleep_every` tasks: consolidate behind the frozen benchmark
              gate (keep or discard, logged to the ledger)
Damping: task and time budgets; it stops by itself when nothing learnable is
left, and stops for human review after repeated discarded cycles. It refuses
to start if the frozen harness lock is broken.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import numpy as np

from hrm.learning.generators import TaskGenerator
from hrm.learning.sleep import CycleReport
from hrm.metacognition import Curiosity

if TYPE_CHECKING:
    from hrm.model import Model


@dataclass
class StudyBudget:
    max_tasks: int = 50
    max_seconds: float = 600.0
    sleep_every: int = 10
    max_consecutive_discards: int = 3
    seed: int = 0


@dataclass
class StudyReport:
    tasks: int = 0
    correct: int = 0
    wrong: int = 0
    abstained: int = 0
    cycles: List[CycleReport] = field(default_factory=list)
    visited: Dict[str, int] = field(default_factory=dict)
    stopped: str = ""


class SelfStudy:
    def __init__(self, model: "Model", generators: Sequence[TaskGenerator]):
        self.model = model
        self.regions: Dict[str, Tuple[TaskGenerator, str]] = {
            f"{g.domain}/{family}": (g, family) for g in generators for family in g.families}

    def run(self, budget: StudyBudget) -> StudyReport:
        m = self.model
        m.require_harness()
        rng = np.random.default_rng(budget.seed)
        curiosity = Curiosity(m.cfg.meta, rng)
        report = StudyReport()
        started = time.time()
        discards = 0
        while report.tasks < budget.max_tasks:
            if time.time() - started > budget.max_seconds:
                report.stopped = "time budget"
                break
            region = curiosity.choose(m.self_model, list(self.regions))
            if region is None:
                report.stopped = "nothing learnable left"
                break
            outcome = self._practice(region, rng)
            m.self_model.observe(region, outcome)
            m.save_metacognition()
            report.tasks += 1
            report.visited[region] = report.visited.get(region, 0) + 1
            setattr(report, outcome, getattr(report, outcome) + 1)
            if report.tasks % budget.sleep_every == 0:
                cycle = m.sleep()
                report.cycles.append(cycle)
                discards = discards + 1 if cycle.status == "discard" else 0
                if discards >= budget.max_consecutive_discards:
                    report.stopped = "repeated discarded cycles: needs human review"
                    break
        else:
            report.stopped = "task budget"
        return report

    def _practice(self, region: str, rng: np.random.Generator) -> str:
        m = self.model
        generator, family = self.regions[region]
        task = generator.generate(family, rng)
        domain = m.domains.get(task.domain)
        task_id = f"study:{uuid.uuid4().hex[:10]}"
        sol = m.solve(task.demos, task.test_input, task_id, observe=False)
        if sol.solved and domain.equal(sol.answer, task.expected):
            return "correct"
        if sol.solved:
            m.learner.feedback(task_id, "wrong", correction=domain.to_json(task.expected),
                               source="self")
            return "wrong"
        m.learner.teach(list(task.demos) + [(task.test_input, task.expected)],
                        f"{task_id}.studied", provenance="studied")
        return "abstained"
