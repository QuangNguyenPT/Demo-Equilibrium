"""Online learning (loop A): runs after every interaction.

  * records the episode and the answer, so feedback can later find what was
    answered and which knowledge produced it;
  * writes only verified knowledge (the store refuses anything else);
  * turns evidence into confidence: self-evidence when a stored program
    solves a new task, human evidence through feedback.
Human feedback outranks self-evidence: a human "wrong" quarantines the
knowledge that produced the answer and drops it from the core's priors; only a
human can reinstate it. Self feedback (a study task's known answer) only adds
failure evidence and relearns from the correction: it never quarantines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from hrm.knowledge import KnowledgeRecord, KnowledgeStore
from hrm.perception.base import DomainRegistry, Example
from hrm.reasoning import Reasoner, Solution

VERDICTS = ("correct", "wrong")


@dataclass
class FeedbackResult:
    task_id: str
    verdict: str
    quarantined: List[int] = field(default_factory=list)
    forgotten_prior: bool = False
    relearned: Optional[Solution] = None      # re-solve with the correction as a new demo


class Learner:
    def __init__(self, knowledge: KnowledgeStore, reasoner: Reasoner, domains: DomainRegistry):
        self.knowledge = knowledge
        self.reasoner = reasoner
        self.domains = domains

    def after_solve(self, sol: Solution, task_id: str, demos: Sequence[Example],
                    test_input: Any) -> Optional[int]:
        """Record the interaction; store a newly synthesized explanation. Returns its id."""
        used = self._record(sol, task_id, demos, test_input)
        for item_id in used:
            self.knowledge.report_outcome(item_id, True, sol.report.margin, task_id, "self")
        if sol.status == "synthesized":
            return self._remember(sol, task_id, "synthesized")
        return None

    def teach(self, demos: Sequence[Example], task_id: str,
              provenance: str = "taught") -> Solution:
        """Explain the given demonstrations and keep the explanation if it verifies."""
        demos = list(demos)
        sol = self.reasoner.solve(demos, demos[-1][0], task_id)
        used = self._record(sol, task_id, demos, demos[-1][0])
        for item_id in used:            # already known: the lesson is evidence, not a copy
            self.knowledge.report_outcome(item_id, True, sol.report.margin, task_id, "self")
        if sol.solved and not used:
            self._remember(sol, task_id, provenance)
        return sol

    def feedback(self, task_id: str, verdict: str, correction: Any = None,
                 source: str = "human") -> FeedbackResult:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        if source not in ("human", "self"):
            raise ValueError("feedback source must be 'human' or 'self'")
        if verdict == "correct" and correction is not None:
            raise ValueError("a correction implies the answer was wrong")
        answer = self.knowledge.answer(task_id)
        episode = self.knowledge.episode(task_id)
        self.knowledge.set_verdict(task_id, verdict)
        result = FeedbackResult(task_id, verdict)
        right = verdict == "correct"
        authoritative = source == "human"
        for item_id in answer.items:
            self.knowledge.report_outcome(item_id, right, 1.0 if right else 0.0, task_id, source)
            if not right and authoritative:
                self.knowledge.set_status(item_id, "quarantined")
                result.quarantined.append(item_id)
        if not right and authoritative and answer.program is not None:
            result.forgotten_prior = self.reasoner.priors.forget(episode.domain, answer.program)
        if correction is not None:
            domain = self.domains.get(episode.domain)
            demos = [(domain.from_json(i), domain.from_json(o)) for i, o in episode.demos]
            demos.append((domain.from_json(episode.test_input), domain.from_json(correction)))
            result.relearned = self.teach(demos, f"{task_id}.corrected", provenance="corrected")
        return result

    # --- internals -------------------------------------------------------------------
    def _record(self, sol: Solution, task_id: str, demos: Sequence[Example],
                test_input: Any) -> List[int]:
        domain = self.domains.get(sol.domain)
        self.knowledge.record_episode(
            task_id, sol.domain, [[domain.to_json(i), domain.to_json(o)] for i, o in demos],
            domain.to_json(test_input))
        used = self._used_items(sol)
        self.knowledge.record_answer(
            task_id, sol.status, sol.program.describe() if sol.program else None,
            sol.program.family if sol.program else None,
            domain.to_json(sol.answer) if sol.solved else None, sol.confidence, used)
        return used

    def _used_items(self, sol: Solution) -> List[int]:
        if not sol.solved:
            return []
        description = sol.program.describe()
        return [i for i in sol.retrieved if self.knowledge.get(i).signature == description]

    def _remember(self, sol: Solution, task_id: str, provenance: str) -> Optional[int]:
        program = sol.program
        record = KnowledgeRecord(kind="program", domain=sol.domain,
                                 signature=program.describe(), payload=program.to_payload(),
                                 complexity=program.complexity, origin=task_id,
                                 provenance=provenance)
        return self.knowledge.write(sol.task_key, record, sol.report)
