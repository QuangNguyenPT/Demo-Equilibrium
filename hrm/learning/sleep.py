"""Sleep cycle (loop B): consolidate proven knowledge into the core, behind a frozen gate.

One cycle, autoresearch-style:
  1. harvest   active knowledge with enough verified uses and a winning record;
  2. propose   raise those programs' priors in the core (knowledge -> neurons);
  3. judge     score the baseline and candidate cores on the frozen benchmark;
  4. keep      gate passes: save a new core generation (live now in 'auto' mode,
               waiting for a human in 'staged' mode); a live core then graduates
               the knowledge it solves without any lookup;
     discard   gate fails: the previous core is restored exactly.
Every cycle is appended to the ledger (ledger/results.tsv).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from hrm.knowledge import KnowledgeItem
from hrm.learning.harness import HarnessResult, gate
from hrm.perception.base import Program
from hrm.reasoning import Reasoner

if TYPE_CHECKING:
    from hrm.model import Model


@dataclass
class CycleReport:
    status: str                              # keep | staged | discard | skip
    generation: int                          # generation live after the cycle
    candidate_generation: Optional[int]      # generation written by this cycle
    consolidated: List[str] = field(default_factory=list)
    graduated: List[int] = field(default_factory=list)
    baseline: Optional[HarnessResult] = None
    candidate: Optional[HarnessResult] = None
    reasons: List[str] = field(default_factory=list)


class SleepCycle:
    def __init__(self, model: "Model"):
        self.model = model

    def run(self) -> CycleReport:
        m = self.model
        harness = m.require_harness()
        problems = m.knowledge.integrity_problems()
        if problems:
            return self._log(CycleReport("discard", m.generation, None, reasons=problems))
        harvest = self._harvest()
        if not harvest:
            return self._log(CycleReport("skip", m.generation, None,
                                         reasons=["no knowledge ready to consolidate"]))

        baseline = harness.evaluate(m.reasoner, m.domains)
        snapshot = m.reasoner.priors.snapshot()
        for item, program in harvest:
            m.reasoner.priors.consolidate(item.domain, program, m.cfg.learning.consolidate_amount)
        candidate = harness.evaluate(m.reasoner, m.domains)
        names = [program.describe() for _, program in harvest]

        reasons = gate(baseline, candidate, m.cfg.learning.regression_tol)
        if reasons:
            m.reasoner.priors.restore(snapshot)
            return self._log(CycleReport("discard", m.generation, None, names, [],
                                         baseline, candidate, reasons))
        live = m.cfg.learning.promotion == "auto"
        generation = m.new_generation(f"consolidated {len(harvest)} item(s)", candidate, live)
        graduated: List[int] = []
        if live:
            graduated = self.graduate([item for item, _ in harvest])
        else:
            m.reasoner.priors.restore(snapshot)       # staged: the live core stays as it was
        return self._log(CycleReport("keep" if live else "staged", m.generation, generation,
                                     names, graduated, baseline, candidate))

    def graduate(self, items: List[KnowledgeItem]) -> List[int]:
        """Retire from SSD retrieval what the live core now solves without a lookup."""
        m = self.model
        masked = Reasoner(m.cfg, m.cb, m.domains, knowledge=None, priors=m.reasoner.priors,
                          qhead=m.qhead)
        masked.engine = m.reasoner.engine            # same learned search scalars
        graduated = []
        for item in items:
            try:
                episode = m.knowledge.episode(item.origin)
            except KeyError:
                continue                             # no record of how it was learned
            domain = m.domains.get(episode.domain)
            demos = [(domain.from_json(i), domain.from_json(o)) for i, o in episode.demos]
            sol = masked.solve(demos, demos[-1][0], f"graduation:{item.id}", domain=domain)
            if sol.solved and domain.equal(sol.answer, demos[-1][1]):
                m.knowledge.set_status(item.id, "graduated", generation=m.generation)
                graduated.append(item.id)
        return graduated

    def _harvest(self) -> List[Tuple[KnowledgeItem, Program]]:
        m = self.model
        ready = []
        for item in m.knowledge.items("active"):
            if item.successes < m.cfg.learning.min_successes or item.failures >= item.successes:
                continue
            try:
                program = m.domains.get(item.domain).program_from_payload(item.payload)
            except (KeyError, TypeError, ValueError):
                continue
            ready.append((item, program))
        return ready

    def _log(self, report: CycleReport) -> CycleReport:
        result = report.candidate or report.baseline
        self.model.append_ledger(
            generation=report.candidate_generation if report.candidate_generation is not None
            else report.generation,
            status=report.status,
            solve_rate=result.solve_rate if result else None,
            false_accepts=result.false_accepts if result else None,
            description="; ".join(report.consolidated[:5] + report.reasons[:3]))
        return report
