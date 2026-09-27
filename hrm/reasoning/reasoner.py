"""The reasoning section: fast recognition, then deep equilibrium, then synthesis.

    fast         L-level recognition. Rank candidates by how well their signature
                 matches the task core plus consolidated priors plus retrieved
                 knowledge; verify the top few without iterating. Cheap, and
                 still verified: fast does not mean unexamined.
    language     inner speech: the language section proposes rules in words
                 (the domain's program language); proposals are verified like
                 any other hypothesis and join the candidate set.
    deep         H-level equilibrium: resonator ticks inside refresh cycles,
                 restarts that swap parse hypotheses.
    synthesized  compose new programs by bounded search.
    abstained    nothing reproduced the demonstrations: the model says it does
                 not know instead of guessing.

The reasoner only reads knowledge. Writing what was learned is the learning
section's job, so inference can never corrupt the store.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hrm.config import ModelConfig
from hrm.knowledge import KnowledgeStore, Retrieved
from hrm.perception.base import Domain, DomainRegistry, Example, Program, ProgramError
from hrm.reasoning.equilibrium import EquilibriumEngine
from hrm.reasoning.priors import PriorTable
from hrm.reasoning.qhead import QHead
from hrm.reasoning.resonator import CandidateSet
from hrm.reasoning.synthesis import Synthesizer
from hrm.substrate import Codebook, PhasorVector
from hrm.trace import Trace, TraceEvent
from hrm.verify import VerificationReport, Verifier

if TYPE_CHECKING:
    from hrm.language import InnerSpeech

log = logging.getLogger("hrm.reasoning")


@dataclass
class Solution:
    status: str
    answer: Any
    second_guess: Any
    program: Optional[Program]
    report: Optional[VerificationReport]
    confidence: float                   # agreement of exact explanations, MDL-discounted
    domain: str
    ticks: int                          # deep-level iterations spent
    evaluations: int                    # distinct programs executed against the demos
    task_key: PhasorVector
    retrieved: List[int] = field(default_factory=list)   # knowledge ids offered as candidates
    trace: Trace = field(default_factory=Trace)

    @property
    def solved(self) -> bool:
        return self.status != "abstained"


class Reasoner:
    def __init__(self, cfg: ModelConfig, cb: Codebook, domains: DomainRegistry,
                 knowledge: Optional[KnowledgeStore] = None,
                 priors: Optional[PriorTable] = None, qhead: Optional[QHead] = None,
                 speech: Optional["InnerSpeech"] = None):
        self.cfg = cfg.reasoning
        self.alpha_mdl = cfg.verify.alpha_mdl
        self.language_prior = cfg.language.language_prior
        self.cb = cb
        self.domains = domains
        self.knowledge = knowledge
        self.speech = speech
        self.priors = priors if priors is not None else PriorTable()
        self.verifier = Verifier(cfg.verify)
        self.engine = EquilibriumEngine(cfg.reasoning, cfg.substrate.seed, qhead)
        self.synthesizer = Synthesizer(cfg.reasoning, self.verifier)

    def solve(self, demos: Sequence[Example], test_input: Any, task_id: str = "task",
              domain: Optional[Domain] = None) -> Solution:
        demos = list(demos)
        if not demos:
            raise ValueError("at least one demonstration is required")
        domain = domain or self.domains.for_demos(demos)
        trace = Trace(task_id)
        evaluated: Dict[str, Tuple[float, float, Program]] = {}

        def energy_fn(program: Program) -> Tuple[float, float]:
            key = program.describe()
            if key not in evaluated:
                energy, margin = self.verifier.energy(domain, program, demos)
                evaluated[key] = (energy, margin, program)
            return evaluated[key][:2]

        task = domain.encode_task(demos, 0)
        key = domain.task_key(task)
        retrieved = (self.knowledge.retrieve(key, domain=domain.name)
                     if self.knowledge is not None else [])
        cands = self._candidates(domain, demos, retrieved)
        done = _Done(self, domain, demos, test_input, key, [r.item.id for r in retrieved],
                     trace, evaluated, energy_fn, cands)

        found = self._fast(domain, demos, task, cands, energy_fn, trace)
        if found is not None:
            return done("fast", *found, ticks=0)

        if self.speech is not None:
            proposals, said = self.speech.propose(domain, demos)
            trace.log(TraceEvent(0, "inner_speech", detail={
                "said": said[:500], "proposals": [p.describe() for p in proposals]}))
            for program in proposals:
                cands.raise_prior(program, self.language_prior, self.cb)
            found = self._best_verified(domain, demos, proposals, energy_fn, trace, "language")
            if found is not None:
                return done("language", *found, ticks=0)

        result = self.engine.run(task, cands, energy_fn, trace,
                                 on_restart=lambda t, failed: self._reparse(domain, demos, t,
                                                                            failed, cands))
        if result.status == "accept":
            report = self.verifier.verify(domain, result.program, demos)
            if report.passed:
                return done("deep", result.program, report, ticks=result.ticks)

        synth = self.synthesizer.synthesize(domain, demos, result.cands.programs, energy_fn)
        if synth is not None:
            trace.log(TraceEvent(result.ticks, "synthesis", synth[0].describe(), energy=0.0))
            return done("synthesized", *synth, ticks=result.ticks)
        trace.log(TraceEvent(result.ticks, "abstain", energy=result.energy,
                             detail={"reason": "no program reproduces the demonstrations"}))
        return done.abstain(result.ticks)

    # --- candidate assembly --------------------------------------------------------
    def _candidates(self, domain: Domain, demos: Sequence[Example],
                    retrieved: Sequence[Retrieved]) -> CandidateSet:
        cands = CandidateSet.build(domain.candidates(demos), self.cb)
        for program, prior in self.priors.programs(domain):
            cands.raise_prior(program, prior, self.cb)
        for r in retrieved:
            try:
                program = domain.program_from_payload(r.item.payload)
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("[REASONER] skipping undecodable knowledge item %d: %s", r.item.id, exc)
                continue
            cands.raise_prior(program, self.cfg.memory_prior * r.item.confidence, self.cb)
        return cands

    def _fast(self, domain: Domain, demos: Sequence[Example], task, cands: CandidateSet,
              energy_fn, trace: Trace) -> Optional[Tuple[Program, VerificationReport]]:
        scores = cands.similarities(task.shared.data) + cands.priors
        top = [cands.programs[int(i)] for i in np.argsort(-scores)[: self.cfg.fast_top_k]]
        return self._best_verified(domain, demos, top, energy_fn, trace, "fast")

    def _best_verified(self, domain: Domain, demos: Sequence[Example],
                       programs: Sequence[Program], energy_fn, trace: Trace,
                       kind: str) -> Optional[Tuple[Program, VerificationReport]]:
        """Simplest (best MDL) program among `programs` that passes verification."""
        best: Optional[Tuple[Program, VerificationReport]] = None
        for program in programs:
            energy, _ = energy_fn(program)
            trace.log(TraceEvent(0, kind, program.describe(), energy=energy))
            if energy == 0.0:
                report = self.verifier.verify(domain, program, demos)
                if report.passed and (best is None or report.mdl_score > best[1].mdl_score):
                    best = (program, report)
        return best

    def _reparse(self, domain: Domain, demos: Sequence[Example], task, failed: List[int],
                 cands: CandidateSet):
        parse = self.engine.restarts.pick_parse(domain.n_parses, task.parse, failed)
        if parse is None:
            return None
        return domain.encode_task(demos, parse), cands


class _Done:
    """Builds the Solution: executes the chosen program on the test input and
    measures how many other exact explanations agree with its answer. The whole
    candidate vocabulary is checked, not only what the search happened to
    visit, so an ambiguous task cannot look certain."""

    def __init__(self, reasoner: Reasoner, domain: Domain, demos, test_input, key,
                 retrieved: List[int], trace: Trace, evaluated, energy_fn,
                 cands: CandidateSet):
        self.r, self.domain, self.demos, self.test_input = reasoner, domain, demos, test_input
        self.key, self.retrieved, self.trace, self.evaluated = key, retrieved, trace, evaluated
        self.energy_fn, self.cands = energy_fn, cands

    def __call__(self, status: str, program: Program, report: VerificationReport,
                 ticks: int) -> Solution:
        try:
            answer = program.apply(self.test_input)
        except ProgramError as exc:
            self.trace.log(TraceEvent(ticks, "abstain", program.describe(),
                                      detail={"reason": f"explanation does not apply to the "
                                                        f"test input: {exc}"}))
            return self.abstain(ticks)
        for candidate in self.cands.programs:
            self.energy_fn(candidate)
        second, agreement = self._alternatives(program, answer)
        confidence = agreement / (1.0 + self.r.alpha_mdl * max(program.complexity - 1, 0))
        return Solution(status, answer, second, program, report, confidence, self.domain.name,
                        ticks, len(self.evaluated), self.key, self.retrieved, self.trace)

    def abstain(self, ticks: int) -> Solution:
        return Solution("abstained", None, None, None, None, 0.0, self.domain.name, ticks,
                        len(self.evaluated), self.key, self.retrieved, self.trace)

    def _alternatives(self, chosen: Program, answer: Any) -> Tuple[Any, float]:
        """Other exact, non-memorizing explanations: do they predict the same answer?"""
        limit = self.r.verifier.complexity_limit(self.domain, self.demos)
        alternatives = sorted((p for energy, _, p in self.evaluated.values()
                               if energy == 0.0 and p.describe() != chosen.describe()
                               and p.complexity <= limit), key=lambda p: p.complexity)
        agree, applicable, second = 1, 1, None
        for p in alternatives:
            try:
                pred = p.apply(self.test_input)
            except ProgramError:
                continue
            applicable += 1
            if self.domain.equal(pred, answer):
                agree += 1
            elif second is None:
                second = pred
        return second, agree / applicable
