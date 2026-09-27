"""The independent judge.

Runs the executable face of a program against every demonstration and
decides by exact equality: no learned component participates, so nothing it
audits can game it. A program passes only if it
  * reproduces every demonstration exactly,
  * is deterministic (two runs on the same input agree),
  * is not a memorization (complexity bounded by input size and a hard cap).
MDL (accuracy - alpha * complexity) ranks programs that pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from hrm.config import VerifyConfig
from hrm.perception.base import Domain, Example, Program, ProgramError


@dataclass(frozen=True)
class Evaluation:
    exact: bool         # every demonstration reproduced
    margin: float       # worst per-demo similarity (1.0 iff exact)
    mean: float         # mean per-demo similarity (search guidance)


@dataclass
class VerificationReport:
    passed: bool
    exact: bool
    margin: float
    mdl_score: float
    complexity: int
    reasons: List[str] = field(default_factory=list)


class Verifier:
    def __init__(self, cfg: VerifyConfig):
        self.cfg = cfg

    def evaluate(self, domain: Domain, program: Program,
                 demos: Sequence[Example]) -> Evaluation:
        sims: List[float] = []
        exact = True
        for inp, out in demos:
            try:
                pred = program.apply(inp)
            except ProgramError:
                sims.append(0.0)
                exact = False
                continue
            if not domain.equal(pred, out):
                exact = False
            sims.append(domain.similarity(pred, out))
        if not sims:
            return Evaluation(False, 0.0, 0.0)
        return Evaluation(exact, min(sims), sum(sims) / len(sims))

    def energy(self, domain: Domain, program: Program,
               demos: Sequence[Example]) -> Tuple[float, float]:
        """(semantic energy, verification margin) for the reasoning loop."""
        ev = self.evaluate(domain, program, demos)
        return (0.0, 1.0) if ev.exact else (1.0 - ev.mean, ev.margin)

    def mdl(self, program: Program, accuracy: float) -> float:
        return accuracy - self.cfg.alpha_mdl * program.complexity

    def complexity_limit(self, domain: Domain, demos: Sequence[Example]) -> int:
        """Largest program complexity that cannot be memorization.

        Two bounds, and the second only applies to some domains. A program that
        repays its own description cost cannot be a lookup table, so
        `max_program_complexity` is the hard cap. The relative bound
        (input_size // 4) is an extra guard for domains whose input width grows
        with the task: a longer sequence really can be pattern-matched by
        position, so a program too complex for the input it explains is suspect.

        It is wrong for fixed-width inputs, and the benchmark found it: a 3-vector
        reports size 3 no matter how deeply the program uses it, so three demos
        implied a limit of 2 and every program of depth 3+ was rejected as
        "memorization". The system looked like it could not reason past two
        steps when the guard had forbidden it. Domains set
        `size_bounds_complexity = False` and rely on the hard cap alone.
        """
        hard = self.cfg.max_program_complexity
        if not domain.size_bounds_complexity:
            return hard
        input_size = sum(domain.size(inp) for inp, _ in demos)
        return min(hard, max(2, input_size // 4))

    def verify(self, domain: Domain, program: Program,
               demos: Sequence[Example]) -> VerificationReport:
        reasons: List[str] = []
        ev = self.evaluate(domain, program, demos)
        if not ev.exact:
            reasons.append(f"not exact on all demos (worst similarity {ev.margin:.3f})")
        deterministic = self._deterministic(domain, program, demos)
        if not deterministic:
            reasons.append("non-deterministic program")
        limit = self.complexity_limit(domain, demos)
        memorized = program.complexity > limit
        if memorized:
            reasons.append(f"complexity {program.complexity} > limit {limit}: memorization")
        return VerificationReport(passed=ev.exact and deterministic and not memorized,
                                  exact=ev.exact, margin=ev.margin,
                                  mdl_score=self.mdl(program, ev.margin),
                                  complexity=program.complexity, reasons=reasons)

    @staticmethod
    def _deterministic(domain: Domain, program: Program, demos: Sequence[Example]) -> bool:
        for inp, _ in demos:
            try:
                if not domain.equal(program.apply(inp), program.apply(inp)):
                    return False
            except ProgramError:
                continue
        return True
