"""Deepest level: compose new programs when no single hypothesis explains the task.

Bounded beam search: the best partial explanations (by graded energy, since
exactness alone is too flat to search on) are extended by every base
program, so a factor that scores poorly on its own (e.g. `reverse` when the
values are also scaled) is still reachable. Compositions inside a family
closed under composition are skipped: they only re-derive base programs.
Among exact compositions the simplest wins. Only a program that reproduces every demonstration and passes
the verifier is returned; a near miss is a guess and is discarded.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from hrm.config import ReasoningConfig
from hrm.perception.base import Domain, Example, Program
from hrm.verify import VerificationReport, Verifier

EnergyFn = Callable[[Program], Tuple[float, float]]


class Synthesizer:
    def __init__(self, cfg: ReasoningConfig, verifier: Verifier):
        self.beam = cfg.synthesis_beam
        self.depth = cfg.synthesis_depth
        self.verifier = verifier

    def synthesize(self, domain: Domain, demos: Sequence[Example],
                   singles: Sequence[Program], energy_fn: EnergyFn
                   ) -> Optional[Tuple[Program, VerificationReport]]:
        limit = self.verifier.complexity_limit(domain, demos)
        base = [p for p in singles if p.complexity >= 1]
        ranked = sorted(base, key=lambda p: (energy_fn(p)[0], p.complexity))
        frontier: List[Tuple[float, Program]] = [(energy_fn(p)[0], p) for p in ranked[:self.beam]]
        pool = ranked
        if not frontier:
            return None
        best = frontier[0]
        for _ in range(self.depth - 1):
            if best[0] == 0.0:
                break
            grown = []
            for _, head in frontier:
                for tail in pool:
                    if domain.redundant(head, tail):
                        continue
                    comp = domain.compose(head, tail)
                    if comp.complexity <= limit:
                        grown.append((energy_fn(comp)[0], comp))
            frontier = sorted(frontier + grown, key=lambda t: (t[0], t[1].complexity))[:self.beam]
            if frontier[0][0] < best[0]:
                best = frontier[0]
        if best[0] > 0.0:
            return None
        report = self.verifier.verify(domain, best[1], demos)
        return (best[1], report) if report.passed else None
