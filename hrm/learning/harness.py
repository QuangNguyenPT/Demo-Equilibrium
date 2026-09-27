"""The frozen judge of the learning loop: a fixed benchmark plus a deterministic scorer.

The benchmark file and this module's source are hashed into a lock kept in the
model manifest. If either changes, the judge changed: the learning loop
refuses to run until a human re-signs the lock. The loop can therefore never
pass its gate by editing the test (the autoresearch rule: the evaluation
harness is read-only).

Scoring is exact: an answer counts only if it equals the expected output.
Decoys have no explanation; answering one instead of abstaining is a false
accept, and the gate tolerates none.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from hrm.learning.generators import DECOY, TaskGenerator
from hrm.perception.base import DomainRegistry
from hrm.reasoning import Reasoner

BENCHMARK_FORMAT = 1


class HarnessLockError(RuntimeError):
    """The benchmark or the scorer changed since a human last signed the lock."""


def _depth_band(depth: Optional[int]) -> str:
    """Group tasks by how many non-identity steps their explanation needs."""
    if depth is None:
        # A benchmark written before the depth axis existed (v1). Its tasks are
        # still scored, but they cannot be attributed to a band, and pretending
        # they are depth 0 would file them with the decoys.
        return "unspecified"
    if depth <= 0:
        return "out-of-vocabulary"
    if depth <= 2:
        return "depth-1-2"
    return f"depth-{depth}"


@dataclass
class HarnessResult:
    solve_rate: float
    per_family: Dict[str, float]
    false_accepts: int
    n_tasks: int
    seconds: float
    per_depth: Dict[str, float] = field(default_factory=dict)
    status_mix: Dict[str, int] = field(default_factory=dict)
    per_depth_n: Dict[str, int] = field(default_factory=dict)
    beyond_depth: int = 0          # in-vocabulary but past synthesis_depth
    beyond_depth_solved: int = 0   # how many of those the reasoner still got

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkTask:
    id: str
    domain: str
    family: str
    demos: List
    test_input: object
    expected: object
    #: Shortest explanation in steps, verified when the task was generated.
    #: None for benchmarks predating the depth axis (v1), which are scored but
    #: not banded. 0 means no enumerated program explains the task.
    depth: Optional[int] = None


class Harness:
    def __init__(self, benchmark_path: str | Path):
        self.path = Path(benchmark_path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("format") != BENCHMARK_FORMAT:
            raise ValueError(f"unsupported benchmark format {data.get('format')}")
        self.tasks = [BenchmarkTask(**t) for t in data["tasks"]]

    @staticmethod
    def build(path: str | Path, generators: Sequence[TaskGenerator], per_family: int,
              decoys_per_domain: int, seed: int) -> Path:
        """Generate and write a benchmark. Run once; the file is then frozen."""
        rng = np.random.default_rng(seed)
        tasks = []
        for gen in generators:
            made = [gen.generate(f, rng) for f in gen.families for _ in range(per_family)]
            made += [gen.decoy(rng) for _ in range(decoys_per_domain)]
            for t in made:
                tasks.append({"id": f"{gen.domain}-{len(tasks):04d}", "domain": t.domain,
                              "family": t.family, "depth": t.depth,
                              "demos": [[np.asarray(i).tolist(), np.asarray(o).tolist()]
                                        for i, o in t.demos],
                              "test_input": np.asarray(t.test_input).tolist(),
                              "expected": np.asarray(t.expected).tolist()})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": BENCHMARK_FORMAT, "seed": seed, "tasks": tasks}),
                        encoding="utf-8")
        return path

    def lock(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.path.read_bytes())
        digest.update(Path(__file__).read_bytes())
        return digest.hexdigest()

    def check(self, expected_lock: str) -> None:
        if self.lock() != expected_lock:
            raise HarnessLockError("the benchmark or its scorer changed since the lock was "
                                   "signed; a human must review and re-sign the harness")

    def evaluate(self, reasoner: Reasoner, domains: DomainRegistry) -> HarnessResult:
        """Solve every task with the (read-only) reasoner and score exactly.

        Tasks are bucketed before they are scored, because a single "solve rate"
        over a mixed set hides which failures are real:

          out-of-vocabulary  no enumerated program explains the demos (decoys,
                            gravity). Answering one is a false accept; the gate
                            tolerates none. Excluded from the solve rate.
          in-budget          the true program needs <= synthesis_depth steps, so
                            the reasoner's own search can reach it. This is the
                            only population the solve rate is computed over.
          beyond-depth       the true program is in the vocabulary but needs more
                            steps than synthesis_depth allows. Reported, with a
                            count, as the measured ceiling — not dropped, because
                            a silently excluded band reads as "no ceiling".

        Views returned:
          solve_rate    correct / in-budget tasks
          per_depth     hit rate per depth band (with per_depth_n counts)
          per_family    hit rate per domain/family
          status_mix    which reasoning path produced each answer
          beyond_depth  how many in-vocabulary tasks sat past the search budget
        """
        started = time.time()
        budget = reasoner.cfg.synthesis_depth
        hits: Dict[str, List[bool]] = {}
        by_depth: Dict[str, List[bool]] = {}
        status_mix: Dict[str, int] = {}
        false_accepts = 0
        in_budget = 0
        beyond = 0
        beyond_solved = 0
        for task in self.tasks:
            domain = domains.get(task.domain)
            demos = [(domain.from_json(i), domain.from_json(o)) for i, o in task.demos]
            if task.family == DECOY or task.depth == 0:
                sol = reasoner.solve(demos, domain.from_json(task.test_input),
                                     f"bench:{task.id}", domain=domain)
                status_mix[sol.status] = status_mix.get(sol.status, 0) + 1
                false_accepts += int(sol.solved)
                continue
            sol = reasoner.solve(demos, domain.from_json(task.test_input),
                                 f"bench:{task.id}", domain=domain)
            status_mix[sol.status] = status_mix.get(sol.status, 0) + 1
            right = bool(sol.solved and domain.equal(sol.answer, domain.from_json(task.expected)))
            if task.depth is None or task.depth <= budget:
                in_budget += 1
                hits.setdefault(f"{task.domain}/{task.family}", []).append(right)
                by_depth.setdefault(_depth_band(task.depth), []).append(right)
            else:
                beyond += 1
                beyond_solved += int(right)
        per_family = {k: sum(v) / len(v) for k, v in sorted(hits.items())}
        per_depth = {k: sum(v) / len(v) for k, v in sorted(by_depth.items())}
        per_depth_n = {k: len(v) for k, v in sorted(by_depth.items())}
        total = [h for v in hits.values() for h in v]
        return HarnessResult(solve_rate=sum(total) / len(total) if total else 0.0,
                             per_family=per_family, false_accepts=false_accepts,
                             n_tasks=in_budget, seconds=time.time() - started,
                             per_depth=per_depth, status_mix=status_mix,
                             per_depth_n=per_depth_n, beyond_depth=beyond,
                             beyond_depth_solved=beyond_solved)

    def solvable(self, domain, task: "BenchmarkTask") -> bool:
        """True when some single enumerated program reproduces the demonstrations.

        An exact-replay cross-check of the benchmark's own depth annotation for
        one-step tasks: if a `depth=1` task is not explained by any single op,
        the file is lying about its own vocabulary and should not be trusted.
        Composed tasks are classified by their frozen depth against the
        reasoner's search budget rather than by an exhaustive replay, which would
        cost ~65k verifications per task and change nothing the annotation does
        not already assert (the annotation is verified at generation time and
        the file is hashed into the lock).
        """
        from hrm.config import VerifyConfig
        from hrm.verify import Verifier
        verifier = getattr(self, "_verifier", None)
        if verifier is None:
            verifier = Verifier(VerifyConfig())
            self._verifier = verifier
        demos = [(domain.from_json(i), domain.from_json(o)) for i, o in task.demos]
        if not demos:
            return False
        try:
            cands = domain.candidates(demos)
        except (ValueError, TypeError):
            return False
        for program in cands:
            if verifier.verify(domain, program, demos).passed:
                return True
        return False

    def solvable_count(self, domains: DomainRegistry) -> int:
        return sum(1 for t in self.tasks if t.family != DECOY and t.depth != 0)


def gate(baseline: HarnessResult, candidate: HarnessResult, tolerance: float) -> List[str]:
    """Reasons a candidate core must be discarded (empty list = keep)."""
    reasons: List[str] = []
    if candidate.false_accepts:
        reasons.append(f"{candidate.false_accepts} decoy(s) answered instead of abstained")
    if candidate.solve_rate < baseline.solve_rate:
        reasons.append(f"solve rate fell {baseline.solve_rate:.3f} -> {candidate.solve_rate:.3f}")
    for family, rate in baseline.per_family.items():
        new = candidate.per_family.get(family, 0.0)
        if new < rate - tolerance:
            reasons.append(f"{family} regressed {rate:.3f} -> {new:.3f}")
    # Depth bands are checked separately: a core that keeps the easy families and
    # loses the deep ones is a regression even when the headline rate holds.
    for band, rate in baseline.per_depth.items():
        new = candidate.per_depth.get(band, 0.0)
        if new < rate - tolerance:
            reasons.append(f"{band} regressed {rate:.3f} -> {new:.3f}")
    return reasons
