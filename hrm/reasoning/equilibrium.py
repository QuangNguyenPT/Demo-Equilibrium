"""H-level (deep): the contractive equilibrium operator.

One tick:   z <- Pi_T( lam z + (1 - lam) Pi_T( M_H . f_L(z) ) )
    f_L   resonator (L-level free search)
    M_H   input-dependent stream-mixing matrix, Sinkhorn-projected onto the
          Birkhoff polytope (spectral norm <= 1). With lam in
          [lam_min, lam_max] the update is an averaged operator, so
          Krasnoselskii-Mann iteration converges.
Every `ticks_per_cycle` L-ticks an H-cycle boundary measures the semantic
energy of the leading hypotheses and folds it into their priors: slow,
deliberate re-ranking on top of the fast dynamics.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from hrm.config import ReasoningConfig
from hrm.perception.base import Program, TaskEncoding
from hrm.reasoning.controller import (Action, BudgetController, ControllerFeatures,
                                      RestartPolicy)
from hrm.reasoning.resonator import CandidateSet, Resonator
from hrm.substrate import PhasorVector, get_backend, torus_project
from hrm.trace import Trace, TraceEvent

EnergyFn = Callable[[Program], Tuple[float, float]]
# (current task, failed parses) -> (re-parsed task, candidates) or None to keep going
RestartHook = Callable[[TaskEncoding, List[int]],
                       Optional[Tuple[TaskEncoding, CandidateSet]]]


@dataclass
class EquilibriumResult:
    status: str                 # accept | abstain | ceiling
    program: Optional[Program]
    ticks: int
    residual: float
    energy: float
    margin: float
    restarts: int
    task: TaskEncoding          # final encoding (the parse may have been swapped)
    cands: CandidateSet


class MixingHead:
    """M_H: learnable base logits + inter-stream coherence, Sinkhorn-projected."""

    def __init__(self, cfg: ReasoningConfig, rng: np.random.Generator):
        n = cfg.n_streams
        self.iters = cfg.sinkhorn_iters
        self.base_logits = 0.1 * rng.standard_normal((n, n)) + 2.0 * np.eye(n)
        self.feature_scale = 0.5

    def matrix(self, z: np.ndarray) -> np.ndarray:
        coherence = (z @ np.conj(z).T).real / z.shape[1]
        return get_backend().sinkhorn(self.base_logits + self.feature_scale * coherence,
                                      self.iters)


class EquilibriumEngine:
    def __init__(self, cfg: ReasoningConfig, seed: int, qhead=None):
        self.cfg = cfg
        self.rng = np.random.default_rng([seed, 4])
        self.resonator = Resonator(cfg)
        self.mixing = MixingHead(cfg, np.random.default_rng([seed, 3]))
        self.controller = BudgetController(cfg, qhead)
        self.restarts = RestartPolicy(cfg, np.random.default_rng([seed, 2]))

    def init_state(self, task_vector: PhasorVector) -> np.ndarray:
        """Stream 0 starts at the task core; the rest at growing perturbations of it."""
        z = np.empty((self.cfg.n_streams, task_vector.dim), dtype=task_vector.data.dtype)
        z[0] = task_vector.data
        for s in range(1, self.cfg.n_streams):
            z[s] = task_vector.noisy(0.5 * s, self.rng).data
        return z

    def tick(self, z: np.ndarray, cands: CandidateSet, task_scores: np.ndarray,
             lam: float) -> Tuple[np.ndarray, float]:
        lam = float(np.clip(lam, self.cfg.lam_min, self.cfg.lam_max))
        proposal = self.resonator.step(z, cands, task_scores)
        mixed = self.mixing.matrix(z).astype(z.dtype) @ proposal
        nxt = torus_project((lam * z + (1.0 - lam) * torus_project(mixed)).astype(z.dtype))
        return nxt, float(np.linalg.norm(nxt - z) / np.sqrt(z.size))

    def h_refresh(self, z: np.ndarray, cands: CandidateSet, energy_fn: EnergyFn) -> None:
        sims = cands.similarities(z[0])
        for i in np.argsort(-(sims + cands.priors))[: self.cfg.refresh_top_k]:
            energy, _ = energy_fn(cands.programs[int(i)])
            cands.priors[int(i)] = max(cands.priors[int(i)], 1.0 - energy)

    def run(self, task: TaskEncoding, cands: CandidateSet, energy_fn: EnergyFn,
            trace: Trace, on_restart: Optional[RestartHook] = None) -> EquilibriumResult:
        cfg = self.cfg
        self.controller.reset()
        self.restarts.reset()
        z = self.init_state(task.shared)
        scores = self.resonator.task_scores(cands, task)
        lam = cfg.lam_max
        residual, energy, margin = math.inf, math.inf, 0.0
        best_energy, best_program = math.inf, None
        residuals: deque = deque(maxlen=8)
        hypotheses: deque = deque(maxlen=16)

        for t in range(cfg.max_ticks):
            z, residual = self.tick(z, cands, scores, lam)
            if (t + 1) % cfg.ticks_per_cycle == 0:
                self.h_refresh(z, cands, energy_fn)
            idx, _ = self.resonator.top_candidate(z, cands)
            program = cands.programs[idx]
            energy, margin = energy_fn(program)
            if energy < best_energy:
                best_energy, best_program = energy, program
            residuals.append(residual)
            hypotheses.append(program.describe())
            trace.log(TraceEvent(t, "step", program.describe(), residual, energy, lam))

            decision = self.controller.decide(ControllerFeatures(
                residual=residual,
                residual_slope=_slope(residuals),
                energy=energy,
                diversity=len(set(hypotheses)) / len(hypotheses),
                memory_confidence=float(cands.priors.max()),
                verification_margin=margin,
                plateau_count=self.controller.observe(energy),
                restarts_used=self.restarts.count))
            lam = decision.lam

            if decision.action is Action.ACCEPT:
                trace.log(TraceEvent(t, "accept", program.describe(), residual, energy, lam,
                                     {"reason": decision.reason}))
                return EquilibriumResult("accept", program, t + 1, residual, energy, margin,
                                         self.restarts.count, task, cands)
            if decision.action is Action.RESTART:
                z = torus_project(self.restarts.kick(z))
                detail = {"reason": decision.reason, "sigma": self.restarts.sigma()}
                if on_restart is not None and self.restarts.should_swap_parse():
                    swapped = on_restart(task, trace.failed_parses())
                    if swapped is not None:
                        detail["failed_parse"] = task.parse
                        task, cands = swapped
                        detail["parse"] = task.parse
                        z = self.init_state(task.shared)
                        scores = self.resonator.task_scores(cands, task)
                self.controller.reset()
                residuals.clear()
                trace.log(TraceEvent(t, "restart", residual=residual, energy=energy, lam=lam,
                                     detail=detail))
                continue
            if decision.action is Action.ABSTAIN:
                trace.log(TraceEvent(t, "abstain", residual=residual, energy=energy, lam=lam,
                                     detail={"reason": decision.reason}))
                return EquilibriumResult("abstain", best_program, t + 1, residual, best_energy,
                                         margin, self.restarts.count, task, cands)
        return EquilibriumResult("ceiling", best_program, cfg.max_ticks, residual, best_energy,
                                 margin, self.restarts.count, task, cands)


def _slope(values: deque) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(values), dtype=np.float64), np.asarray(values), 1)[0])
