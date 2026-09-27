"""Budget controller: ACCEPT / CONTINUE / RESTART / ABSTAIN plus the KM coefficient.

ACCEPT is conjunctive: the state stopped moving (kinetic), the hypothesis
explains the demonstrations (semantic) and it verifies. Kinetic convergence
without semantic convergence is a false attractor and routes to RESTART,
never ACCEPT. A learned Q-head may only choose the timing of an early
restart; ACCEPT and ABSTAIN are rails in code, not in convention.

Every non-ACCEPT state is terminal or restarts: once the restart budget is
spent, the controller ABSTAINs. Continuing could only ever reach ACCEPT, and
a plateaued hypothesis is not converging, so the decision is made rather than
deferred to ``max_ticks``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Sequence

import numpy as np

from hrm.config import ReasoningConfig


class Action(Enum):
    ACCEPT = auto()
    CONTINUE = auto()
    RESTART = auto()
    ABSTAIN = auto()


@dataclass
class ControllerFeatures:
    residual: float
    residual_slope: float
    energy: float
    diversity: float
    memory_confidence: float
    verification_margin: float
    plateau_count: int
    restarts_used: int


@dataclass
class Decision:
    action: Action
    lam: float
    reason: str


class BudgetController:
    def __init__(self, cfg: ReasoningConfig, qhead=None):
        self.cfg = cfg
        self.qhead = qhead
        self._plateau = 0
        self._best_energy = math.inf

    def reset(self) -> None:
        self._plateau = 0
        self._best_energy = math.inf

    def observe(self, energy: float) -> int:
        """Track energy improvement; returns ticks since the last improvement."""
        if energy < self._best_energy - 1e-6:
            self._best_energy = energy
            self._plateau = 0
        else:
            self._plateau += 1
        return self._plateau

    def decide(self, f: ControllerFeatures) -> Decision:
        cfg = self.cfg
        lam = self._lambda(f)
        kinetic = f.residual < cfg.eps_kinetic
        semantic = f.energy < cfg.eps_energy
        verified = f.verification_margin >= cfg.tau_verify
        if kinetic and semantic and verified:
            return Decision(Action.ACCEPT, lam, "kinetic, semantic and verification all pass")

        plateaued = f.plateau_count >= cfg.plateau_cycles * cfg.ticks_per_cycle
        false_attractor = kinetic and not semantic
        if plateaued or false_attractor:
            if f.restarts_used < cfg.restart_budget:
                return Decision(Action.RESTART, lam,
                                "false attractor" if false_attractor else "energy plateau")
            # Budget exhausted: the only remaining action is ACCEPT, and a plateau
            # means the leading hypothesis is not improving, so the verification
            # margin cannot rise to tau_verify either. Terminate instead of
            # spinning to max_ticks on an unreachable ACCEPT.
            return Decision(Action.ABSTAIN, lam,
                            "restart budget exhausted, energy high" if not semantic
                            else "restart budget exhausted, verification unreachable")

        if (self.qhead is not None and f.plateau_count >= 2
                and f.restarts_used < cfg.restart_budget):
            if self.qhead.choose(f, (Action.CONTINUE, Action.RESTART)) is Action.RESTART:
                return Decision(Action.RESTART, lam, "qhead: early restart")
        return Decision(Action.CONTINUE, lam, "progressing")

    def _lambda(self, f: ControllerFeatures) -> float:
        """Exploratory (low lambda) far from acceptance, damped (high) near it."""
        cfg = self.cfg
        progress = 1.0 / (1.0 + max(f.energy / cfg.eps_energy, 0.0))
        return float(np.clip(cfg.lam_min + (cfg.lam_max - cfg.lam_min) * progress,
                             cfg.lam_min, cfg.lam_max))


class RestartPolicy:
    """Annealed phase kick sigma_k = sigma0 * gamma^k; every other restart also
    swaps the parse (segmentation) hypothesis, down-weighting ones that failed."""

    def __init__(self, cfg: ReasoningConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def sigma(self) -> float:
        return self.cfg.restart_sigma0 * self.cfg.restart_gamma ** self.count

    def kick(self, z: np.ndarray) -> np.ndarray:
        out = z * np.exp(1j * self.sigma() * self.rng.standard_normal(z.shape))
        self.count += 1
        return out.astype(z.dtype)

    def should_swap_parse(self) -> bool:
        return self.count % 2 == 1

    def pick_parse(self, n_parses: int, current: int, failed: Sequence[int]) -> Optional[int]:
        options: List[int] = [p for p in range(n_parses) if p != current]
        if not options:
            return None
        weights = np.array([0.25 if p in failed else 1.0 for p in options])
        return int(self.rng.choice(options, p=weights / weights.sum()))
