"""Q-learning ACT head over controller features, confined to discretionary actions.

The head chooses only between CONTINUE and an early RESTART: pure compute
allocation (reward = solved - compute cost). It is never consulted for ACCEPT
or ABSTAIN, so even adversarial weights cannot make the model accept an
unverified answer. Linear Q over 9 features: tens of floats, numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from hrm.reasoning.controller import Action, ControllerFeatures

N_FEATURES = 9
DISCRETIONARY = (Action.CONTINUE, Action.RESTART)


def featurize(f: ControllerFeatures) -> np.ndarray:
    return np.array([
        min(f.residual, 1.0),
        np.tanh(100.0 * f.residual_slope),
        min(f.energy, 2.0) / 2.0,
        f.diversity,
        f.memory_confidence,
        f.verification_margin,
        min(f.plateau_count, 20) / 20.0,
        min(f.restarts_used, 10) / 10.0,
        1.0,
    ], dtype=np.float64)


@dataclass
class Transition:
    features: np.ndarray
    action: int                              # index into DISCRETIONARY
    reward: float
    next_features: Optional[np.ndarray]      # None = terminal


class QHead:
    def __init__(self, lr: float = 0.05, gamma: float = 0.97, epsilon: float = 0.1,
                 seed: int = 0):
        self.w = np.zeros((len(DISCRETIONARY), N_FEATURES))
        self.lr, self.gamma, self.epsilon = lr, gamma, epsilon
        self.rng = np.random.default_rng(seed)
        self.train_mode = False

    def q(self, feats: np.ndarray) -> np.ndarray:
        return self.w @ feats

    def choose(self, f: ControllerFeatures, allowed: Tuple[Action, ...]) -> Action:
        allowed = tuple(a for a in allowed if a in DISCRETIONARY)
        if not allowed:
            return Action.CONTINUE
        if len(allowed) == 1:
            return allowed[0]
        if self.train_mode and self.rng.random() < self.epsilon:
            return allowed[int(self.rng.integers(len(allowed)))]
        qs = self.q(featurize(f))
        return allowed[int(np.argmax([qs[DISCRETIONARY.index(a)] for a in allowed]))]

    def update(self, batch: List[Transition]) -> float:
        """One TD(0) pass; returns mean |TD error|."""
        errs = []
        for t in batch:
            target = t.reward
            if t.next_features is not None:
                target += self.gamma * float(np.max(self.q(t.next_features)))
            td = target - float(self.w[t.action] @ t.features)
            self.w[t.action] += self.lr * td * t.features
            errs.append(abs(td))
        return float(np.mean(errs)) if errs else 0.0
