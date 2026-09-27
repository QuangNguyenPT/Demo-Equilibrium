"""Hormone system: four global neuromodulators in a box, updated by a projected EMA (spec §6, H1).

Roles follow Doya (2002): dopamine = reward-prediction error, serotonin = patience
(discount), noradrenaline = inverse temperature, acetylcholine = learning rate. They act
only outside the certified maps, so no hormonal state can break convergence (H1b).
"""

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

HORMONES = ("dopamine", "serotonin", "noradrenaline", "acetylcholine")


@dataclass
class Neuromodulators:
    """Levels m in [lower, upper] ⊂ (0, 1)^4; step: m <- Π_box((1 - rate) m + rate drive)."""

    level: Tensor
    lower: Tensor
    upper: Tensor
    rate: float

    @classmethod
    def create(cls, lower: Sequence[float], upper: Sequence[float], rate: float) -> "Neuromodulators":
        """Start at the box midpoint."""
        lo = torch.tensor(lower, dtype=torch.float64)
        hi = torch.tensor(upper, dtype=torch.float64)
        if lo.shape != (len(HORMONES),) or hi.shape != lo.shape:
            raise ValueError(f"need {len(HORMONES)} bounds per side")
        if not bool(((0.0 < lo) & (lo < hi) & (hi < 1.0)).all()):
            raise ValueError("bounds must satisfy 0 < lower < upper < 1")
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"rate must be in (0, 1], got {rate}")
        return cls(level=(lo + hi) / 2.0, lower=lo, upper=hi, rate=float(rate))

    def step(self, drive: Tensor) -> Tensor:
        """One projected EMA update toward drive; returns the new levels."""
        mixed = (1.0 - self.rate) * self.level + self.rate * drive.to(self.level.dtype)
        self.level = torch.minimum(torch.maximum(mixed, self.lower), self.upper)
        return self.level

    def get(self, name: str) -> float:
        """Level of one hormone by name."""
        return float(self.level[HORMONES.index(name)])

    def segment_budget(self) -> int:
        """Maximum refinement segments ⌈1 / (1 - serotonin)⌉ (patience as a discount)."""
        return math.ceil(1.0 / (1.0 - self.get("serotonin")))
