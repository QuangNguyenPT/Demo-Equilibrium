"""Curiosity: the model's own choice of what to study next.

Study where learning is happening or has never been tried; leave what is
mastered; give up on what shows no progress (the vocabulary cannot express
it, so effort there is wasted until a human extends it). Scores:
    unexplored  1 + exploration bonus (try everything at least a few times)
    learning    |progress| + (1 - competence)
    gap         (1 - competence) * 0.5
A small epsilon keeps occasional random choices among eligible regions.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from hrm.config import MetaConfig
from hrm.metacognition.self_model import SelfModel

_SKIP = ("mastered", "stuck")


class Curiosity:
    def __init__(self, cfg: MetaConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def choose(self, self_model: SelfModel, regions: Sequence[str]) -> Optional[str]:
        """Next region to study, or None when nothing learnable is left."""
        eligible: List[str] = [r for r in regions if self_model.status(r) not in _SKIP]
        if not eligible:
            return None
        if self.rng.random() < self.cfg.explore:
            return str(self.rng.choice(eligible))
        return max(eligible, key=lambda r: (self._score(self_model, r), -eligible.index(r)))

    def _score(self, self_model: SelfModel, region: str) -> float:
        status = self_model.status(region)
        c = self_model.get(region)
        if status == "unexplored":
            return 1.0 + (1.0 - (c.attempts if c else 0) / self.cfg.min_attempts)
        if status == "learning":
            return abs(c.progress) + (1.0 - c.competence)
        return 0.5 * (1.0 - c.competence)
