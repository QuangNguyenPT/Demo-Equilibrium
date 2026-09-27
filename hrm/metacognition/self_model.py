"""The self-model: what the model knows it knows, and what it knows it does not.

A competence map over concept regions ("domain/family"). Each region keeps
counts and two moving averages of success, fast and slow. Their difference is
learning progress (Oudeyer & Kaplan): positive while competence is rising.
A region is
    unexplored  fewer than `min_attempts` tries
    mastered    competence >= `mastery`
    stuck       `patience` or more tries, low competence, no progress: likely
                unlearnable with the current vocabulary
    learning    competence moving
    gap         low competence, not yet given up on
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from hrm.config import MetaConfig

OUTCOMES = {"correct": 1.0, "solved": 1.0, "wrong": 0.0, "abstained": 0.0}
_FAST, _SLOW = 0.3, 0.05
_PROGRESS_EPS = 0.02


@dataclass
class Competence:
    region: str
    attempts: int = 0
    successes: int = 0
    wrong: int = 0
    abstained: int = 0
    fast: float = 0.0
    slow: float = 0.0

    @property
    def competence(self) -> float:
        return self.fast

    @property
    def progress(self) -> float:
        return self.fast - self.slow


class SelfModel:
    def __init__(self, cfg: MetaConfig):
        self.cfg = cfg
        self._regions: Dict[str, Competence] = {}
        self._lock = threading.Lock()

    def observe(self, region: str, outcome: str) -> Competence:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
        x = OUTCOMES[outcome]
        with self._lock:
            c = self._regions.setdefault(region, Competence(region))
            c.attempts += 1
            c.successes += int(x > 0)
            c.wrong += int(outcome == "wrong")
            c.abstained += int(outcome == "abstained")
            c.fast += _FAST * (x - c.fast)
            c.slow += _SLOW * (x - c.slow)
            return c

    def get(self, region: str) -> Optional[Competence]:
        return self._regions.get(region)

    def status(self, region: str) -> str:
        c = self._regions.get(region)
        cfg = self.cfg
        if c is None or c.attempts < cfg.min_attempts:
            return "unexplored"
        if c.competence >= cfg.mastery:
            return "mastered"
        if (c.attempts >= cfg.patience and c.competence < 0.2
                and abs(c.progress) < _PROGRESS_EPS):
            return "stuck"
        if abs(c.progress) >= _PROGRESS_EPS:
            return "learning"
        return "gap"

    def report(self) -> List[Dict[str, object]]:
        """Every known region with its status, most competent first."""
        rows = [{"region": r, "status": self.status(r), "competence": round(c.competence, 3),
                 "progress": round(c.progress, 3), "attempts": c.attempts}
                for r, c in self._regions.items()]
        return sorted(rows, key=lambda row: (-row["competence"], row["region"]))

    def to_dict(self) -> Dict[str, Dict]:
        return {r: asdict(c) for r, c in self._regions.items()}

    def load_dict(self, data: Dict[str, Dict]) -> None:
        self._regions = {r: Competence(**c) for r, c in data.items()}
