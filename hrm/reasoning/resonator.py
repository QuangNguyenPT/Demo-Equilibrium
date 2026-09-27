"""L-level (fast): one resonator tick of soft VSA factorization.

Each stream scores every candidate signature against its own state, the
task's shared core and the example-specific deltas, then moves toward the
softmax-weighted mix of signatures. Streams run at different temperatures so
they explore at different scales. The step may be locally expansive;
contraction is enforced one level up by the H-level mixing and KM averaging.

Cost discipline (measured): the task-core and delta scores are constant for a
whole run and are computed once (`task_scores`); the conjugated signature
matrix is cached on the candidate set; all streams share one batched matmul.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hrm.config import ReasoningConfig
from hrm.perception.base import Program, TaskEncoding
from hrm.substrate import Codebook


@dataclass
class CandidateSet:
    programs: List[Program]
    signatures: np.ndarray          # (C, D) complex
    priors: np.ndarray              # (C,) H-level hypothesis prior
    _conj_t: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def build(cls, programs: Sequence[Program], cb: Codebook,
              priors: Optional[Sequence[float]] = None) -> "CandidateSet":
        sigs = np.stack([p.signature(cb).data for p in programs])
        pri = np.zeros(len(programs)) if priors is None else np.asarray(priors, dtype=np.float64)
        return cls(list(programs), sigs, pri.copy())

    @property
    def conj_t(self) -> np.ndarray:
        """(D, C) conjugated signatures: `states @ conj_t` gives D * similarities."""
        if self._conj_t is None:
            self._conj_t = np.ascontiguousarray(np.conj(self.signatures).T)
        return self._conj_t

    def similarities(self, states: np.ndarray) -> np.ndarray:
        """(..., D) states -> (..., C) similarities in [-1, 1]."""
        return (states @ self.conj_t).real / self.signatures.shape[1]

    def index_of(self, description: str) -> Optional[int]:
        for i, p in enumerate(self.programs):
            if p.describe() == description:
                return i
        return None

    def raise_prior(self, program: Program, prior: float, cb: Codebook) -> None:
        """Lift an existing candidate's prior, or append the program with it."""
        i = self.index_of(program.describe())
        if i is not None:
            self.priors[i] = max(self.priors[i], prior)
            return
        self.programs.append(program)
        self.signatures = np.concatenate([self.signatures, program.signature(cb).data[None, :]])
        self.priors = np.append(self.priors, prior)
        self._conj_t = None


class Resonator:
    """Parameters (the core's search scalars):
        log_beta  per-stream softmax inverse temperature
        w_prior   per-stream weight on the H-level prior
        gain      per-stream blend between state and proposal
    """

    def __init__(self, cfg: ReasoningConfig):
        n = cfg.n_streams
        self.params: Dict[str, np.ndarray] = {
            "log_beta": np.log(np.linspace(4.0, 24.0, n)),
            "w_prior": np.full(n, 1.0),
            "gain": np.full(n, 0.7),
        }

    @staticmethod
    def task_scores(cands: CandidateSet, task: TaskEncoding) -> np.ndarray:
        """(C,) how well each candidate explains the task core, disambiguated by the
        example-specific residues. Constant for a run: compute once per task/candidates."""
        scores = cands.similarities(task.shared.data)
        if task.deltas:
            scores = scores + 0.5 * cands.similarities(np.stack([d.data for d in task.deltas])).mean(0)
        return scores

    def step(self, z: np.ndarray, cands: CandidateSet, task_scores: np.ndarray) -> np.ndarray:
        """z: (n, D) state -> unprojected proposal (n, D)."""
        beta = np.exp(self.params["log_beta"])[:, None]
        logits = beta * (cands.similarities(z) + task_scores[None, :]
                         + self.params["w_prior"][:, None] * cands.priors[None, :])
        attn = np.exp(logits - logits.max(axis=1, keepdims=True))
        attn /= attn.sum(axis=1, keepdims=True)
        gain = np.clip(self.params["gain"], 0.0, 1.0)[:, None]
        return ((1.0 - gain) * z + gain * (attn @ cands.signatures)).astype(z.dtype)

    @staticmethod
    def top_candidate(z: np.ndarray, cands: CandidateSet) -> Tuple[int, float]:
        sims = cands.similarities(z[0])
        idx = int(np.argmax(sims + cands.priors))
        return idx, float(sims[idx])
