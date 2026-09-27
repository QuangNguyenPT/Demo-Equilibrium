"""The language section: the model's native language ability.

It is not a translator bolted onto the front; it works inside the model:
  * inner speech: when recognition fails, the reasoning section asks it for
    hypotheses in the domain's program language. Proposals join the candidate
    set and only verification decides: pattern proposes, structure verifies;
  * perception of text: free language is rewritten into the task notation,
    which is then parsed deterministically;
  * intent: its embeddings feed the intent matrix.
Everything it produces is untrusted text and is validated before use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LanguageError(RuntimeError):
    """The language backend is unavailable or failed."""


class LanguageSection(ABC):
    name: str = "language"
    has_hidden_states: bool = False

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Unit-norm (or zero) float32 vector representing the text."""

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Deterministic (greedy) continuation of an instruction prompt."""


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v
