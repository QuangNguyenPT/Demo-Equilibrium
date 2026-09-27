"""Intent: what does the sender want from this input?

    answer    solve the task and reply
    learn     take the examples as a lesson (examples, or a query with its answer)
    feedback  a judgement or correction of an earlier answer

A softmax regression (the intent matrix W, intents x features) over features
of the message. W starts from a prior that encodes the obvious structure, so
it works from the first message, and it learns from human-confirmed intents.
When the most likely intent is below the threshold the model asks instead of
guessing. Three text columns carry the similarity of the message text to each
intent's prototype phrase in the language section's embedding space (zero when
the model runs without a language section).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from hrm.perception.base import Example

INTENTS = ("answer", "learn", "feedback")
PROTOTYPES = {
    "answer": "what is the answer? solve this and predict the output",
    "learn": "learn this, remember this rule, here is how it works",
    "feedback": "that was wrong, that is correct, your earlier answer was right",
}

_PRIOR = np.array([
    # demos query expected refers_to verdict text n_demos bias ~answer ~learn ~feedback
    [1.0,   3.0,  -3.0,    -2.0,    -3.0,   0.0, 0.0,   -2.0, 3.0,    0.0,   0.0],  # answer
    [2.0,  -1.0,   3.0,    -2.0,    -3.0,   0.0, 0.5,   -1.0, 0.0,    3.0,   0.0],  # learn
    [-1.0, -1.0,   0.0,     3.0,     3.0,   0.0, 0.0,   -1.0, 0.0,    0.0,   3.0],  # feedback
])


class TextSignal:
    """Similarity of a text to each intent prototype, in a language embedding space."""

    def __init__(self, embed: Callable[[str], np.ndarray]):
        self.embed = embed
        self._prototypes: Optional[np.ndarray] = None

    def __call__(self, text: Optional[str]) -> np.ndarray:
        if not text:
            return np.zeros(len(INTENTS))
        if self._prototypes is None:
            self._prototypes = np.stack([self.embed(PROTOTYPES[i]) for i in INTENTS])
        return np.clip(self._prototypes @ self.embed(text), 0.0, 1.0)


@dataclass
class Message:
    demos: Optional[Sequence[Example]] = None
    query: Any = None                 # a test input to answer
    expected: Any = None              # an answer supplied by the sender
    refers_to: Optional[str] = None   # an earlier task this message is about
    verdict: Optional[str] = None     # "correct" / "wrong" about that task
    text: Optional[str] = None        # natural language (language section)
    task_id: Optional[str] = None     # name for a new task (not an intent signal)


class IntentHead:
    def __init__(self, lr: float = 0.1):
        self.lr = lr
        self.w = _PRIOR.copy()

    @staticmethod
    def features(msg: Message, text_signal: Optional[np.ndarray] = None) -> np.ndarray:
        n_demos = len(msg.demos) if msg.demos else 0
        structure = [float(n_demos > 0), float(msg.query is not None),
                     float(msg.expected is not None), float(msg.refers_to is not None),
                     float(msg.verdict is not None), float(bool(msg.text)),
                     min(n_demos, 5) / 5.0, 1.0]
        text = np.zeros(len(INTENTS)) if text_signal is None else text_signal
        return np.concatenate([structure, text])

    def predict(self, msg: Message, text_signal: Optional[np.ndarray] = None) -> Dict[str, float]:
        logits = self.w @ self.features(msg, text_signal)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return {intent: float(v) for intent, v in zip(INTENTS, p)}

    def update(self, msg: Message, intent: str,
               text_signal: Optional[np.ndarray] = None) -> None:
        """One cross-entropy gradient step toward a human-confirmed intent."""
        if intent not in INTENTS:
            raise ValueError(f"intent must be one of {INTENTS}")
        x = self.features(msg, text_signal)
        p = np.array(list(self.predict(msg, text_signal).values()))
        target = np.array([float(i == intent) for i in INTENTS])
        self.w -= self.lr * np.outer(p - target, x)

    def to_list(self) -> List[List[float]]:
        return self.w.tolist()

    def load_list(self, rows: List[List[float]]) -> None:
        w = np.asarray(rows, dtype=np.float64)
        if w.shape != _PRIOR.shape:
            raise ValueError(f"intent matrix has shape {w.shape}, expected {_PRIOR.shape}")
        self.w = w
