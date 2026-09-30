"""Mushroom-Body Index (MBI): a learned router that orders constructed features, never answers.

Doc 14 N1, derived from the local MaleCNS connectome and re-measured in
`experiments/connectome_malecns.py` (M2/M3), made into a component:

    input   s = FHRR evidence signature, real view (hrm_x/signature.py)           2d numbers
    expand  a = J s,  J sparse random: each of M Kenyon units samples `claws` inputs  M numbers
    inhibit h = kWTA_f(a): the top f*M units are 1, the rest 0 (the APL neuron)         M bits
    readout s_c = W_c (h - h_bar) for every channel c                                  one per class
    learn   one-shot:  W_c += eta * (y_c - sigmoid(s_c)) (h - h_bar)   (dopamine-gated, per channel)
            at sleep:  softmax regression over channels on dreamed tasks (gradient descent)

A channel is a *behaviour class* of features (the canonical key modulo the base span, the same
key concept memory uses), so "max", "min" and "|x - y|" share one channel, and a stored concept
and the construction that first found it are the same channel whatever their expressions.

The invariant, taken from Kimi K3's router ("the routing bias steers selection only"): MBI
scores only *order* candidates for `construct.construct`, whose stop rule is answer-preserving,
so a wrong or memorising index costs operations and cannot change a verified answer. M3 is the
biological reason for the shape: one sparse code, many separately gated readouts (median 9 MBON
types per Kenyon cell); a new concept is a new readout, never a new representation.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hrm_x.construct import Feature, _base_projector, _key
from hrm_x.signature import SignatureEncoder

Q = Fraction


def feature_key(values: Sequence[float]) -> Optional[str]:
    """Behaviour-class key of a feature's probe values (modulo span{1, x, y, xy}); None if in span."""
    v = np.asarray(values, dtype=float)
    r = v - _base_projector() @ v
    if np.max(np.abs(r)) < 1e-7 * max(1.0, float(np.max(np.abs(v)))):
        return None
    return str(_key(r / r[np.argmax(np.abs(r))]))


class MushroomBodyIndex:
    """Sparse random expansion + kWTA + one readout per behaviour class."""

    def __init__(self, n_kc: int = 2048, claws: int = 6, frac: float = 0.02, eta: float = 1.0,
                 kwta: bool = True, shuffle_seed: Optional[int] = None, seed: int = 11,
                 encoder: Optional[SignatureEncoder] = None) -> None:
        self.enc = encoder or SignatureEncoder()
        n_in = 2 * self.enc.dim
        rng = np.random.default_rng(seed if shuffle_seed is None else shuffle_seed)
        self.idx = rng.integers(0, n_in, size=(n_kc, claws))
        self.w = rng.normal(0.0, 1.0, size=(n_kc, claws))
        self.n_kc, self.frac, self.eta, self.kwta = n_kc, frac, eta, kwta
        self.channels: Dict[str, np.ndarray] = {}
        self.h_bar = np.full(n_kc, frac)
        self.writes = 0

    # ----- the code -----
    def code(self, evidence: Sequence[Tuple[Tuple[Q, Q], Q]]) -> np.ndarray:
        s = self.enc.real(evidence)
        a = np.einsum("kc,kc->k", self.w, s[self.idx])
        if not self.kwta:
            h = np.maximum(a, 0.0)
            n = np.linalg.norm(h)
            return h / n if n > 0 else h
        k = max(1, int(round(self.frac * self.n_kc)))
        h = np.zeros(self.n_kc)
        h[np.argpartition(a, -k)[-k:]] = 1.0
        return h

    def scores(self, h: np.ndarray) -> Dict[str, float]:
        c = h - self.h_bar
        return {key: float(w @ c) for key, w in self.channels.items()}

    def novelty(self, evidence: Sequence[Tuple[Tuple[Q, Q], Q]]) -> float:
        """1 - best readout confidence: high when nothing stored resembles this evidence."""
        sc = self.scores(self.code(evidence))
        return 1.0 - (1.0 / (1.0 + math.exp(-max(sc.values())))) if sc else 1.0

    # ----- learning -----
    def learn(self, evidence: Sequence[Tuple[Tuple[Q, Q], Q]], key: Optional[str], mind: Any = None) -> None:
        """One-shot, prediction-error-gated write: the verified channel up, confident wrong ones down."""
        if key is None:
            return
        h = self.code(evidence)
        c = h - self.h_bar
        self.channels.setdefault(key, np.zeros(self.n_kc))
        for k, w in self.channels.items():
            y = 1.0 if k == key else 0.0
            p = 1.0 / (1.0 + math.exp(-float(w @ c)))
            if y == 1.0 or p > 0.5:
                w += self.eta * (y - p) * c / max(1.0, float(c @ c)) * self.n_kc * self.frac
        self.h_bar = 0.99 * self.h_bar + 0.01 * h
        self.writes += 1

    def train(self, samples: Sequence[Tuple[np.ndarray, str]], epochs: int = 20, lr: float = 0.5,
              seed: int = 0) -> float:
        """Softmax regression of the readouts on (code, key) pairs -- the sleep-time gradient step.

        Returns the final training accuracy. Codes are fixed (J is not learned), as in the fly.
        """
        keys = sorted({k for _, k in samples} | set(self.channels))
        for k in keys:
            self.channels.setdefault(k, np.zeros(self.n_kc))
        W = np.stack([self.channels[k] for k in keys])
        H = np.stack([h - self.h_bar for h, _ in samples])
        y = np.array([keys.index(k) for _, k in samples])
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for i in rng.permutation(len(samples)):
                z = W @ H[i]
                p = np.exp(z - z.max())
                p /= p.sum()
                p[y[i]] -= 1.0
                W -= lr * np.outer(p, H[i]) / max(1.0, float(H[i] @ H[i])) * 8.0
        for j, k in enumerate(keys):
            self.channels[k] = W[j]
        pred = np.argmax(H @ W.T, axis=1)
        return float(np.mean(pred == y))

    # ----- the router interface used by Concept._order -----
    def order(self, feats: Sequence[Feature], evidence: Sequence[Tuple[Tuple[Q, Q], Q]], mind: Any = None) -> List[Feature]:
        """Features sorted by readout score (known classes first), ties and unknowns by MDL."""
        sc = self.scores(self.code(evidence))
        def rank(f: Feature) -> Tuple[float, float]:
            k = feature_key(f.values)
            return (-sc.get(k, -1e9) if k is not None else 1e9, f.bits)
        return sorted(feats, key=rank)

    def cost(self) -> int:
        """Operations of one routing: the expansion plus the active units times the channels."""
        return int(self.n_kc * self.idx.shape[1] + self.frac * self.n_kc * max(1, len(self.channels)))
