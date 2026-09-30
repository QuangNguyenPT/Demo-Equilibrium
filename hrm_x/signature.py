"""FHRR evidence signatures: a set of (x, y) -> z observations as one relational phasor vector.

This is EQB's rotation binding carried into HRM-X. In `legacy/hrm_eqb` a token was bound to its
(row, col, slot) by rotating each complex plane by an angle proportional to the coordinate; here a
*relation* is bound to its *role* the same way. For every evidence row the code is

    r = O[order(x, y, z)]  +  R_zx * phi(lr(z, x))  +  R_zy * phi(lr(z, y))  +  R_xy * phi(lr(x, y))
        +  R_zs * phi(lr(z, x + y))  +  R_zp * phi(lr(z, x * y))

  O[.]        a random phasor per weak order of (x, y, z) (13 patterns): ordinal structure
  R_ab        random role codes (bind = elementwise product, exact inverse by conjugation)
  phi(v)      character code exp(i w v), w ~ N(0, 1) per dimension: phi(u) phi(v) = phi(u + v)
  lr(a, b)    signed log-ratio sign(a) sign(b) log(1 + |a|) - log(1 + |b|): scale-free comparison

and the signature is the normalised bundle (sum) over rows. Every channel is a *relation between
roles*, not a value, so the same operator sampled at different points gives nearby signatures --
the property the router needs and that the EQB token field lacked (it memorised values, ADR-012).

The relations are generic comparisons (order, ratio to each argument, to their sum and product);
none names a target concept. Whether they are enough is measured, not assumed (E16).
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Sequence, Tuple

import numpy as np

Q = Fraction
ROLES = ("zx", "zy", "xy", "zs", "zp")
DIM = 256


def _order_pattern(x: float, y: float, z: float) -> int:
    """Index of the weak order of three numbers (13 patterns), from the signs of pairwise differences."""
    s = lambda a, b: 0 if a == b else (1 if a > b else 2)
    return s(x, y) * 9 + s(x, z) * 3 + s(y, z)


def _lr(a: float, b: float) -> float:
    sa = 1.0 if a >= 0 else -1.0
    sb = 1.0 if b >= 0 else -1.0
    return sa * sb * (math.log1p(abs(a)) - math.log1p(abs(b)))


class SignatureEncoder:
    """Seeded, parameter-free encoder: evidence rows -> unit-norm complex vector of length `dim`."""

    def __init__(self, dim: int = DIM, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.orders = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(27, dim)))
        self.roles = {r: np.exp(1j * rng.uniform(-np.pi, np.pi, size=dim)) for r in ROLES}
        self.omega = rng.normal(0.0, 1.0, size=dim)

    def phi(self, v: float) -> np.ndarray:
        return np.exp(1j * self.omega * v)

    def row(self, x: float, y: float, z: float) -> np.ndarray:
        rel = {"zx": _lr(z, x), "zy": _lr(z, y), "xy": _lr(x, y), "zs": _lr(z, x + y), "zp": _lr(z, x * y)}
        code = self.orders[_order_pattern(x, y, z)].copy()
        for r in ROLES:
            code += self.roles[r] * self.phi(rel[r])
        return code

    def encode(self, evidence: Sequence[Tuple[Tuple[Q, Q], Q]]) -> np.ndarray:
        rows = [self.row(float(a), float(b), float(c)) for (a, b), c in evidence]
        s = np.sum(rows, axis=0) if rows else np.zeros(self.dim, dtype=complex)
        n = np.linalg.norm(s)
        return s / n if n > 0 else s

    def real(self, evidence: Sequence[Tuple[Tuple[Q, Q], Q]]) -> np.ndarray:
        """Real view [Re, Im] (length 2 dim) for the sparse expansion."""
        s = self.encode(evidence)
        return np.concatenate([s.real, s.imag])

    def unbind(self, signature: np.ndarray, role: str) -> np.ndarray:
        """Recover the bundled relation channel of one role (exact inverse of the binding)."""
        return np.conj(self.roles[role]) * signature

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.real(np.vdot(a, b)))


def signature_similarity_report(enc: SignatureEncoder, same: List[Sequence], different: List[Sequence]) -> Tuple[float, float]:
    """Mean similarity within a list of evidence sets from one operator vs across operators."""
    sims_same = [enc.similarity(enc.encode(a), enc.encode(b)) for i, a in enumerate(same) for b in same[i + 1:]]
    sims_diff = [enc.similarity(enc.encode(a), enc.encode(b)) for a in same for b in different]
    return float(np.mean(sims_same)), float(np.mean(sims_diff))
