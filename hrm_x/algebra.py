"""Phasor (FHRR) binding algebra and character codes (doc 03, sections 4 and 10).

Tensor contract (reference, numpy):
  code / structure : complex128, shape (d,), |z_k| = 1 for codes; structures are sums of bound codes
  codebook         : complex128, shape (M, d), rows unit-modulus
ReferenceAlgebra is the oracle; OptimizedAlgebra (torch complex64, batched) is checked against it.

torch is imported lazily inside OptimizedAlgebra.__init__ so that importing this module -- and hence
hrm_x -- never pulls in torch; only the batched fast path pays for it.

Where this sits in the repo: this is the FHRR binding used by the capacity experiment E1 and by role
binding. It is the tensor half of HRM-X, complementary to the exact-rational story model in
hrm_x.situation: here role/value binding is elementwise multiplication in U(1)^d, cleanup against a
codebook is the retrieval rule, and quantities enter as character codes phi_Omega(x) = exp(i Omega x).

Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


class ReferenceAlgebra:
    """bind = elementwise product (group law of U(1)^d), unbind = product with the conjugate (inverse)."""

    def __init__(self, d: int, seed: int = 0) -> None:
        self.d = d
        self.rng = np.random.default_rng(seed)

    def random_codes(self, m: int) -> np.ndarray:
        """m independent uniform-phase codes, shape (m, d). E[<u, v>/d] = 0 for u != v."""
        return np.exp(1j * self.rng.uniform(-np.pi, np.pi, size=(m, self.d)))

    @staticmethod
    def bind(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        assert u.shape == v.shape and u.shape[-1] > 0
        return u * v

    @staticmethod
    def unbind(role: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Exact inverse on unit-modulus codes: unbind(r, bind(r, f)) = f."""
        assert role.shape == s.shape
        return np.conj(role) * s

    @staticmethod
    def superpose(items: Sequence[np.ndarray]) -> np.ndarray:
        return np.sum(np.stack(items), axis=0)

    @staticmethod
    def similarity(codebook: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Re<c_i, z>/d for each codebook row; 1 for an exact match, ~N(0, 1/(2d)) for a random code."""
        assert codebook.ndim == 2 and codebook.shape[1] == z.shape[0]
        return np.real(np.conj(codebook) @ z) / z.shape[0]

    def cleanup(self, codebook: np.ndarray, z: np.ndarray) -> int:
        return int(np.argmax(self.similarity(codebook, z)))

    # ----- character codes for quantities (doc 03, Theorem 4 / section 1) -----
    @staticmethod
    def character(omega: np.ndarray, x: float) -> np.ndarray:
        """phi_Omega(x) = exp(i Omega x): a homomorphism (R,+) -> U(1)^d, phi(x+y) = phi(x) phi(y)."""
        return np.exp(1j * omega * x)

    @staticmethod
    def scale(omega: np.ndarray, s: float) -> np.ndarray:
        """Multiplication by s acts on the dual: phi_Omega(s x) = phi_{s Omega}(x) (frequency scaling)."""
        return omega * s


def cleanup_error_bound(d: int, k: int, m: int) -> float:
    """Prop. 8: P(wrong cleanup) <= (m-1) exp(-d / (4 (k-1))) for k superposed pairs, m fillers."""
    if k <= 1:
        return 0.0
    return min(1.0, (m - 1) * math.exp(-d / (4.0 * (k - 1))))


class OptimizedAlgebra:
    """Batched torch complex64 version; semantics identical to ReferenceAlgebra within float32 error."""

    def __init__(self, device: Optional[str] = None) -> None:
        import torch
        self.torch = torch
        self.device = device or "cpu"

    def bind(self, u, v):
        assert u.shape == v.shape
        return u * v

    def unbind(self, role, s):
        return self.torch.conj(role) * s

    def similarity(self, codebook, z):
        """codebook (M, d), z (B, d) -> (B, M)."""
        assert codebook.ndim == 2 and z.ndim == 2 and codebook.shape[1] == z.shape[1]
        return (z @ self.torch.conj(codebook).T).real / z.shape[1]
