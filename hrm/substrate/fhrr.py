"""FHRR phasor algebra.

A symbol is a D-vector of unit-magnitude complex numbers. Native kernels see
the same memory as interleaved (re, im) float pairs, so nothing is converted
at the FFI boundary.

    bind        elementwise complex multiply   (phase addition)
    unbind      bind with the conjugate        (exact inverse at any D)
    power       fractional binding e^{i x theta}
    bundle      superposition + torus projection
    similarity  Re<a, b> / D  in [-1, 1]
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from hrm.substrate.backend import get_backend


class PhasorVector:
    """Unit-phasor vector. ``a * b`` bind, ``a / b`` unbind, ``a ** x`` power,
    ``~a`` conjugate, ``a @ b`` similarity."""

    __slots__ = ("data",)

    def __init__(self, data: np.ndarray, *, check: bool = False):
        if data.ndim != 1 or not np.iscomplexobj(data):
            raise TypeError("PhasorVector wraps a 1-D complex array")
        if check and not np.allclose(np.abs(data), 1.0, atol=1e-4):
            raise ValueError("phasor magnitudes deviate from the unit circle")
        self.data = data

    @classmethod
    def random(cls, dim: int, rng: np.random.Generator,
               dtype: str = "complex64") -> "PhasorVector":
        return cls.from_angles(rng.uniform(0.0, 2.0 * np.pi, size=dim), dtype)

    @classmethod
    def identity(cls, dim: int, dtype: str = "complex64") -> "PhasorVector":
        return cls(np.ones(dim, dtype=dtype))

    @classmethod
    def from_angles(cls, theta: np.ndarray, dtype: str = "complex64") -> "PhasorVector":
        return cls(np.exp(1j * theta.astype(np.float64)).astype(dtype))

    @property
    def dim(self) -> int:
        return self.data.shape[0]

    @property
    def angles(self) -> np.ndarray:
        return np.angle(self.data)

    def bind(self, other: "PhasorVector") -> "PhasorVector":
        return PhasorVector(get_backend().bind(self.data, other.data))

    def unbind(self, other: "PhasorVector") -> "PhasorVector":
        return PhasorVector(get_backend().bind(self.data, np.conj(other.data)))

    def conj(self) -> "PhasorVector":
        return PhasorVector(np.conj(self.data))

    def power(self, x: float) -> "PhasorVector":
        """Rotate every phase by factor x; exact for any real x."""
        return PhasorVector.from_angles(self.angles * float(x), dtype=str(self.data.dtype))

    def similarity(self, other: "PhasorVector") -> float:
        return get_backend().similarity(self.data, other.data)

    def noisy(self, sigma: float, rng: np.random.Generator) -> "PhasorVector":
        """Phase-space perturbation z * exp(i sigma xi): the restart kick."""
        xi = rng.standard_normal(self.dim)
        return PhasorVector.from_angles(self.angles + sigma * xi, dtype=str(self.data.dtype))

    def __mul__(self, other: "PhasorVector") -> "PhasorVector":
        return self.bind(other)

    def __truediv__(self, other: "PhasorVector") -> "PhasorVector":
        return self.unbind(other)

    def __pow__(self, x: float) -> "PhasorVector":
        return self.power(x)

    def __invert__(self) -> "PhasorVector":
        return self.conj()

    def __matmul__(self, other: "PhasorVector") -> float:
        return self.similarity(other)

    def __repr__(self) -> str:
        return f"PhasorVector(dim={self.dim}, dtype={self.data.dtype})"


def bundle(vectors: Sequence[PhasorVector] | Iterable[PhasorVector],
           weights: np.ndarray | None = None) -> PhasorVector:
    """(Weighted) superposition projected back to the torus.

    Per-item retrieval SNR ~ sqrt(D / N): callers own the capacity budget.
    """
    vecs = list(vectors)
    if not vecs:
        raise ValueError("cannot bundle zero vectors")
    stack = np.stack([v.data for v in vecs])
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != stack.shape[0]:
            raise ValueError("weights/vector count mismatch")
        stack = (stack * w[:, None]).astype(stack.dtype)
    return PhasorVector(get_backend().bundle(stack))


def torus_project(v: np.ndarray) -> np.ndarray:
    return get_backend().torus_project(v)


def similarity_matrix(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """(Q, D) x (K, D) -> (Q, K) real similarities in [-1, 1]."""
    if queries.ndim == 1:
        queries = queries[None, :]
    return (queries @ np.conj(keys).T).real / queries.shape[-1]
