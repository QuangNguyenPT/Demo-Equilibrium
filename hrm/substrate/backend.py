"""Kernel backend dispatch.

`KernelBackend` is the numpy reference implementation and the correctness
oracle. A faster backend (native C++/CUDA) is installed only through
`set_backend`, which refuses any candidate that does not reproduce the
reference on a fixed probe set: a wrong kernel produces fluent wrong answers,
not crashes, so it must never serve traffic unchecked.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

log = logging.getLogger("hrm.backend")

_EPS = 1e-12


class BackendRejected(RuntimeError):
    """A candidate backend disagreed with the numpy reference."""


class KernelBackend:
    """Numpy reference. Subclasses must match it within float32 tolerance."""

    name = "numpy"

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def torus_project(self, v: np.ndarray) -> np.ndarray:
        # exact cancellation can leave a zero element: map it to 1+0j
        # (deterministic, keeps downstream binds valid) instead of dividing by zero
        mag = np.abs(v)
        out = np.where(mag > _EPS, v / np.maximum(mag, _EPS), np.ones_like(v))
        return out.astype(v.dtype)

    def bundle(self, stack: np.ndarray) -> np.ndarray:
        return self.torus_project(stack.sum(axis=0).astype(stack.dtype))

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.vdot(b, a).real / a.shape[0])

    def sinkhorn(self, logits: np.ndarray, iters: int) -> np.ndarray:
        """Log-domain Sinkhorn-Knopp projection onto the Birkhoff polytope.

        Double stochasticity bounds the spectral norm of the mixing matrix,
        which the KM convergence guarantee depends on, so `iters` is a
        minimum: iteration continues until the row marginals are within
        1e-6 (capped at 10x), and the result is scaled to be sub-stochastic
        if extreme logits still leave residual error.
        """
        m = logits.astype(np.float64).copy()
        for it in range(10 * iters):
            m -= _logsumexp(m, axis=1)
            m -= _logsumexp(m, axis=0)
            if it + 1 >= iters and np.abs(np.exp(m).sum(axis=1) - 1.0).max() < 1e-6:
                break
        out = np.exp(m)
        excess = max(out.sum(axis=0).max(), out.sum(axis=1).max(), 1.0)
        return out / excess


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    mx = np.max(x, axis=axis, keepdims=True)
    return mx + np.log(np.sum(np.exp(x - mx), axis=axis, keepdims=True))


def cross_validate(candidate: KernelBackend) -> List[str]:
    """Names of the probes where `candidate` disagrees with the reference."""
    ref = KernelBackend()
    rng = np.random.default_rng(0)
    a = np.exp(1j * rng.uniform(0, 2 * np.pi, 256)).astype(np.complex64)
    b = np.exp(1j * rng.uniform(0, 2 * np.pi, 256)).astype(np.complex64)
    with_zeros = (a * np.float32(3.0)).astype(np.complex64)
    with_zeros[:4] = 0
    stack = np.stack([a, b, np.conj(a)])
    logits = rng.standard_normal((4, 4)) * 3.0

    failures: List[str] = []

    def check(name: str, got, want, atol: float) -> None:
        if not np.allclose(np.asarray(got), np.asarray(want), atol=atol):
            failures.append(name)

    try:
        check("bind", candidate.bind(a, b), ref.bind(a, b), 1e-5)
        check("unbind_round_trip", candidate.bind(candidate.bind(a, b), np.conj(b)), a, 1e-4)
        check("torus_project", candidate.torus_project(with_zeros),
              ref.torus_project(with_zeros), 1e-5)
        check("bundle", candidate.bundle(stack), ref.bundle(stack), 1e-4)
        check("similarity", candidate.similarity(a, b), ref.similarity(a, b), 1e-4)
        check("sinkhorn", candidate.sinkhorn(logits, 20), ref.sinkhorn(logits, 20), 1e-5)
    except Exception as exc:  # noqa: BLE001 - any fault in untrusted kernels means reject
        failures.append(f"raised {type(exc).__name__}: {exc}")
    return failures


_ACTIVE: KernelBackend = KernelBackend()


def get_backend() -> KernelBackend:
    return _ACTIVE


def set_backend(backend: KernelBackend) -> KernelBackend:
    """Install `backend` only if it reproduces the numpy reference."""
    global _ACTIVE
    failures = cross_validate(backend)
    if failures:
        raise BackendRejected(f"backend '{backend.name}' failed validation: {failures}")
    _ACTIVE = backend
    log.info("active kernel backend: %s", backend.name)
    return backend
