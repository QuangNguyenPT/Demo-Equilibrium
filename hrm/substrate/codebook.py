"""Deterministic symbol store and cleanup memory.

Every symbol is derived from (seed, name, period), not from creation order,
so any section (grid perception, relational facts, the language bridge) can
mint symbols independently and two models with the same seed agree on every
symbol. Symbols and fractional powers share one byte-bounded LRU; an evicted
entry is regenerated bit-identically on the next request. Cached arrays are
read-only so no caller can corrupt a symbol in place.
"""
from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple

import numpy as np

from hrm.config import SubstrateConfig
from hrm.substrate.fhrr import PhasorVector, similarity_matrix
from hrm.substrate.pool import ByteLRUCache


class Codebook:
    def __init__(self, cfg: SubstrateConfig):
        self.dim = cfg.dim
        self.seed = cfg.seed
        self.dtype = cfg.dtype
        self._cache = ByteLRUCache(cfg.codebook_cache_mb * 2 ** 20)
        self._fresh_rng = np.random.default_rng([cfg.seed, _name_hash("__fresh__")])

    def symbol(self, name: str) -> PhasorVector:
        """Atomic symbol with i.i.d. uniform phases (near-orthogonal to all other names)."""
        return self._get(("sym", name, 0))

    def circular(self, name: str, period: int) -> PhasorVector:
        """Symbol whose phases are multiples of 2*pi/period, so ``c ** period`` is the
        identity: cyclic quantities (colors, weekdays) become single-step rotations."""
        if period < 2:
            raise ValueError("circular period must be >= 2")
        return self._get(("circ", name, period))

    def power(self, name: str, exponent: float, period: int = 0) -> PhasorVector:
        """``symbol(name) ** exponent``, or ``circular(name, period) ** exponent``."""
        key = ("pow", name, period, round(float(exponent), 9))
        hit = self._cache.get(key)
        if hit is not None:
            return PhasorVector(hit)
        base = self.circular(name, period) if period else self.symbol(name)
        return self._store(key, base.power(exponent))

    def fresh(self) -> PhasorVector:
        """New random symbol not tied to any name."""
        return PhasorVector.random(self.dim, self._fresh_rng, self.dtype)

    @staticmethod
    def cleanup(query: PhasorVector, candidates: Sequence[Tuple[str, PhasorVector]],
                top_k: int = 1) -> List[Tuple[str, float]]:
        """Snap a noisy vector to its nearest named candidates."""
        if not candidates:
            return []
        keys = np.stack([v.data for _, v in candidates])
        sims = similarity_matrix(query.data, keys)[0]
        order = np.argsort(-sims)[:top_k]
        return [(candidates[i][0], float(sims[i])) for i in order]

    def _get(self, key: Tuple[str, str, int]) -> PhasorVector:
        hit = self._cache.get(key)
        if hit is not None:
            return PhasorVector(hit)
        kind, name, period = key
        rng = np.random.default_rng([self.seed, _name_hash(name), period])
        if kind == "sym":
            vec = PhasorVector.random(self.dim, rng, self.dtype)
        else:
            steps = rng.integers(1, period, size=self.dim)   # nonzero step -> invertible
            vec = PhasorVector.from_angles(2.0 * np.pi * steps.astype(np.float64) / period,
                                           self.dtype)
        return self._store(key, vec)

    def _store(self, key: tuple, vec: PhasorVector) -> PhasorVector:
        vec.data.flags.writeable = False
        self._cache.put(key, vec.data)
        return vec


def _name_hash(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "little")
