"""Byte-bounded caches for the substrate."""
from __future__ import annotations

from collections import OrderedDict
from typing import Hashable, Optional

import numpy as np


class ByteLRUCache:
    """LRU bounded by bytes, not entry count: at D=4096 one phasor is 32 KB,
    so a count-bounded cache could quietly grow to gigabytes."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max(int(max_bytes), 1)
        self._data: "OrderedDict[Hashable, np.ndarray]" = OrderedDict()
        self._bytes = 0

    def get(self, key: Hashable) -> Optional[np.ndarray]:
        hit = self._data.get(key)
        if hit is not None:
            self._data.move_to_end(key)
        return hit

    def put(self, key: Hashable, value: np.ndarray) -> None:
        if value.nbytes > self.max_bytes:
            return
        old = self._data.pop(key, None)
        if old is not None:
            self._bytes -= old.nbytes
        self._data[key] = value
        self._bytes += value.nbytes
        while self._bytes > self.max_bytes:
            _, evicted = self._data.popitem(last=False)
            self._bytes -= evicted.nbytes

    def __len__(self) -> int:
        return len(self._data)

    @property
    def nbytes(self) -> int:
        return self._bytes
