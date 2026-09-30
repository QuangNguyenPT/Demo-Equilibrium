"""Native backend (C ABI over ctypes) for the measured hot paths; Python modules stay the reference.

Built by scripts/build_native.ps1 into hrm_x/native/bin/. Every entry point here has a Python
twin with identical results (tests/test_x_native.py checks this on random inputs).
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, c_int32, c_int64, c_void_p
from pathlib import Path
from typing import Dict, Hashable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from hrm_x.workspace import EGraphAPI, ENode, Why

LIB_PATH = Path(__file__).resolve().parent / "bin" / "hrmx_native.dll"
_lib = None

I32P = POINTER(c_int32)
F64P = POINTER(ctypes.c_double)


def _load():
    global _lib
    if _lib is not None or not LIB_PATH.exists():
        return _lib
    lib = ctypes.CDLL(str(LIB_PATH))
    sig = {
        "hx_version": (c_int32, []),
        "hx_components": (c_int32, [I32P, c_int32, c_int32, I32P]),
        "hx_object_step": (c_int32, [I32P, c_int32, c_int32, c_int32, c_int32, c_int32, c_int32, c_int32,
                                     c_int32, I32P]),
        "hx_eg_new": (c_void_p, [c_int32]),
        "hx_eg_free": (None, [c_void_p]),
        "hx_eg_add": (c_int32, [c_void_p, c_int32, I32P, c_int32, c_int64]),
        "hx_eg_find": (c_int32, [c_void_p, c_int32]),
        "hx_eg_union": (c_int32, [c_void_p, c_int32, c_int32, c_int32, I32P, c_int32]),
        "hx_eg_rebuild": (None, [c_void_p, I32P, I32P]),
        "hx_eg_take_conflicts": (c_int32, [c_void_p, I32P, c_int32]),
        "hx_eg_lookup": (c_int32, [c_void_p, c_int32, I32P, c_int32, c_int64]),
        "hx_eg_const_node": (c_int32, [c_void_p, c_int32]),
        "hx_eg_num_nodes": (c_int32, [c_void_p]),
        "hx_eg_classes": (c_int32, [c_void_p, I32P, c_int32]),
        "hx_eg_members": (c_int32, [c_void_p, c_int32, I32P, c_int32]),
        "hx_eg_node": (c_int32, [c_void_p, c_int32, I32P, POINTER(c_int64), I32P, c_int32, c_int32]),
        "hx_eg_explain": (c_int32, [c_void_p, c_int32, c_int32, I32P, c_int32]),
        "hx_eg_snapshot_index": (None, [c_void_p]),
        "hx_eg_ematch": (c_int32, [c_void_p, I32P, c_int32, c_int32, c_int32, I32P, c_int32]),
        "hx_enumerate": (c_int32, [c_int32, F64P, F64P, c_int32, I32P, c_int32, I32P, F64P, c_int32]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    _lib = lib
    return lib


def available() -> bool:
    return _load() is not None


def enabled() -> bool:
    """Native execution is used unless HRMX_BACKEND=python (or the library is not built)."""
    import os
    return os.environ.get("HRMX_BACKEND", "auto") != "python" and available()


def _arr(values: Sequence[int]):
    a = (c_int32 * max(1, len(values)))(*values)
    return a


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(I32P)


# ---------- grid ----------

def components(grid: np.ndarray) -> Tuple[np.ndarray, int]:
    g = np.ascontiguousarray(grid, dtype=np.int32)
    labels = np.empty_like(g)
    n = _load().hx_components(_ptr(g), g.shape[0], g.shape[1], _ptr(labels))
    return labels, n


SELECTORS = {"all": 0, "color": 1, "largest": 2, "smallest": 3}
ACTIONS = {"keep": 0, "delete": 1, "recolor": 2, "move": 3, "flip": 4}


def object_step(selector: Tuple, action: Tuple, grid: np.ndarray, strict: int) -> Optional[np.ndarray]:
    """One (selector, action) step; None when the Python twin would raise Invalid."""
    g = np.ascontiguousarray(grid, dtype=np.int32)
    out = np.empty_like(g)
    kind = action[0]
    a1 = a2 = 0
    if kind == "recolor":
        a1 = action[1]
    elif kind == "move":
        a1, a2 = action[1], action[2]
    elif kind == "flip":
        a1 = 0 if action[1] == "h" else 1
    sel_color = selector[1] if selector[0] == "color" else 0
    rc = _load().hx_object_step(_ptr(g), g.shape[0], g.shape[1], SELECTORS[selector[0]], sel_color,
                                ACTIONS[kind], a1, a2, int(strict), _ptr(out))
    return None if rc else out.astype(grid.dtype, copy=False)


# ---------- e-graph ----------

class NativeWorkspace(EGraphAPI):
    """The e-graph primitives in C++; ops/payloads are interned to ints, reasons kept here by id."""

    native = True

    def __init__(self) -> None:
        self._lib = _load()
        self._op_ids: Dict[str, int] = {"const": 0}
        self._op_names: List[str] = ["const"]
        self._pay_ids: Dict[Hashable, int] = {None: 0}
        self._payloads: List[Hashable] = [None]
        self._whys: List[Why] = []
        self._h = self._lib.hx_eg_new(0)
        self.conflicts = []
        self.ops = 0
        self._buf = (c_int32 * 64)()

    def __del__(self) -> None:
        if getattr(self, "_h", None):
            self._lib.hx_eg_free(self._h)
            self._h = None

    def _op(self, name: str) -> int:
        k = self._op_ids.get(name)
        if k is None:
            k = self._op_ids[name] = len(self._op_names)
            self._op_names.append(name)
        return k

    def _pid(self, payload: Hashable) -> int:
        k = self._pay_ids.get(payload)
        if k is None:
            k = self._pay_ids[payload] = len(self._payloads)
            self._payloads.append(payload)
        return k

    # primitives
    def find(self, i: int) -> int:
        return self._lib.hx_eg_find(self._h, i)

    def add(self, op: str, args: Sequence[int] = (), payload: Hashable = None) -> int:
        return self._lib.hx_eg_add(self._h, self._op(op), _arr(args), len(args), self._pid(payload))

    def classes(self) -> List[int]:
        n = self._lib.hx_eg_classes(self._h, None, 0)
        out = (c_int32 * max(1, n))()
        self._lib.hx_eg_classes(self._h, out, n)
        return list(out[:n])

    def _members(self, c: int) -> List[int]:
        n = self._lib.hx_eg_members(self._h, c, self._buf, 64)
        if n > 64:
            big = (c_int32 * n)()
            self._lib.hx_eg_members(self._h, c, big, n)
            return list(big[:n])
        return list(self._buf[:n])

    def _node(self, i: int, canonical: bool) -> ENode:
        op, pay = c_int32(), c_int64()
        args = (c_int32 * 16)()
        n = self._lib.hx_eg_node(self._h, i, ctypes.byref(op), ctypes.byref(pay), args, 16, int(canonical))
        return self._op_names[op.value], tuple(args[:n]), self._payloads[pay.value]

    def nodes(self, c: int) -> Iterator[Tuple[int, ENode]]:
        for i in self._members(c):
            yield i, self._node(i, True)

    def lookup(self, op: str, args: Sequence[int] = (), payload: Hashable = None) -> Optional[int]:
        if op not in self._op_ids or payload not in self._pay_ids:
            return None
        hit = self._lib.hx_eg_lookup(self._h, self._op_ids[op], _arr(args), len(args), self._pay_ids[payload])
        return None if hit < 0 else hit

    def const_node(self, c: int) -> Optional[int]:
        k = self._lib.hx_eg_const_node(self._h, c)
        return None if k < 0 else k

    def original_args(self, node: int) -> Tuple[int, ...]:
        return self._node(node, False)[1]

    def payload_of(self, node: int) -> Hashable:
        return self._node(node, False)[2]

    def union(self, a: int, b: int, why: Why) -> bool:
        self.ops += 1
        wid = len(self._whys)
        self._whys.append(why)
        flat = [x for pair in why.premises for x in pair]
        rc = self._lib.hx_eg_union(self._h, a, b, wid, _arr(flat), len(why.premises))
        if rc == -1:
            self.record_conflict(a, b, why)
        return rc == 1

    def rebuild(self) -> int:
        attempts, merges = c_int32(), c_int32()
        self._lib.hx_eg_rebuild(self._h, ctypes.byref(attempts), ctypes.byref(merges))
        self.ops += attempts.value
        while True:
            pairs = (c_int32 * 128)()
            n = self._lib.hx_eg_take_conflicts(self._h, pairs, 64)
            if n == 0:
                break
            for k in range(n):
                j, i = pairs[2 * k], pairs[2 * k + 1]
                self.record_conflict(j, i, Why("congruence", nodes=(j, i)))
        return merges.value

    def explain(self, a: int, b: int) -> List[Why]:
        if a == b:
            return []
        n = self._lib.hx_eg_explain(self._h, a, b, None, 0)
        out = (c_int32 * max(1, n))()
        self._lib.hx_eg_explain(self._h, a, b, out, n)
        return [self._whys[k] for k in out[:n]]

    # e-matching
    def snapshot_index(self) -> None:
        self._lib.hx_eg_snapshot_index(self._h)

    def compile(self, pattern) -> Optional[Tuple[List[int], List[str]]]:
        """Pattern -> preorder code; None if it uses a construct the native matcher does not have."""
        from hrm_x.rules import Lit
        code: List[int] = []
        slots: List[str] = []

        def emit(p) -> bool:
            if isinstance(p, str):
                if p not in slots:
                    slots.append(p)
                code.extend([0, slots.index(p)])
                return True
            if isinstance(p, Lit):
                if p.value is not None:
                    return False
                code.append(1)
                return True
            op, *kids = p
            code.extend([2, self._op(op), len(kids)])
            return all(emit(k) for k in kids)

        return (code, slots) if emit(pattern) else None

    def ematch(self, compiled: Tuple[List[int], List[str]], use_index: bool) -> List[Tuple[int, Dict[str, int]]]:
        code, slots = compiled
        stride = 1 + len(slots)
        cap = 4096
        while True:
            out = (c_int32 * (cap * stride))()
            n = self._lib.hx_eg_ematch(self._h, _arr(code), len(code), len(slots), int(use_index), out, cap)
            if n < cap:
                break
            cap *= 4
        return [(out[k * stride], {s: out[k * stride + 1 + j] for j, s in enumerate(slots)}) for k in range(n)]


# ---------- construct ----------

def enumerate_bank(max_size: int, xs: np.ndarray, ys: np.ndarray) -> Optional[List[Tuple[Tuple[int, ...], np.ndarray]]]:
    """(expr, probe values) in the order of construct.enumerate_bank, or None if not built."""
    lib = _load()
    if lib is None:
        return None
    px = np.ascontiguousarray(xs, dtype=np.float64)
    py = np.ascontiguousarray(ys, dtype=np.float64)
    n = len(px)
    cap = 1 << 16
    while True:
        toks = np.zeros(cap * max_size, dtype=np.int32)
        offs = np.zeros(cap + 1, dtype=np.int32)
        vals = np.zeros(cap * n, dtype=np.float64)
        count = lib.hx_enumerate(max_size, px.ctypes.data_as(F64P), py.ctypes.data_as(F64P), n,
                                 _ptr(toks), len(toks), _ptr(offs), vals.ctypes.data_as(F64P), cap)
        if count >= 0:
            break
        cap *= 4
    return [(tuple(int(t) for t in toks[offs[i]:offs[i + 1]]), vals[i * n:(i + 1) * n].copy()) for i in range(count)]
