"""Phase quantization: unit phasors stored as uint16 angle codes.

A unit phasor carries information only in its phase, so complex64 (8 bytes
per dimension) compresses 4x to uint16 codes with worst-case phase error
pi/65536 ~ 4.8e-5 rad, far below superposition crosstalk. Used for archival
of knowledge keys and snapshots; the hot path stays complex64.
"""
from __future__ import annotations

import numpy as np

_SCALE = 65536.0 / (2.0 * np.pi)


def encode_angles(v: np.ndarray) -> np.ndarray:
    """complex array -> uint16 angle codes (same shape)."""
    codes = np.round((np.angle(v) + np.pi) * _SCALE).astype(np.int64) % 65536
    return codes.astype(np.uint16)


def decode_angles(codes: np.ndarray, dtype: str = "complex64") -> np.ndarray:
    """uint16 angle codes -> unit phasors."""
    theta = codes.astype(np.float32) / np.float32(_SCALE) - np.float32(np.pi)
    return np.exp(1j * theta).astype(dtype)


def quantization_error(v: np.ndarray) -> float:
    """Max |v - decode(encode(v))|."""
    return float(np.max(np.abs(v - decode_angles(encode_angles(v), str(v.dtype)))))
