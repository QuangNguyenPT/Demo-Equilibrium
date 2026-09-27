"""One state, two algebras: 2k complex planes (text, positions) = k quaternions (knowledge).

A quaternion (w, x, y, z) is the complex pair (w + xi, y + zi) (Cayley-Dickson), so
consecutive real pairs are complex planes in both views. Spec §2 (T1-T4), §4 (K1), §5 (L1, L2).
"""

import math

import torch
from torch import Tensor


def as_complex(x: Tensor) -> Tensor:
    """(..., 2m) real -> (..., m) complex, pairing consecutive reals."""
    if x.shape[-1] % 2:
        raise ValueError(f"last dim must be even, got {x.shape[-1]}")
    return torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())


def as_real(z: Tensor) -> Tensor:
    """(..., m) complex -> (..., 2m) real; inverse of as_complex."""
    return torch.view_as_real(z).flatten(-2)


def rope(x: Tensor, pos: Tensor, freqs: Tensor) -> Tensor:
    """Rotary position = binding with the phasor e^{i pos freqs} (T1)."""
    angle = torch.as_tensor(pos, dtype=freqs.dtype).unsqueeze(-1) * freqs
    return as_real(as_complex(x) * torch.polar(torch.ones_like(angle), angle))


def phasor_codebook(size: int, planes: int, seed: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """(size, planes) i.i.d. uniform unit phasors from a seeded generator (L1)."""
    gen = torch.Generator().manual_seed(seed)
    angle = torch.rand(size, planes, generator=gen, dtype=dtype) * (2.0 * math.pi)
    return torch.polar(torch.ones_like(angle), angle)


def phasor_attention(query: Tensor, keys: Tensor, beta: float) -> Tensor:
    """softmax_c(beta Re<q, k_c>) over a (C, m) complex key set (T2)."""
    return (beta * (query @ keys.conj().T).real).softmax(dim=-1)


def cleanup(state: Tensor, codebook: Tensor) -> Tensor:
    """Index of the codebook entry with the largest Re<state, e_c> (L2 decoding)."""
    return (state @ codebook.conj().T).real.argmax(dim=-1)


def qmul(p: Tensor, q: Tensor) -> Tensor:
    """Hamilton product on (..., 4) tensors laid out (w, x, y, z)."""
    pw, px, py, pz = p.unbind(-1)
    qw, qx, qy, qz = q.unbind(-1)
    return torch.stack((
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ), dim=-1)


def qconj(q: Tensor) -> Tensor:
    """Quaternion conjugate (w, -x, -y, -z)."""
    return q * q.new_tensor((1.0, -1.0, -1.0, -1.0))


def rotate3(q: Tensor, v: Tensor) -> Tensor:
    """3-D rotation v -> q v q̄ of (..., 3) vectors by unit q (T4a)."""
    pure = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return qmul(qmul(q, pure), qconj(q))[..., 1:]


def rotate4(left: Tensor, right: Tensor, x: Tensor) -> Tensor:
    """4-D rotation x -> p x q̄ of (..., 4) vectors by unit p, q (T4b)."""
    return qmul(qmul(left, x), qconj(right))


def relation_distance(head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
    """||head ⊗ relation - tail|| over (..., k, 4) quaternion embeddings (K1)."""
    return torch.linalg.vector_norm(qmul(head, relation) - tail, dim=(-2, -1))
