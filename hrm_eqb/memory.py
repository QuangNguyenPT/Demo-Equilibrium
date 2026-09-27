"""Knowledge memory: product-key addressing (M1) over quaternion values (K1), bounded reads (K2).

The addressing layers (query, relation, out) are core weights and live in the .hrmc.
keys_a, keys_b and values are knowledge: they live in knowledge.pt and, at runtime, may be
SSD-backed CPU tensors of which a read touches only the top-k rows.
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from hrm_eqb.algebra import qmul
from hrm_eqb.product_keys import product_topk

KNOWLEDGE_TENSORS = ("keys_a", "keys_b", "values")


class KnowledgeMemory(nn.Module):
    """read(u) = W_out( (Σ_i α_i v_i) ⊗ r(u) ) with α = softmax over the product top-k slots."""

    def __init__(self, dim: int, side: int, topk: int, key_dim: int, value_quaternions: int):
        super().__init__()
        self.side, self.topk, self.value_quaternions = side, topk, value_quaternions
        self.query = nn.Linear(dim, key_dim)
        self.relation = nn.Linear(dim, 4 * value_quaternions)
        self.out = nn.Linear(4 * value_quaternions, dim)
        self.keys_a = nn.Parameter(torch.randn(side, key_dim // 2) / math.sqrt(key_dim // 2))
        self.keys_b = nn.Parameter(torch.randn(side, key_dim // 2) / math.sqrt(key_dim // 2))
        self.values = nn.Parameter(torch.randn(side * side, 4 * value_quaternions) * 0.02)
        self.overlay: Optional[Tensor] = None
        self.silent = False           # empty-brain mode: every read returns zero knowledge

    @property
    def slots(self) -> int:
        return self.side * self.side

    def address(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        """(weights, slots) of the top-k knowledge rows for each reader; weights sum to 1."""
        qa, qb = self.query(u).chunk(2, dim=-1)
        keys_a, keys_b = self.keys_a.to(u.device), self.keys_b.to(u.device)
        scores, slots = product_topk(qa @ keys_a.T, qb @ keys_b.T, self.topk)
        return scores.softmax(dim=-1), slots

    def rows(self, slots: Tensor) -> Tensor:
        """Value rows for slot indices, fetched from wherever the values live (RAM, VRAM or SSD map)."""
        if self.silent:
            return torch.zeros(*slots.shape, 4 * self.value_quaternions, device=slots.device)
        rows = self.values[slots.to(self.values.device)].to(slots.device, torch.float32)
        if self.overlay is not None:
            rows = rows + self.overlay[slots]
        return rows

    def mixed(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        """(rotated knowledge (..., k_v, 4) before the out projection, slots read)."""
        weights, slots = self.address(u)
        blend = (weights.unsqueeze(-1) * self.rows(slots)).sum(dim=-2)
        rel = self.relation(u).unflatten(-1, (self.value_quaternions, 4))
        rel = rel / torch.linalg.vector_norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)
        return qmul(blend.unflatten(-1, (self.value_quaternions, 4)), rel), slots

    def forward(self, u: Tensor) -> Tensor:
        return self.out(self.mixed(u)[0].flatten(-2))

    def attach(self, keys_a: Tensor, keys_b: Tensor, values: Tensor) -> None:
        """Swap in knowledge tensors from a knowledge file (not trained as parameters)."""
        for name, tensor in zip(KNOWLEDGE_TENSORS, (keys_a, keys_b, values)):
            if name in self._parameters:
                del self._parameters[name]
            setattr(self, name, tensor)

    def knowledge(self) -> dict:
        """Current knowledge tensors, detached, on CPU."""
        return {name: getattr(self, name).detach().to("cpu", torch.float32).contiguous()
                for name in KNOWLEDGE_TENSORS}
