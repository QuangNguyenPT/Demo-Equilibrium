"""Attention Residuals (arXiv:2603.15031) over a chain of states: softmax over depth (spec A1)."""

import torch
from torch import Tensor, nn


def rms_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm without a learned scale."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class DepthAttention(nn.Module):
    """u_q = Σ_i softmax_i(w_q · RMSNorm(v_i)) v_i with zero-initialized pseudo-queries w_q."""

    def __init__(self, dim: int, queries: int = 1):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(queries, dim))

    def weights(self, chain: Tensor) -> Tensor:
        """(queries, S, ...) depth weights for a (S, ..., dim) chain; each sums to 1 over S."""
        logits = torch.einsum("qd,s...d->qs...", self.query, rms_normalize(chain))
        return logits.softmax(dim=1)

    def forward(self, chain: Tensor) -> Tensor:
        """(S, ..., dim) chain -> (queries, ..., dim) convex combinations of its states."""
        return torch.einsum("qs...,s...d->q...d", self.weights(chain), chain)
