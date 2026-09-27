"""Operators whose Lipschitz constants are known exactly from their parameters (T5, S1, N1)."""

import torch
from torch import Tensor, nn


def cayley(raw: Tensor) -> Tensor:
    """Orthogonal (I + S)^-1 (I - S) from the skew part S of a free square matrix (T5)."""
    skew = raw - raw.T
    eye = torch.eye(raw.shape[0], dtype=raw.dtype, device=raw.device)
    return torch.linalg.solve(eye + skew, eye - skew)


class CertifiedLinear(nn.Module):
    """Square map W = Q1 diag(s) Q2 with s = budget * sigmoid(logit), so ||W||_2 = max s < budget."""

    def __init__(self, dim: int, budget: float):
        super().__init__()
        if not budget > 0.0:
            raise ValueError(f"budget must be positive, got {budget}")
        self.budget = float(budget)
        self.left = nn.Parameter(torch.randn(dim, dim) / dim)
        self.right = nn.Parameter(torch.randn(dim, dim) / dim)
        self.logit = nn.Parameter(torch.zeros(dim))

    def scales(self) -> Tensor:
        """Singular values of W."""
        return self.budget * torch.sigmoid(self.logit)

    def weight(self) -> Tensor:
        """Materialized W."""
        return (cayley(self.left) * self.scales()) @ cayley(self.right)

    def spectral_norm(self) -> float:
        """||W||_2 of the matrix actually used in compute (float64 SVD on CPU)."""
        with torch.no_grad():
            return float(torch.linalg.matrix_norm(self.weight().detach().cpu().double(), ord=2))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight().T


def substochastic(route: Tensor) -> Tensor:
    """Scale a nonnegative (..., n, n) matrix so row and column sums are <= 1, hence ||A||_2 <= 1 (S1)."""
    if bool((route < 0).any()):
        raise ValueError("routing matrix must be nonnegative")
    rows = route.sum(dim=-1).amax(dim=-1)
    cols = route.sum(dim=-2).amax(dim=-1)
    scale = torch.clamp(torch.maximum(rows, cols), min=1.0)
    return route / scale[..., None, None]


RADIUS_EPS = 1e-6


def radial_relu(x: Tensor, shrink: float) -> Tensor:
    """Per quaternion max(0, ρ - shrink) x/ρ with ρ = sqrt(|x|^2 + ε^2): 1-Lipschitz, equivariant (N1).

    The smoothed radius keeps the gradient finite at x = 0 and is the identity when shrink = 0.
    """
    if shrink < 0.0:
        raise ValueError(f"shrink must be >= 0, got {shrink}")
    blocks = x.reshape(*x.shape[:-1], -1, 4)
    radius = torch.sqrt(blocks.pow(2).sum(dim=-1, keepdim=True) + RADIUS_EPS ** 2)
    return (blocks * (torch.relu(radius - shrink) / radius)).reshape(x.shape)
