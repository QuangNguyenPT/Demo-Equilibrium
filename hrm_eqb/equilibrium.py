"""Two-level equilibrium core with a convergence certificate computed from its weights (spec §3).

T_L(l; h) = (1 - a) l + a σ(A_L (l W_ll^T + h W_lh^T) + c_L)      fast level
T_H(h; l) = (1 - b) h + b σ(A_H (h V_hh^T + l V_hl^T) + c_H)      slow level
A cycle runs n_low fast steps, then one slow step (the HRM schedule).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from hrm_eqb.certified import CertifiedLinear, radial_relu, substochastic


@dataclass(frozen=True)
class ContractionBounds:
    """Norm bounds of the four certified maps plus step sizes; decides provable convergence (C1)."""

    s_l: float
    s_h: float
    v_h: float
    v_l: float
    alpha: float
    beta: float

    def __post_init__(self):
        if not 0.0 < self.s_l < 1.0:
            raise ValueError(f"s_l must be in (0, 1), got {self.s_l}")
        if not (self.s_h > 0.0 and self.v_l > 0.0 and self.v_h >= 0.0):
            raise ValueError("coupling bounds s_h, v_l must be positive and v_h nonnegative")
        if not (0.0 < self.alpha <= 1.0 and 0.0 < self.beta <= 1.0):
            raise ValueError("alpha and beta must be in (0, 1]")

    def gamma(self) -> float:
        """Γ = v_h + v_l s_h / (1 - s_l); the model is certified iff Γ < 1."""
        return self.v_h + self.v_l * self.s_h / (1.0 - self.s_l)

    def lipschitz(self) -> Tuple[float, float, float, float]:
        """(ρ_L, κ_L, a, b) = Lip_l T_L, Lip_h T_L, Lip_h T_H, Lip_l T_H."""
        return (1.0 - self.alpha + self.alpha * self.s_l, self.alpha * self.s_h,
                1.0 - self.beta + self.beta * self.v_h, self.beta * self.v_l)

    def gamma_a(self) -> float:
        """Nested contraction rate a + b κ_L / (1 - ρ_L) of h -> T_H(h, l*(h)) (E2)."""
        rho, kappa, a, b = self.lipschitz()
        return a + b * kappa / (1.0 - rho)

    def cycle_matrix(self, n_low: int) -> Tensor:
        """M_n bounding one cycle componentwise (E3)."""
        rho, kappa, a, b = self.lipschitz()
        k_n = kappa * (1.0 - rho ** n_low) / (1.0 - rho)
        return torch.tensor([[a + b * k_n, b * rho ** n_low], [k_n, rho ** n_low]], dtype=torch.float64)

    def perron(self, n_low: int) -> Tuple[float, Tuple[float, float]]:
        """Cycle rate r_n and left Perron weights (1, w_l) of M_n, normalized so w_h = 1 (E3)."""
        if n_low < 1:
            raise ValueError(f"n_low must be >= 1, got {n_low}")
        m = self.cycle_matrix(n_low).tolist()
        trace, det = m[0][0] + m[1][1], m[0][0] * m[1][1] - m[0][1] * m[1][0]
        rate = 0.5 * (trace + math.sqrt(max(trace * trace - 4.0 * det, 0.0)))
        return rate, (1.0, (rate - m[0][0]) / m[1][0])

    def injection_gain(self, n_low: int) -> float:
        """Lipschitz constant of one cycle in (c_L, c_H), weighted norm vs ||Δc_L|| + ||Δc_H|| (E5)."""
        rho, _, _, b = self.lipschitz()
        _, (w_h, w_l) = self.perron(n_low)
        geometric = (1.0 - rho ** n_low) / (1.0 - rho)
        return max(w_h * self.beta, self.alpha * geometric * (w_h * b + w_l))


@dataclass(frozen=True)
class Context:
    """Per-segment routings and injections; routings are made sub-stochastic on construction (S1)."""

    route_low: Tensor
    route_high: Tensor
    inject_low: Tensor
    inject_high: Tensor

    def __post_init__(self):
        object.__setattr__(self, "route_low", substochastic(self.route_low))
        object.__setattr__(self, "route_high", substochastic(self.route_high))


@dataclass(frozen=True)
class Solution:
    """Equilibrium iterate with its certified distance to the true fixed point (E6)."""

    high: Tensor
    low: Tensor
    cycles: int
    error_bound: float
    converged: bool


def weighted_distance(dh: Tensor, dl: Tensor, weights: Tuple[float, float]) -> Tensor:
    """w_h ||Δh|| + w_l ||Δl|| over the last two dims (tokens, features)."""
    return (weights[0] * torch.linalg.vector_norm(dh, dim=(-2, -1))
            + weights[1] * torch.linalg.vector_norm(dl, dim=(-2, -1)))


class EquilibriumCore(nn.Module):
    """Fast and slow levels on (tokens, 4k) states, contractive for every weight value (C1)."""

    def __init__(self, dim: int, budgets: ContractionBounds, shrink: float = 0.0):
        super().__init__()
        if dim % 4:
            raise ValueError(f"dim must be a multiple of 4 (quaternion blocks), got {dim}")
        if not budgets.gamma() < 1.0:
            raise ValueError(f"budgets give Γ = {budgets.gamma():.4f} >= 1: not certifiable")
        self.budgets = budgets
        self.shrink = float(shrink)
        self.low_self = CertifiedLinear(dim, budgets.s_l)
        self.low_from_high = CertifiedLinear(dim, budgets.s_h)
        self.high_self = CertifiedLinear(dim, budgets.v_h)
        self.high_from_low = CertifiedLinear(dim, budgets.v_l)

    def bounds(self) -> ContractionBounds:
        """Certificate constants measured from the current weights (each <= its budget)."""
        b = self.budgets
        return ContractionBounds(
            s_l=self.low_self.spectral_norm(), s_h=self.low_from_high.spectral_norm(),
            v_h=self.high_self.spectral_norm(), v_l=self.high_from_low.spectral_norm(),
            alpha=b.alpha, beta=b.beta)

    def maps(self) -> "CoreMaps":
        """Materialize the four certified matrices once (reused by every step of a solve)."""
        return CoreMaps(self.low_self.weight(), self.low_from_high.weight(),
                        self.high_self.weight(), self.high_from_low.weight())

    def low_step(self, low: Tensor, high: Tensor, ctx: Context, maps: Optional["CoreMaps"] = None,
                 plan: Optional[Tensor] = None) -> Tensor:
        """T_L(l; h); `plan` = h W_lhᵀ may be passed in, since h is fixed during the fast steps."""
        m = maps or self.maps()
        plan = high @ m.low_from_high.T if plan is None else plan
        drive = mix(ctx.route_low, low @ m.low_self.T + plan) + ctx.inject_low
        return (1.0 - self.budgets.alpha) * low + self.budgets.alpha * radial_relu(drive, self.shrink)

    def high_step(self, high: Tensor, low: Tensor, ctx: Context, maps: Optional["CoreMaps"] = None) -> Tensor:
        """T_H(h; l)."""
        m = maps or self.maps()
        drive = mix(ctx.route_high, high @ m.high_self.T + low @ m.high_from_low.T) + ctx.inject_high
        return (1.0 - self.budgets.beta) * high + self.budgets.beta * radial_relu(drive, self.shrink)

    def cycle(self, high: Tensor, low: Tensor, ctx: Context, n_low: int,
              maps: Optional["CoreMaps"] = None) -> Tuple[Tensor, Tensor]:
        """C_n(h, l): n_low fast steps against a fixed plan, then one slow step."""
        m = maps or self.maps()
        plan = high @ m.low_from_high.T
        for _ in range(n_low):
            low = self.low_step(low, high, ctx, m, plan)
        return self.high_step(high, low, ctx, m), low

    def certificate(self, n_low: int) -> Tuple[float, Tuple[float, float]]:
        """(r_n, Perron weights) from the current weights; refuses an uncertified core."""
        rate, weights = self.bounds().perron(n_low)
        if not rate < 1.0:
            raise RuntimeError(f"measured cycle rate {rate:.4f} >= 1; weights left their budgets")
        return rate, weights

    def solve(self, high: Tensor, low: Tensor, ctx: Context, n_low: int, tol: float, max_cycles: int,
              certificate: Optional[Tuple[float, Tuple[float, float]]] = None,
              maps: Optional["CoreMaps"] = None) -> Solution:
        """Iterate cycles until the certified bound ||z - z*||_w <= tol (E6); w_h = 1 bounds ||h - h*||."""
        rate, weights = certificate or self.certificate(n_low)
        factor = rate / (1.0 - rate)
        maps = maps or self.maps()
        bound = math.inf
        for cycles in range(1, max_cycles + 1):
            new_high, new_low = self.cycle(high, low, ctx, n_low, maps)
            step = weighted_distance(new_high - high, new_low - low, weights)
            bound = factor * float(step.max())
            high, low = new_high, new_low
            if bound <= tol:
                return Solution(high, low, cycles, bound, True)
        return Solution(high, low, max_cycles, bound, False)


@dataclass(frozen=True)
class CoreMaps:
    """The four certified matrices of one core, materialized."""

    low_self: Tensor
    low_from_high: Tensor
    high_self: Tensor
    high_from_low: Tensor


def mix(route: Tensor, x: Tensor) -> Tensor:
    """A X for one routing matrix; head-wise [A_1 X_1, ..., A_H X_H] for (..., H, n, n) routings (S2)."""
    if route.dim() == x.dim():
        return route @ x
    heads = route.shape[-3]
    parts = x.unflatten(-1, (heads, -1)).transpose(-3, -2)
    return (route @ parts).transpose(-3, -2).flatten(-2)
