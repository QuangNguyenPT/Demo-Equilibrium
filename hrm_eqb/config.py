"""Architecture of one HRM-EQB model: a frozen dataclass with a strict JSON round-trip."""

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

from hrm_eqb.equilibrium import ContractionBounds


@dataclass(frozen=True)
class ModelConfig:
    """Every architectural number of the model; unknown or missing keys are refused on load."""

    dim: int = 128                  # state width, 4 x quaternions
    heads: int = 4                  # routing heads; each mixes dim / heads features
    prelude_layers: int = 3
    max_side: int = 10              # largest training canvas side
    n_low: int = 2                  # fast steps per slow step (E3: speed only, never the answer)
    max_segments: int = 3           # every served segment is trained (deep supervision)
    solve_tol: float = 1e-3         # certified ||h - h*|| at halt (E6)
    max_cycles: int = 40
    grad_cycles: int = 3            # backprop through the last k cycles (E4)
    route_iters: int = 4            # Sinkhorn sweeps before the S1 normalization
    shrink: float = 0.0             # radial-ReLU threshold μ
    s_l: float = 0.5
    s_h: float = 0.25
    v_h: float = 0.4
    v_l: float = 0.4
    alpha: float = 0.9
    beta: float = 0.9
    memory_side: int = 64           # N = memory_side^2 knowledge slots
    memory_topk: int = 8
    key_dim: int = 64
    value_quaternions: int = 16
    coda_dim: int = 128
    coda_layers: int = 2
    thought_tokens: int = 4
    max_text: int = 64

    def __post_init__(self):
        if self.dim % 4 or self.dim % self.heads:
            raise ValueError("dim must be a multiple of 4 (quaternions) and of heads")
        if self.key_dim % 2 or self.memory_topk > self.memory_side:
            raise ValueError("key_dim must be even and memory_topk <= memory_side")
        if min(self.n_low, self.max_segments, self.grad_cycles, self.max_cycles) < 1:
            raise ValueError("n_low, max_segments, grad_cycles, max_cycles must be >= 1")
        self.budgets()  # validates Γ < 1

    def budgets(self) -> ContractionBounds:
        """Configured norm budgets and step sizes of the equilibrium core."""
        bounds = ContractionBounds(self.s_l, self.s_h, self.v_h, self.v_l, self.alpha, self.beta)
        if not bounds.gamma() < 1.0:
            raise ValueError(f"budgets give Γ = {bounds.gamma():.4f} >= 1")
        return bounds

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        unknown, missing = set(data) - names, names - set(data)
        if unknown or missing:
            raise ValueError(f"config keys mismatch: unknown={sorted(unknown)} missing={sorted(missing)}")
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


PRESETS = {
    "tiny": ModelConfig(dim=24, heads=2, prelude_layers=1, max_side=6, n_low=2, max_segments=2,
                        max_cycles=30, memory_side=8, memory_topk=4, key_dim=8, value_quaternions=4,
                        coda_dim=32, coda_layers=1, thought_tokens=2, max_text=48),
    "small": ModelConfig(),
}
