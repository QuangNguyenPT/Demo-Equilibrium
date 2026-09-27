"""Self-improvement keep gate with a provable false-keep rate (spec §8, G1)."""

import math

from torch import Tensor


def keep_margin(n: int, delta: float) -> float:
    """Hoeffding margin sqrt(2 ln(1/δ) / n) for paired differences in [-1, 1]."""
    if n < 1 or not 0.0 < delta < 1.0:
        raise ValueError(f"need n >= 1 and 0 < delta < 1, got n={n}, delta={delta}")
    return math.sqrt(2.0 * math.log(1.0 / delta) / n)


def should_keep(diffs: Tensor, delta: float, experiments: int = 1) -> bool:
    """Keep iff mean(candidate - baseline) >= margin at δ / experiments (Bonferroni)."""
    if bool((diffs.abs() > 1.0).any()):
        raise ValueError("paired differences must lie in [-1, 1]")
    return float(diffs.double().mean()) >= keep_margin(diffs.numel(), delta / experiments)
