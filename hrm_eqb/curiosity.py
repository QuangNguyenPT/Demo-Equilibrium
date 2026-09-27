"""Curiosity: the most informative question is where the committee disagrees most (spec §7, Q1)."""

from typing import Tuple

import torch
from torch import Tensor


def vote_entropy(outputs: Tensor) -> float:
    """Entropy in bits of K members' exact votes at one input; outputs is (K, ...)."""
    _, counts = torch.unique(outputs.reshape(outputs.shape[0], -1), dim=0, return_counts=True)
    p = counts.double() / outputs.shape[0]
    return float(-(p * torch.log2(p)).sum())


def most_informative(probe_outputs: Tensor) -> Tuple[int, float]:
    """(probe index, bits) maximizing expected information gain over (P, K, ...) committee outputs."""
    gains = [vote_entropy(outputs) for outputs in probe_outputs]
    best = max(range(len(gains)), key=gains.__getitem__)
    return best, gains[best]
