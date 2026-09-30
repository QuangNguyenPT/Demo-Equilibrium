"""Curiosity: the most informative question is where the committee disagrees most (spec §7, Q1)."""

import math
from typing import List, Sequence, Tuple

import numpy as np
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


def tally(answers: Sequence[np.ndarray]) -> List[Tuple[np.ndarray, int]]:
    """Distinct answers (exact, including shape) with their vote counts, most voted first."""
    groups: dict = {}
    for a in answers:
        key = (a.shape, a.tobytes())
        if key in groups:
            groups[key][1] += 1
        else:
            groups[key] = [a, 1]
    return sorted(((a, c) for a, c in groups.values()), key=lambda ac: -ac[1])


def entropy_bits(counts: Sequence[int]) -> float:
    """Entropy in bits of a vote distribution given by counts."""
    total = float(sum(counts))
    return -sum((c / total) * math.log2(c / total) for c in counts if c)
