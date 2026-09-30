"""Product-key retrieval: exact top-k over S1 x S2 slots from two top-k searches (spec M1)."""

from typing import Tuple

from torch import Tensor


def product_topk(scores_a: Tensor, scores_b: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """Top-k of scores_a[i] + scores_b[j] over all pairs; returns (values, slot = i * S2 + j)."""
    size_b = scores_b.shape[-1]
    if not 1 <= k <= min(scores_a.shape[-1], size_b):
        raise ValueError(f"k must be in [1, min sub-key count], got {k}")
    top_a, idx_a = scores_a.topk(k, dim=-1)
    top_b, idx_b = scores_b.topk(k, dim=-1)
    grid = (top_a[..., :, None] + top_b[..., None, :]).flatten(-2)
    values, flat = grid.topk(k, dim=-1)
    rows = idx_a.gather(-1, flat // k)
    cols = idx_b.gather(-1, flat % k)
    return values, rows * size_b + cols
