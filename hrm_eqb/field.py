"""The task field: every task becomes one set of tokens with (content, row, col, slot).

Slots: demo i input = 2i, demo i output = 2i + 1, query = 2D, answer = 2D + 1. All grids
of a task share one canvas (the largest demo/query shape); cells outside a grid are VOID,
answer cells are MASK. The answer is read back as the top-left block of non-VOID cells.
Augmentations (dihedral x color permutation, grids only) act on whole tasks with an exact
inverse on answers (spec §14, lemma G1a).
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from hrm_eqb.tasks import Grid, Task

VOID, MASK = 0, 1
GRID_BASE = 2                               # colors 0..9 -> 2..11
INT_MIN, INT_MAX = -64, 64
INT_BASE = GRID_BASE + 10                   # ints -64..64 -> 12..140
BYTE_BASE = INT_BASE + INT_MAX - INT_MIN + 1  # bytes 0..255 -> 141..396
VOCAB = BYTE_BASE + 256
INVALID = -10_000                           # decoded cell that is not a value of the kind


def to_ids(kind: str, values: Grid) -> Grid:
    """Domain values -> token ids."""
    v = np.asarray(values, dtype=np.int64)
    if kind == "grid":
        lo, hi, base = 0, 9, GRID_BASE
    elif kind == "int":
        lo, hi, base = INT_MIN, INT_MAX, INT_BASE - INT_MIN
    elif kind == "byte":
        lo, hi, base = 0, 255, BYTE_BASE
    else:
        raise ValueError(f"unknown kind '{kind}'")
    if v.size and (v.min() < lo or v.max() > hi):
        raise ValueError(f"{kind} value out of range [{lo}, {hi}]")
    return v + base


def from_ids(kind: str, ids: Grid) -> Grid:
    """Token ids of an answer canvas -> domain values (top-left non-VOID block)."""
    ids = np.asarray(ids, dtype=np.int64)
    filled = ids != VOID
    if not filled.any():
        return np.zeros((0, 0), dtype=np.int64)
    h = int(np.nonzero(filled.any(axis=1))[0].max()) + 1
    w = int(np.nonzero(filled.any(axis=0))[0].max()) + 1
    block = ids[:h, :w]
    lo, hi, shift = {"grid": (GRID_BASE, GRID_BASE + 9, GRID_BASE),
                     "int": (INT_BASE, INT_BASE + INT_MAX - INT_MIN, INT_BASE - INT_MIN),
                     "byte": (BYTE_BASE, BYTE_BASE + 255, BYTE_BASE)}[kind]
    return np.where((block >= lo) & (block <= hi), block - shift, INVALID)


def answers_equal(a: Optional[Grid], b: Optional[Grid]) -> bool:
    """Exact match including shape; two empty answers are equal."""
    if a is None or b is None:
        return False
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 and b.size == 0:
        return True
    return a.shape == b.shape and bool((a == b).all())


# ---------- augmentation group (grids): D4 x color permutations fixing 0 ----------

@dataclass(frozen=True)
class Augmentation:
    """g(x) = rot90^r(flip^f(π(x))), clockwise quarter turns; π fixes color 0."""

    rotations: int = 0
    flip: bool = False
    colors: Tuple[int, ...] = tuple(range(10))

    def forward(self, g: Grid) -> Grid:
        out = np.asarray(self.colors, dtype=np.int64)[g]
        if self.flip:
            out = out[:, ::-1]
        return np.rot90(out, -self.rotations).copy()

    def inverse(self, g: Grid) -> Grid:
        out = np.rot90(np.asarray(g), self.rotations)
        if self.flip:
            out = out[:, ::-1]
        return np.argsort(np.asarray(self.colors, dtype=np.int64))[out].copy()

    def apply(self, task: Task) -> Task:
        """The augmented task g·T (demos, query and answer all transformed)."""
        if task.kind != "grid":
            raise ValueError("augmentations act on grid tasks only")
        answer = None if task.answer is None else self.forward(task.answer)
        return Task(task.domain, task.family, [(self.forward(x), self.forward(y)) for x, y in task.demos],
                    self.forward(task.query), answer, task.description, dict(task.meta))


IDENTITY = Augmentation()


def random_augmentation(rng: np.random.Generator) -> Augmentation:
    perm = np.concatenate(([0], rng.permutation(9) + 1))
    return Augmentation(int(rng.integers(0, 4)), bool(rng.integers(0, 2)), tuple(int(c) for c in perm))


def views(task: Task, count: int, rng: np.random.Generator) -> List[Augmentation]:
    """Identity plus distinct random augmentations (identity only for non-grid tasks)."""
    if task.kind != "grid":
        return [IDENTITY]
    out, seen = [IDENTITY], {IDENTITY}
    while len(out) < count:
        aug = random_augmentation(rng)
        if aug not in seen:
            seen.add(aug)
            out.append(aug)
    return out


# ---------- encoding ----------

@dataclass
class EncodedTask:
    content: np.ndarray        # (n,) token ids
    coords: np.ndarray         # (n, 3) row, col, slot
    answer_pos: np.ndarray     # (m,) token indices of the answer canvas
    target: Optional[np.ndarray]  # (m,) ids, VOID outside the true answer
    canvas: Tuple[int, int]
    kind: str


MAX_CANVAS = 30


def canvas_of(task: Task) -> Tuple[int, int]:
    """Largest demo/query shape, grown to the query shape times the demos' largest out/in ratio.

    Covers rules whose output size is a fixed multiple of the input size (copying, doubling,
    upscaling) or its transpose (rotations); capped at MAX_CANVAS.
    """
    grids = [g for pair in task.demos for g in pair] + [task.query]
    h, w = max(g.shape[0] for g in grids), max(g.shape[1] for g in grids)
    if task.demos:
        rh = max(y.shape[0] / max(x.shape[0], 1) for x, y in task.demos)
        rw = max(y.shape[1] / max(x.shape[1], 1) for x, y in task.demos)
        h = max(h, min(MAX_CANVAS, int(np.ceil(task.query.shape[0] * rh))))
        w = max(w, min(MAX_CANVAS, int(np.ceil(task.query.shape[1] * rw))))
        if any(y.shape == x.shape[::-1] != x.shape for x, y in task.demos):
            h, w = max(h, task.query.shape[1]), max(w, task.query.shape[0])
    return h, w


def _place(ids: Grid, canvas: Tuple[int, int]) -> Grid:
    out = np.full(canvas, VOID, dtype=np.int64)
    out[:ids.shape[0], :ids.shape[1]] = ids
    return out


def encode_task(task: Task) -> EncodedTask:
    """Token field of a task; the answer target is None if the answer is unknown or does not fit."""
    kind, canvas = task.kind, canvas_of(task)
    rows, cols = np.meshgrid(np.arange(canvas[0]), np.arange(canvas[1]), indexing="ij")
    grids = [g for pair in task.demos for g in pair] + [task.query]
    contents = [_place(to_ids(kind, g), canvas).ravel() for g in grids]
    contents.append(np.full(canvas[0] * canvas[1], MASK, dtype=np.int64))
    coords = [np.stack((rows.ravel(), cols.ravel(), np.full(rows.size, slot)), axis=1)
              for slot in range(len(contents))]
    content, coord = np.concatenate(contents), np.concatenate(coords).astype(np.float32)
    answer_pos = np.arange(content.size - canvas[0] * canvas[1], content.size)
    target = None
    if task.answer is not None and task.answer.shape[0] <= canvas[0] and task.answer.shape[1] <= canvas[1]:
        target = _place(to_ids(kind, task.answer), canvas).ravel()
    return EncodedTask(content, coord, answer_pos, target, canvas, kind)


def decode_answer(enc: EncodedTask, predicted_ids: np.ndarray) -> Grid:
    """Predicted answer-canvas ids (m,) -> domain values."""
    return from_ids(enc.kind, np.asarray(predicted_ids).reshape(enc.canvas))


@dataclass
class Batch:
    content: Tensor        # (B, n) long
    coords: Tensor         # (B, n, 3) float
    valid: Tensor          # (B, n) bool
    answer_pos: Tensor     # (B, m) long
    answer_valid: Tensor   # (B, m) bool
    target: Optional[Tensor]  # (B, m) long, VOID where padded
    encoded: List[EncodedTask]

    def to(self, device: torch.device) -> "Batch":
        move = (lambda t: None if t is None else t.to(device, non_blocking=True))
        return Batch(move(self.content), move(self.coords), move(self.valid), move(self.answer_pos),
                     move(self.answer_valid), move(self.target), self.encoded)


def budget_chunks(encoded: Sequence[EncodedTask], pairs: int) -> List[List[int]]:
    """Consecutive index groups whose padded attention cost (count x max_n^2) stays within `pairs`.

    Routing tensors are (B, heads, n, n); bounding B * n_max^2 bounds their memory. A single
    task larger than the budget still forms its own group.
    """
    groups: List[List[int]] = []
    current: List[int] = []
    n_max = 0
    for i, e in enumerate(encoded):
        grown = max(n_max, e.content.size)
        if current and (len(current) + 1) * grown * grown > pairs:
            groups.append(current)
            current, grown = [], e.content.size
        current.append(i)
        n_max = grown
    if current:
        groups.append(current)
    return groups


def collate(encoded: Sequence[EncodedTask]) -> Batch:
    """Pad a list of encoded tasks into one batch (target present only if every task has one)."""
    b = len(encoded)
    n = max(e.content.size for e in encoded)
    m = max(e.answer_pos.size for e in encoded)
    content = torch.full((b, n), VOID, dtype=torch.long)
    coords = torch.zeros((b, n, 3), dtype=torch.float32)
    valid = torch.zeros((b, n), dtype=torch.bool)
    answer_pos = torch.zeros((b, m), dtype=torch.long)
    answer_valid = torch.zeros((b, m), dtype=torch.bool)
    with_target = all(e.target is not None for e in encoded)
    target = torch.full((b, m), VOID, dtype=torch.long) if with_target else None
    for i, e in enumerate(encoded):
        k, j = e.content.size, e.answer_pos.size
        content[i, :k] = torch.from_numpy(e.content)
        coords[i, :k] = torch.from_numpy(e.coords)
        valid[i, :k] = True
        answer_pos[i, :j] = torch.from_numpy(e.answer_pos)
        answer_valid[i, :j] = True
        if target is not None:
            target[i, :j] = torch.from_numpy(e.target)
    return Batch(content, coords, valid, answer_pos, answer_valid, target, list(encoded))
