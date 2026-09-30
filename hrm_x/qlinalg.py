"""Exact linear algebra over Q (fractions.Fraction). Reference implementation; sizes here are tiny.

Part of the operator inference over structured situations core: a query is determined by linear
evidence iff its feature row lies in the row space of the evidence matrix (doc 03, Theorem 3).
Pure stdlib and exact on purpose: determinacy is a theorem, not an approximation, so no floating
point or third-party array library may creep into this file.

Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

Row = List[Fraction]


def rref(rows: Sequence[Sequence[Fraction]]) -> Tuple[List[Row], List[int]]:
    """Reduced row echelon form and pivot columns. Exact; O(n m min(n, m)) Fraction ops."""
    m = [list(map(Fraction, r)) for r in rows]
    if not m:
        return [], []
    ncol = len(m[0])
    pivots: List[int] = []
    r = 0
    for c in range(ncol):
        p = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        inv = 1 / m[r][c]
        m[r] = [v * inv for v in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
        if r == len(m):
            break
    return m[:r], pivots


def rank(rows: Sequence[Sequence[Fraction]]) -> int:
    return len(rref(rows)[1])


def solve(A: Sequence[Sequence[Fraction]], b: Sequence[Fraction]) -> Optional[List[Fraction]]:
    """One exact solution of A p = b (free variables set to 0), or None if inconsistent."""
    if not A:
        return []
    k = len(A[0])
    aug = [list(map(Fraction, row)) + [Fraction(v)] for row, v in zip(A, b)]
    R, piv = rref(aug)
    if k in piv:                                   # pivot in the augmented column: 0 = nonzero
        return None
    p = [Fraction(0)] * k
    for row, c in zip(R, piv):
        p[c] = row[k]
    return p


def in_row_space(A: Sequence[Sequence[Fraction]], v: Sequence[Fraction]) -> bool:
    """v in rowspace(A)  <=>  rank([A; v]) == rank(A)   (Theorem 3 determinacy test)."""
    if not A:
        return all(x == 0 for x in v)
    return rank(list(A) + [list(v)]) == rank(A)


class RowSpace:
    """rowspace(A) reduced once; membership is then one elimination pass instead of an RREF.

    Equivalent to in_row_space(A, v) (tests/test_x_construct.py checks it on random matrices):
    R is reduced with unit pivots and zeros in every other row's pivot column, so removing each
    pivot component of v in turn leaves the component of v outside rowspace(A), which is zero iff
    v is in it.
    """

    def __init__(self, A: Sequence[Sequence[Fraction]]) -> None:
        self.R, self.pivots = rref(A) if A else ([], [])

    def contains(self, v: Sequence[Fraction]) -> bool:
        w = list(map(Fraction, v))
        for row, c in zip(self.R, self.pivots):
            f = w[c]
            if f != 0:
                w = [a - f * b for a, b in zip(w, row)]
        return all(x == 0 for x in w)
