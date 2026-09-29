"""Place-Value Reader (PVR): a learned layer numeral string -> exact rational (doc 03, section 3).

    v(s) = sum_k theta[d_k] * b**k      (k = position relative to the radix point; negative after it)

theta (10 digit values) and b (base) are learned from written sums "a + c = e" only (no values are
given), by additive consistency. Theorem 5: with single-digit no-carry sums, 0+0=0 and one carry,
the unique solution with theta_1 = 1 is theta_d = d, b = 10. After training the parameters are
snapped to integers and ACCEPTED only if every training sum then holds exactly in Fraction arithmetic.

Where this sits in the repo: this is the v0.5 PLACE-VALUE READER, the ONLY learned numeric
component of the system. Everything else in `hrm_x` is symbolic (`mind`, `library`, `workspace`,
`rules`, `operators`, ...); here a two-parameter model (theta, b) is fitted by self-supervision on
written sums and then frozen into an exact `Fraction` map, which is what lets the symbolic layers
reason over values without any float. The `DigitMLP` / `DigitTransformer` classes are the
value-supervised baselines -- they are given the true values, i.e. MORE information than the PVR --
and exist only to show the PVR's self-supervised route is not simply luck.

IMPORT COST: this module imports `torch` at module level and must stay optional. Nothing else in
`hrm_x` imports it, and it must NOT be added to `hrm_x/__init__.py`: the package core is pure
stdlib + numpy, and only this learned reader needs torch.

Tensor contract: digits (B, W) int64 in 0..9 (pad = 0 with weight 0), powers (B, W) float32
(position exponents k), mask (B, W) float32 in {0,1}; theta (10,) float32; log_b () float32.

Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

W = 16                                      # max digits per numeral in a batch


def encode(numerals: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = len(numerals)
    digits = torch.zeros(B, W, dtype=torch.long)
    powers = torch.zeros(B, W)
    mask = torch.zeros(B, W)
    for i, s in enumerate(numerals):
        ip, _, fp = s.partition(".")
        ks = list(range(len(ip) - 1, -1, -1)) + [-(j + 1) for j in range(len(fp))]
        ds = [int(c) for c in ip + fp]
        assert len(ds) <= W, f"numeral too long: {s}"
        for j, (d, k) in enumerate(zip(ds, ks)):
            digits[i, j], powers[i, j], mask[i, j] = d, float(k), 1.0
    return digits, powers, mask


class PlaceValueReader(nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(torch.randn(10, generator=g))
        self.log_b = nn.Parameter(torch.log(torch.empty(()).uniform_(2.0, 20.0, generator=g)))
        self.snapped: Optional[Tuple[List[int], int]] = None

    def forward(self, digits: torch.Tensor, powers: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert digits.shape == powers.shape == mask.shape and digits.ndim == 2
        b = torch.exp(self.log_b)
        return (self.theta[digits] * b ** powers * mask).sum(-1)          # (B,)

    def exact(self, s: str) -> Fraction:
        """Exact value with snapped parameters (raises if the layer has not been verified)."""
        assert self.snapped is not None, "reader not verified; exact reading unavailable"
        theta, b = self.snapped
        ip, _, fp = s.partition(".")
        v = Fraction(0)
        for j, c in enumerate(ip):
            v += theta[int(c)] * Fraction(b) ** (len(ip) - 1 - j)
        for j, c in enumerate(fp):
            v += theta[int(c)] * Fraction(b) ** (-(j + 1))
        return v


def make_sums(rng: random.Random, n: int, base: int = 10, max_int: int = 99, decimals: bool = True,
              carries: bool = True) -> List[Tuple[str, str, str]]:
    """Written sums in `base`; always includes 0+0=0, the no-carry single-digit sums, and (if carries) 5+5."""
    def show(v: Fraction) -> str:
        ip, fp = int(v), v - int(v)
        digs = []
        x = ip
        while True:
            digs.append(str(x % base)); x //= base
            if x == 0:
                break
        s = "".join(reversed(digs))
        if fp:
            s += "." + str(int(fp * base))
        return s
    out = [("0", "0", "0")] + [(str(a), str(c), str(a + c)) for a in range(base) for c in range(base) if a + c < base]
    if carries:
        out.append((show(Fraction(base // 2)), show(Fraction(base // 2 + base % 2)), show(Fraction(base))))
    while len(out) < n:
        a, c = rng.randint(0, max_int), rng.randint(0, max_int)
        if not carries and any(int(x) + int(y) >= base for x, y in zip(str(a).zfill(3), str(c).zfill(3))):
            continue
        fa = Fraction(a) + (Fraction(rng.randint(0, base - 1), base) if decimals and rng.random() < 0.3 else 0)
        fc = Fraction(c) + (Fraction(rng.randint(0, base - 1), base) if decimals and rng.random() < 0.3 else 0)
        if not carries and (fa - int(fa)) + (fc - int(fc)) >= 1:
            continue
        out.append((show(fa), show(fc), show(fa + fc)))
    return out


@dataclass
class TrainReport:
    loss: float
    theta: List[float]
    base: float
    snapped: Optional[Tuple[List[int], int]]
    verified: bool
    steps: int
    # Every integer base in 2..64 that reproduces the training sums with the same snapped theta.
    # Theorem 5's identifiability check: acceptance requires this to be exactly [base].
    consistent_bases: List[int] = field(default_factory=list)


def _design(enc: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], b: float) -> torch.Tensor:
    """Feature matrix F (B, 10) with v = F @ theta: F[i, d] = sum_k [digit_k = d] b^k."""
    digits, powers, mask = enc
    w = (torch.tensor(b, dtype=torch.float64) ** powers.double()) * mask.double()
    return torch.zeros(digits.shape[0], 10, dtype=torch.float64).scatter_add_(1, digits, w)


def profile_base(A, C, E, grid: Sequence[float]) -> Tuple[float, torch.Tensor, List[float]]:
    """Given b the consistency residual M(b) theta is linear in theta: solve theta by constrained least
    squares (theta_1 = 1) for each b on a grid, return the b with the lowest normalised residual."""
    best, curve = (float("inf"), None, None), []
    for b in grid:
        M = _design(A, b) + _design(C, b) - _design(E, b)
        scale = (_design(E, b).abs().sum(1, keepdim=True) + 1.0)
        M = M / scale                                           # each sum counts equally
        rest = [d for d in range(10) if d != 1]
        theta_rest = torch.linalg.lstsq(M[:, rest], -M[:, 1:2]).solution.squeeze(-1)
        theta = torch.zeros(10, dtype=torch.float64); theta[1] = 1.0; theta[rest] = theta_rest
        r = float(((M @ theta) ** 2).mean())
        curve.append(r)
        if r < best[0]:
            best = (r, torch.tensor(b), theta)
    return float(best[1]), best[2], curve


def train_reader(sums: Sequence[Tuple[str, str, str]], seed: int = 0, steps: int = 3000, lr: float = 0.05,
                 method: str = "profile") -> Tuple[PlaceValueReader, TrainReport]:
    """method='gd': Adam from random init (reported as a negative result: it stalls);
    method='profile': exact theta-solve per b on a log grid, then Adam refinement from the best b."""
    torch.manual_seed(seed)
    model = PlaceValueReader(seed)
    a, c, e = zip(*sums)
    A, C, E = encode(a), encode(c), encode(e)
    if method == "profile":
        grid = [float(v) for v in torch.logspace(torch.log10(torch.tensor(1.5)), torch.log10(torch.tensor(40.0)), 400)]
        b0, th0, _ = profile_base(A, C, E, grid)
        with torch.no_grad():
            model.log_b.fill_(float(torch.log(torch.tensor(b0))))
            model.theta.copy_(th0.float())
        lr = 0.005
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        va, vc, ve = model(*A), model(*C), model(*E)
        # Self-supervised additive consistency (scale-normalised) + unit anchor theta_1 = 1.
        loss = (((va + vc - ve) ** 2) / (va ** 2 + vc ** 2 + ve ** 2 + 1e-6)).mean() + (model.theta[1] - 1.0) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    theta = [float(t) for t in model.theta.detach()]
    base = float(torch.exp(model.log_b).detach())
    snap = ([int(round(t)) for t in theta], int(round(base)))

    def passes(b: int) -> bool:
        model.snapped = (snap[0], b)
        return all(model.exact(x) + model.exact(y) == model.exact(z) for x, y, z in sums)

    # Accept only if the snapped base reproduces every training sum exactly AND it is the only
    # integer base in 2..64 that does (Theorem 5 identifiability; without a carry this fails).
    consistent_bases = [b for b in range(2, 65) if passes(b)]
    ok = snap[1] >= 2 and consistent_bases == [snap[1]]
    model.snapped = snap if ok else None
    rep = TrainReport(float(loss.detach()), theta, base, snap if ok else None, ok, steps, consistent_bases)
    return model, rep


# ---------------- baselines (value-supervised, i.e. given MORE information than the PVR) ----------------

class DigitMLP(nn.Module):
    """Right-aligned fixed-width digits (one-hot) -> value. Positions unseen in training carry zeros."""

    def __init__(self, width: int = W, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width * 11, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def digit_features(numerals: Sequence[str], width: int = W) -> torch.Tensor:
    """Aligned at the radix point: slot j holds the digit of power (width//2 - 1 - j); 11th channel = present."""
    out = torch.zeros(len(numerals), width, 11)
    half = width // 2
    for i, s in enumerate(numerals):
        ip, _, fp = s.partition(".")
        for j, ch in enumerate(reversed(ip)):
            if half - 1 - j >= 0:
                out[i, half - 1 - j, int(ch)] = 1; out[i, half - 1 - j, 10] = 1
        for j, ch in enumerate(fp):
            if half + j < width:
                out[i, half + j, int(ch)] = 1; out[i, half + j, 10] = 1
    return out.reshape(len(numerals), -1)


class DigitTransformer(nn.Module):
    """Digits + learned position embeddings (radix-aligned) -> value by a small encoder."""

    def __init__(self, dim: int = 32, width: int = W) -> None:
        super().__init__()
        self.width = width
        self.tok = nn.Embedding(11, dim)
        self.pos = nn.Embedding(width, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, 2 * dim, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.reshape(B, self.width, 11)
        ids = torch.where(x[..., 10] > 0, x[..., :10].argmax(-1), torch.full((B, self.width), 10))
        h = self.tok(ids) + self.pos(torch.arange(self.width))[None]
        return self.head(self.enc(h).mean(1)).squeeze(-1)


def train_baseline(model: nn.Module, numerals: Sequence[str], values: Sequence[float], seed: int, steps: int = 3000) -> nn.Module:
    torch.manual_seed(seed)
    x = digit_features(numerals)
    y = torch.tensor(values, dtype=torch.float32)
    mu, sd = y.mean(), y.std() + 1e-6
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        loss = ((model(x) - (y - mu) / sd) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.mu, model.sd = mu, sd                    # type: ignore[attr-defined]
    return model


@torch.no_grad()
def baseline_predict(model: nn.Module, numerals: Sequence[str]) -> List[float]:
    return [float(v) for v in model(digit_features(numerals)) * model.sd + model.mu]   # type: ignore[attr-defined]
