"""HRM-EQB: prelude -> segments of certified equilibrium -> structured head, halting head, coda.

One segment s (spec §0, §14):
    u_s   = x + carry(h*_{s-1})                         routing / memory input
    A_L,A_H = S1(Sinkhorn(attention(u_s)))              per head, fixed during the solve (S1, S2)
    c_L   = inject_low(x) + memory(u_s)                 knowledge enters as context (K2, E5)
    c_H   = inject_high(x) + chain(AttnRes(h*_1..h*_{s-1}))   the thinking chain enters as context (A1)
    (h*_s, l*_s) = certified solve from the warm start (h*_{s-1}, l*_{s-1})   (E2: start is irrelevant)
    logits = readout(h*_s at answer cells) · Embedᵀ     the answer is read in token space (L2)
Training backpropagates through the last grad_cycles cycles from the equilibrium (E4, E7).
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from hrm_eqb.algebra import as_complex, as_real
from hrm_eqb.attnres import DepthAttention
from hrm_eqb.certified import substochastic
from hrm_eqb.coda import Coda
from hrm_eqb.config import ModelConfig
from hrm_eqb.equilibrium import Context, EquilibriumCore
from hrm_eqb.field import VOCAB, Batch
from hrm_eqb.memory import KNOWLEDGE_TENSORS, KnowledgeMemory

AXES = 3  # row, col, slot


def position_frequencies(planes: int) -> Tensor:
    """(3, planes) rotation frequencies, all distinct: ω_{a,j} = 100^{-(3j + a) / (3 planes)}."""
    exponent = (3 * torch.arange(planes)[None, :] + torch.arange(AXES)[:, None]).float() / (3 * planes)
    return 100.0 ** (-exponent)


MASKED = -1e4  # finite "minus infinity": exp underflows to 0 but every gradient stays finite


def sinkhorn_route(logits: Tensor, valid: Tensor, iters: int) -> Tensor:
    """Nonnegative (B, H, n, n) routing: log-domain Sinkhorn over valid tokens, then S1 normalization.

    Normalizing in log space keeps every gradient a bounded softmax weight; dividing by
    underflowed row sums in probability space overflowed the backward pass.
    """
    mask = valid[:, None, :, None] & valid[:, None, None, :]
    log = logits.masked_fill(~mask, MASKED)
    for _ in range(iters):
        log = (log - torch.logsumexp(log, dim=-1, keepdim=True)).masked_fill(~mask, MASKED)
        log = (log - torch.logsumexp(log, dim=-2, keepdim=True)).masked_fill(~mask, MASKED)
    return substochastic(torch.exp(log) * mask)


class Prelude(nn.Module):
    """Token embedding bound to its (row, col, slot) by rotation (T1), then transformer layers."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, cfg.dim)
        nn.init.normal_(self.embed.weight, std=cfg.dim ** -0.5)
        self.register_buffer("freqs", position_frequencies(cfg.dim // 2), persistent=False)
        layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, 4 * cfg.dim, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.layers = nn.TransformerEncoder(layer, cfg.prelude_layers, enable_nested_tensor=False)

    def bind(self, content: Tensor, coords: Tensor) -> Tensor:
        """Embedding ⊛ X^row ⊛ Y^col ⊛ S^slot as one rotation per complex plane."""
        angle = coords @ self.freqs
        return as_real(as_complex(self.embed(content)) * torch.polar(torch.ones_like(angle), angle))

    def forward(self, batch: Batch) -> Tensor:
        x = self.bind(batch.content, batch.coords)
        return self.layers(x, src_key_padding_mask=~batch.valid)


@dataclass
class SegmentOutput:
    logits: Tensor        # (B, m, VOCAB) at answer cells
    halt: Tensor          # (B,) logit of "this answer is right"
    high: Tensor          # (B, n, dim) equilibrium, detached
    low: Tensor
    cycles: int
    error_bound: float    # certified ||h - h*|| at the halt (E6)


class HRMEQB(nn.Module):
    """The whole model; core weights go to the .hrmc, memory knowledge to knowledge.pt."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.prelude = Prelude(cfg)
        self.route_low = nn.Linear(d, 2 * d)
        self.route_high = nn.Linear(d, 2 * d)
        self.carry = nn.Linear(d, d)
        self.inject_low = nn.Linear(d, d)
        self.inject_high = nn.Linear(d, d)
        self.memory = KnowledgeMemory(d, cfg.memory_side, cfg.memory_topk, cfg.key_dim, cfg.value_quaternions)
        self.chain = DepthAttention(d, queries=1)
        self.chain_in = nn.Linear(d, d)
        self.core = EquilibriumCore(d, cfg.budgets(), cfg.shrink)
        self.readout = nn.Linear(d, d)
        self.halt = nn.Linear(d, 1)
        self.coda = Coda(d, cfg.coda_dim, cfg.coda_layers, cfg.thought_tokens, cfg.max_text, cfg.heads)

    @staticmethod
    def knowledge_names() -> List[str]:
        return [f"memory.{n}" for n in KNOWLEDGE_TENSORS]

    def core_state(self) -> dict:
        """Every weight except the knowledge tensors."""
        skip = set(self.knowledge_names())
        return {k: v for k, v in self.state_dict().items() if k not in skip}

    def routing(self, proj: nn.Linear, u: Tensor, valid: Tensor, gain: float) -> Tensor:
        heads, d = self.cfg.heads, self.cfg.dim
        q, k = proj(u).chunk(2, dim=-1)
        q = q.unflatten(-1, (heads, d // heads)).transpose(1, 2)
        k = k.unflatten(-1, (heads, d // heads)).transpose(1, 2)
        logits = gain * (q @ k.transpose(-2, -1)) / math.sqrt(d // heads)
        return sinkhorn_route(logits, valid, self.cfg.route_iters)

    def context(self, x: Tensor, high: Tensor, chain: List[Tensor], batch: Batch, route_gain: float) -> Context:
        """Segment context: routings, memory read and chain summary; pads get zero injection."""
        u = x + self.carry(high)
        pad = batch.valid.unsqueeze(-1).float()
        chain_ctx = self.chain(torch.stack(chain))[0] if chain else torch.zeros_like(x)
        return Context(self.routing(self.route_low, u, batch.valid, route_gain),
                       self.routing(self.route_high, u, batch.valid, route_gain),
                       (self.inject_low(x) + self.memory(u)) * pad,
                       (self.inject_high(x) + self.chain_in(chain_ctx)) * pad)

    def answer_logits(self, high: Tensor, batch: Batch) -> Tensor:
        idx = batch.answer_pos.unsqueeze(-1).expand(-1, -1, high.shape[-1])
        return self.readout(high.gather(1, idx)) @ self.prelude.embed.weight.T

    def halt_logit(self, high: Tensor, batch: Batch) -> Tensor:
        idx = batch.answer_pos.unsqueeze(-1).expand(-1, -1, high.shape[-1])
        mask = batch.answer_valid.unsqueeze(-1).float()
        pooled = (high.gather(1, idx) * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.halt(pooled).squeeze(-1)

    def forward(self, batch: Batch, segments: Optional[int] = None, route_gain: float = 1.0,
                halt_threshold: Optional[float] = None, tol: Optional[float] = None) -> List[SegmentOutput]:
        """Run up to `segments` segments; with halt_threshold, stop once every task's halt prob exceeds it.

        `tol` overrides the certified solve tolerance (training may solve more loosely: the gradient
        error then grows by at most tol / (1 - r) on top of E4's truncation bound).
        """
        cfg = self.cfg
        x = self.prelude(batch)
        high, low = torch.zeros_like(x), torch.zeros_like(x)
        chain: List[Tensor] = []
        outputs: List[SegmentOutput] = []
        certificate = self.core.certificate(cfg.n_low)
        with torch.no_grad():
            frozen_maps = self.core.maps()
        maps = self.core.maps() if torch.is_grad_enabled() else frozen_maps
        for _ in range(segments or cfg.max_segments):
            ctx = self.context(x, high, chain, batch, route_gain)
            with torch.no_grad():
                sol = self.core.solve(high, low, ctx, cfg.n_low, tol or cfg.solve_tol, cfg.max_cycles, certificate,
                                      frozen_maps)
            h, l = sol.high, sol.low
            if torch.is_grad_enabled():
                for _ in range(cfg.grad_cycles):
                    # recomputed in backward: keeps only each cycle's inputs alive (memory ~ tokens, not steps)
                    h, l = checkpoint(self.core.cycle, h, l, ctx, cfg.n_low, maps, use_reentrant=False)
            out = SegmentOutput(self.answer_logits(h, batch), self.halt_logit(h, batch), h.detach(), l.detach(),
                                sol.cycles, sol.error_bound)
            outputs.append(out)
            high, low = out.high, out.low
            chain.append(out.high)
            if halt_threshold is not None and bool((torch.sigmoid(out.halt) > halt_threshold).all()):
                break
        return outputs
