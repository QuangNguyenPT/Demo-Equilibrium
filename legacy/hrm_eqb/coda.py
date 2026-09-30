"""Language coda: a semi-embedder from the thinking chain into a byte-level transformer decoder.

Thought tokens are Attention Residuals over the per-segment equilibria (A1): each is a
convex combination of chain summaries, so it cannot grow with the chain. The decoder also
cross-attends to the final equilibrium field. The coda reads the chain detached, so its loss
has zero gradient on every reasoning weight (spec §14, L3): language describes the solution,
it never changes it.
"""

from typing import List

import torch
from torch import Tensor, nn

from hrm_eqb.attnres import DepthAttention

BOS, EOS, PAD = 256, 257, 258
TEXT_VOCAB = 259


def encode_text(text: str, max_len: int) -> List[int]:
    """BOS + utf-8 bytes (truncated) + EOS."""
    return [BOS] + list(text.encode("utf-8"))[: max_len - 2] + [EOS]


def decode_text(tokens: List[int]) -> str:
    body = []
    for t in tokens:
        if t == EOS:
            break
        if t < 256:
            body.append(t)
    return bytes(body).decode("utf-8", errors="replace")


class Coda(nn.Module):
    """Describes, in words, the rule the reasoning core converged on."""

    def __init__(self, dim: int, coda_dim: int, layers: int, thought_tokens: int, max_text: int, heads: int):
        super().__init__()
        self.max_text = max_text
        self.summarize = nn.Linear(2 * dim, coda_dim)
        self.thoughts = DepthAttention(coda_dim, queries=thought_tokens)
        self.field = nn.Linear(dim, coda_dim)
        self.embed = nn.Embedding(TEXT_VOCAB, coda_dim)
        self.pos = nn.Embedding(max_text, coda_dim)
        layer = nn.TransformerDecoderLayer(coda_dim, heads, 4 * coda_dim, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.out = nn.Linear(coda_dim, TEXT_VOCAB)

    def memory(self, chain: List[Tensor], valid: Tensor, answer_mask: Tensor) -> Tensor:
        """(B, thoughts + n, coda_dim) cross-attention memory from detached segment equilibria."""
        states = [h.detach() for h in chain]
        v = valid.unsqueeze(-1).float()
        a = answer_mask.unsqueeze(-1).float()
        summaries = torch.stack([
            self.summarize(torch.cat(((h * v).sum(1) / v.sum(1).clamp_min(1.0),
                                      (h * a).sum(1) / a.sum(1).clamp_min(1.0)), dim=-1))
            for h in states])                                   # (S, B, coda_dim)
        thoughts = self.thoughts(summaries).transpose(0, 1)     # (B, Q, coda_dim)
        return torch.cat((thoughts, self.field(states[-1])), dim=1)

    def memory_mask(self, valid: Tensor) -> Tensor:
        """True where the decoder must not attend (padded field tokens)."""
        thoughts = torch.zeros(valid.shape[0], self.thoughts.query.shape[0], dtype=torch.bool, device=valid.device)
        return torch.cat((thoughts, ~valid), dim=1)

    def forward(self, memory: Tensor, memory_mask: Tensor, tokens: Tensor) -> Tensor:
        """Teacher-forced next-byte logits (B, T, TEXT_VOCAB)."""
        t = tokens.shape[1]
        x = self.embed(tokens) + self.pos(torch.arange(t, device=tokens.device))
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=tokens.device), diagonal=1)
        y = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=tokens == PAD,
                         memory_key_padding_mask=memory_mask)
        return self.out(y)

    @torch.no_grad()
    def describe(self, memory: Tensor, memory_mask: Tensor) -> List[str]:
        """Greedy decoding, one sentence per batch element."""
        tokens = torch.full((memory.shape[0], 1), BOS, dtype=torch.long, device=memory.device)
        for _ in range(self.max_text - 1):
            nxt = self(memory, memory_mask, tokens)[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, nxt), dim=1)
            if bool((tokens == EOS).any(dim=1).all()):
                break
        return [decode_text(row[1:].tolist()) for row in tokens]
