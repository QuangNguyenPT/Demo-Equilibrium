---
title: External Research Index (Kimi K3)
type: moc
tags:
  - knowledge-graph
  - external-research
  - reference
status: active
created: 2026-09-26
updated: 2026-09-29
agent: "opencode/stealth/space-bunny-alpha"
model: "space-bunny-alpha"
---

# External Research: Kimi K3 in C

These six notes are a study of **someone else's model**, kept because the
systems lessons transfer. They are not a description of this project, and until
2026-09-29 they sat in `Entities & Concepts/` beside the project's own concepts,
which made the vault look as though HRM ran 896 routed experts on a 2.78-trillion
parameter checkpoint. It does not, and it will not.

Sources: Moonshot AI Kimi K3 and Fareed Khan's `kimi-k3-in-c` write-up. The
vendored copy of that repository is at `docs/kimi-k3-in-c-main/` and is excluded
from the knowledge graph (`.graphifyignore`) and from git.

## What transfers to this project

| note | transferable idea | where it would matter here |
|---|---|---|
| [[Concept - Microscaling MXFP4 & E8M0 Quantization]] | sub-byte weight encodings | only if a neural component is ever trained; the current model trains nothing |
| [[Concept - Block Sparse Mixture of Experts (MoE)]] | most parameters can stay asleep | the ADR-006 finding was that a trained neural baseline could not do arithmetic at all; sparsity does not address that |
| [[Concept - Kimi Delta Attention (KDA)]] | constant-memory recurrence | contrasts with this project's exact `Fraction` arithmetic, which is O(bits) and deliberately not a neural recurrence |
| [[Concept - Multi-Head Latent Attention (MLA)]] | low-rank KV compression | irrelevant while there is no KV cache |
| [[Concept - Direct IO & NVMe Streaming]] | `O_DIRECT` bypasses the page cache | relevant only to a model too large for RAM, which this one is not |
| [[Concept - The Four Memory Reductions]] | the 675x cascade, 5,560 GB to 8.24 GB | a *serving* result about someone else's checkpoint |

The one item that is about *our* own substrate is not on this page: the
zero-allocation arena and the C ABI are described in
[[Engine Architecture & Memory Model]] and in ADR-001 and ADR-002, which is where
they belong.

**Index**: [[00 - Knowledge Graph Index]] — the project's own conceptual map.
