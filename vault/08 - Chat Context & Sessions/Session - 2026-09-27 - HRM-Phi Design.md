---
title: "Session: 2026-09-27 — HRM-Φ Design"
type: session-log
tags:
  - session
  - design
  - research
  - hrm
date: 2026-09-27
status: design-proposed
---

# Session: 2026-09-27 — HRM-Φ Design

## Request

The current build is patches, not HRM. The user asked for:
- a deep reflection and an elegant, math-first design in PEM's style, with a plan before
  any code;
- language as a sub-part of HRM, through a semi-embedder from the thinking chain into
  transformer and neural-network layers, on PyTorch;
- two model files: a custom core format, and a `.pt` knowledge file used like SSD
  unified memory ("without it, an empty brain");
- vault and graphify kept up to date.

## Research (verified sources)

- **ARC Prize HRM analysis:**
  - the hierarchy ≈ a same-size transformer;
  - the outer refinement loop drives the gains;
  - HRM is transductive (`puzzle_id`);
  - augmentation matters (about 300 suffice).
- **TRM:** a single 2-layer net, 7M parameters, with latent z and answer y; last-cycle
  backprop. Sudoku 87.4%, Maze 85.3%, ARC-1 44.6%, ARC-2 7.8%.
- **Mechanistic study (2026):** the solution lives in z_H; z_L is transient.
- **Recurrent-depth LM (Geiping 2025):** prelude → recurrent core → coda; truncated
  backprop over the last k iterations.
- **Memory Layers at Scale (ICML 2025):** product-key memory, millions of slots, sparse
  reads.

## Key insights

1. RoPE **is** fractional phasor binding; multi-axis RoPE = PEM's X^x ⊛ Y^y encoding.
2. Plane-wise QK-norm **is** torus projection. On the torus, dot-product attention =
   L2 attention, with a Jacobian bounded by β and m.
3. PEM's resonator step **is** attention over a codebook, as are self-attention, memory
   recall and the semi-embedder: one operator reads everything.
4. TRM's input `x + y + z` **is** VSA bundling; role binding lets one network play both
   the L and H roles.
5. PEM's convergence proof ignored the torus projection, which is only 1/r-Lipschitz, so
   convergence must be measured and certified, not assumed.

## Measured feasibility

The `.pt` knowledge file can be written in place (torch 2.11, Windows): entries are
stored uncompressed and 64-byte aligned, `mmap` loading works, and a shared-map row
write persists. The zip CRC goes stale, so the design needs `seal()`.

## Produced

- [[HRM-Phi Design Document v0.1]]: the full design, sections 0–14.
- [[ADR-005 Neural HRM-Phi Replaces Program Search]]: the decision and the N0–N9 plan
  with machine-decidable gates.

## Open decisions (user)

1. Package name.
2. Inductive vs transductive ARC protocol.
3. First real benchmark.
4. Coda tokenizer.

No code was written this session, by request.
