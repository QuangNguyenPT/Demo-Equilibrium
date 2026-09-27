---
title: "ADR-005: Neural HRM-Φ Replaces Program Search as the Model"
type: adr
tags:
  - adr
  - architecture
  - hrm
  - plan
  - pytorch
status: proposed
date: 2026-09-27
supersedes: "[[ADR-003 Unified Cognitive Model]] (as the definition of *the model*; its loops, gate and harness are kept)"
related:
  - "[[HRM-EQB Specification v0.2 - Theorems and Proofs]]"
  - "[[HRM-Phi Design Document v0.1]]"
  - "[[ADR-004 Discriminating Benchmark and Reasoning Ceiling]]"
---

> **Update 2026-09-27: the model is HRM-EQB.** The user answered the open decisions and
> asked for provable mathematics. The design is now
> [[HRM-EQB Specification v0.2 - Theorems and Proofs]]; v0.1 is superseded. See
> "Resolved decisions" and "Plan (revised)" at the end. The N0 scaffold (certified
> operators, equilibrium core, certificate) is built as `hrm_eqb/`, with 23 theorem tests
> passing.

# ADR-005: Neural HRM-Φ Replaces Program Search as the Model

## Context

The user (2026-09-27): the current build "is not a POC model; it is a bunch of patches…
not HRM". They asked for:
- a mathematically principled design in the style of PEM v1.1;
- language as a subordinate part of HRM (a semi-embedder from the thinking chain into
  transformer layers that produce output, not a small LLM inside the machine);
- PyTorch as the base;
- two model files: a custom-format core, and a `.pt` knowledge file handled like
  SSD-unified memory, without which the model is an empty brain.

The evidence agrees with the critique:
- ADR-004's ablation is flat. Program search subsumes the H-engine and the phasor
  substrate.
- HRM's own gains come from learned latent refinement with deep supervision (ARC Prize
  analysis, TRM), not from enumeration.

## Decision

Build **HRM-Φ** as specified in [[HRM-Phi Design Document v0.1]]:
- a learned, inductive, two-stream latent refiner in PyTorch;
- PEM's phasor algebra as its native geometry (RoPE = binding, QK-norm = torus
  projection, attention = resonator);
- a product-key parametric memory in an SSD-resident `.pt`;
- a semi-embedder and coda for language;
- a pickle-free core file (`.hrmc`) that carries its own contraction certificate and
  self-test.

The existing `hrm/` package stays as the **teacher, judge and baseline**: generators,
verifier, frozen harness and gate, and program search as the ablation baseline. It is no
longer "the model".

## Feasibility checked before planning

The knowledge file needs in-place SSD writes to a normal `.pt`. This was measured on
torch 2.11 + Windows:
- `torch.save` zip entries are stored uncompressed and 64-byte aligned;
- `torch.load(mmap=True)` works;
- a row written through a shared memory map persists, and `torch.load` reads it.

Caveat: the zip CRC-32 goes stale, so the design includes `seal()`.

## Plan (each phase: tests green → graphify update → vault note; no phase starts before the previous gate passes)

| Phase | Deliverable | Acceptance (machine-decidable) | Compute |
|---|---|---|---|
| **N0** scaffold | package `hrmphi/` (name to confirm), strict config dataclasses, test layout | config round-trip; unknown/missing keys refused | — |
| **N1** substrate in torch | planes, binding, fractional power, multi-axis RoPE-as-binding, $\Pi_\varepsilon$, torus attention $\mathcal{A}$ | Prop 1 and Prop 2 hold to 1e-5; `gradcheck` passes on $\Pi_\varepsilon$ and $\mathcal{A}$; exact bind/unbind round-trip | seconds |
| **N2** task field | tokenizer from existing generators (grid, sequence, vector3d) into (content, coords, role) tensors; group augmentation + exact de-augmentation; leave-one-out masking | augment→de-augment is identity on 10k random tasks; every token field reconstructs its source task | seconds |
| **N3** core Φ + training loop | prelude, shared block with role binding, two streams, structured head, halting head, deep supervision with last-cycle gradient, EMA, residual/ρ̂ logging | (a) a tiny model overfits 64 tasks to 100%; (b) on sequence/offset it reaches ≥ 99% on 1,000 unseen generated tasks within a fixed step budget; (c) median residual falls across segments | minutes–1 h |
| **N4** core file `.hrmc` | writer/reader, strict header, SHA-256s, generator-regenerated substrate, probe self-test + certificate | save→load reproduces outputs bit-exactly; one flipped byte → refused; tampered certificate → refused; the format contains no pickle stream | seconds |
| **N5** knowledge file `.pt` | product-key memory layer; `MAP_SHARED` write-through on the `.pt` payload; WAL; `seal()`; schema pairing | `torch.load(weights_only=True)` loads it; an in-place row write persists across processes; kill mid-write → journal recovers a consistent file; zip CRCs valid after `seal()`; schema mismatch refused; an all-zero ("empty brain") memory runs | minutes |
| **N6** test-time memory + sleep | surprise-gated overlay updates, verified commit (E_loo = 0), consolidation behind the existing frozen harness gate + ledger | on a cross-episode recall family, accuracy rises after writes; an unverified overlay never reaches disk (kill test); a regressing consolidation is discarded | ~1 h |
| **N7** language | byte tokenizer, semi-embedder Σ, coda C, program→description data from the symbolic teacher | on held-out tasks the stated answer agrees with the structured head ≥ 95%; with $\lambda_\ell=0$ the head's accuracy is unchanged | hours |
| **N8** experiments | ablation ladder A1–A8, Sudoku-Extreme/Maze comparison, benchmark-v2 depth bands vs program search | every rung reported with numbers in ADR-006, including negative results | hours–days |
| **N9** product | CLI (`python -m hrmphi train / ask / export / load`), README, CLAUDE.md, vault, graphify | CLI round-trip test; docs match code | — |

## Loop design (training and self-learning)

| Loop | Judge (independent, deterministic) | Boundary | Damping |
|---|---|---|---|
| training runs (N3, N8) | held-out generated tasks + the frozen benchmarks; never the training batch | the model never sees the benchmark tasks; seeds are disjoint | fixed step and wall-clock budgets; early stop on a validation plateau |
| test-time memory writes (N6) | leave-one-out energy $E_{\text{loo}}=0$ | writes only touched rows; importance shield; journaled | one write attempt per task; overlay discarded on failure |
| sleep consolidation (N6) | existing hash-locked harness + gate | the loop cannot edit the harness or benchmark | stop after repeated discards; human promote/rollback |

## Consequences

- `hrm/` stops being "the model". Its README and CLAUDE.md sections are re-labelled as
  teacher/judge/baseline in N9, not deleted.
- The torch and transformers dependencies installed last session are now part of the
  project (transformers is not needed by HRM-Φ and can be removed later).
- Benchmark v2 becomes the first cross-architecture comparison (A7).

## Open decisions for the user (as first asked)

1. Package name for the new model (default `hrmphi/`).
2. ARC protocol: inductive (demonstration context, generalizes; recommended as primary)
   vs transductive puzzle-ID (HRM/TRM's numbers), reported side by side.
3. First real benchmark after generated tasks: Sudoku-Extreme (direct HRM/TRM
   comparison) or ARC-AGI-1.
4. Coda tokenizer: byte-level (no dependency; recommended) or a small BPE.

## Resolved decisions (user, 2026-09-27)

| # | Decision | Where it lands in v0.2 |
|---|---|---|
| 1 | Name **hrm-eqb** ("since it relies on equilibrium"); package `hrm_eqb` | whole spec |
| 2 | ARC is solved by the model itself (inductive), not by a wrapper. It has a hormone system and cognition aimed at curiosity: once loaded it notices a missing pattern and asks for the answer. Influenced by karpathy/autoresearch. | §6 hormones (H1), §7 curiosity (Q1), §8 keep gate (G1) |
| 3 | Benchmarks: Sudoku-Extreme **and** ARC **and** language tasks. The low level does general tasks; the high level thinks the solution in language. | §5 L2 (H decodes in the text space), plan N8 |
| 4 | Text stored as rotation vectors; learn arXiv 2603.15031. Knowledge and understanding live in a 4-D vector space. | T1/T2 (rotation-vector text), A1 (2603.15031 is depth-wise attention, used over the thinking chain), T3/T4/K1 (quaternions) |
| — | `.hrmc` must be the full model, not a regeneration rule | §9 F1: all weights, certificate recomputed from them on load |
| — | Coda tokenizer: not answered, so the byte-level default stands | N7 |

Clarified for the record: arXiv 2603.15031 (*Attention Residuals*, Kimi Team) replaces
fixed residual sums with softmax attention over depth. It is not a rotation-vector method.
Rotation-vector text comes from T1 (RoPE is phasor binding).

## Plan (revised for HRM-EQB)

Same gates as above, with the package renamed `hrm_eqb/` and these changes:

| Phase | Change | Status |
|---|---|---|
| **N0** | Certified operators (T5, S1, N1), equilibrium core with certificate and halting (C1, E1–E6), algebra (T1–T4, K1, L1), AttnRes (A1), product keys (M1), hormones (H1), curiosity (Q1), keep gate (G1); 23 theorem tests | **done 2026-09-27** |
| N1 | Task field + prelude: tokens, rotation-vector positions, group augmentation, routing $A_s$ from input | next |
| N3 | Training through the certified core: deep supervision, backprop through the last $k$ cycles (error $\le r^k/(1-r)$, E4) | |
| N4 | `.hrmc` = every weight + budgets; the loader recomputes $\Gamma$, $r_n$ from SVD norms (F1) | |
| N5 | `knowledge.pt` with 4-D quaternion values (K1, M1) | |
| N6 | Curiosity protocol (Answer or Question), hormones wired into segments, autoresearch-style keep gate on large held-out sets | |
| N8 | Sudoku-Extreme, ARC-AGI-1 (inductive), language tasks; ablation A1′ (budgets vs unconstrained) | |
