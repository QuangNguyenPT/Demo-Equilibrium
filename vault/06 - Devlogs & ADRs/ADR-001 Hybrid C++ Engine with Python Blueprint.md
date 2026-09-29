---
title: ADR-001 - Hybrid C++ Engine with Python Blueprint
type: adr
tags:
  - adr
  - architecture
  - cpp
  - python
status: superseded-not-implemented
created: 2026-09-26
superseded_by: "[[ADR-003 Unified Cognitive Model]]"
---

# ADR-001: Hybrid C++ Engine with Python Blueprint Architecture

> [!WARNING] Never implemented — superseded, kept for the record
> This ADR was **accepted on paper but never built**. The C++ engine, the Python
> blueprint layer and the C-ABI bridge described below were never live in any
> measured run; [[ADR-003 Unified Cognitive Model]] moved all of them to `legacy/`
> after evaluation on 2026-09-27 (fp32-only tensors, no phasor kernels, a global
> `-ffast-math` that broke the deterministic FP contract, and a stale build that
> did not compile).
>
> The **C-ABI idea** is the one survivor: it was re-landed, scoped to the
> e-graph kernels, as `hrm_x/native/include/hrmx.h`. The current model is
> **HRM-X v0.4 — Operator Inference over Structured Situations**
> ([[ADR-008 Operator Inference over Structured Situations]]) in `hrm_x/`, and
> [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]] is
> the normative design.
>
> Do not implement against this ADR.

## Context & Problem Statement
Machine learning frameworks frequently suffer from a fundamental tension:
- Python is excellent for rapid prototyping, declarative graph design, and experimentation, but suffers from severe performance penalties, GIL bottlenecks, and memory overhead when executing tight loops or custom recurrences (such as the HRM reasoning loop).
- C++ is exceptional for bare-metal speed, SIMD vectorization, cache blocking, and zero-allocation runtimes, but hardcoding entire neural network architectures in C++ severely slows down research experimentation and makes quick architectural alterations tedious.

## Decision
We decouple the codebase into two strictly defined layers:
1. **Python Blueprint Layer (`python/hrm_blueprint/`)**: Serves as the declarative architectural layout. It builds and mutates layer graphs (`LayerLayout`, `ModelBlueprint`), validates tensor shapes, calculates parameter counts, and configures experiment hyperparameters.
2. **C++ Compute & System Engine (`src/`, `include/`)**: Executes all computational kernels (GEMM, FlashAttention, RMSNorm, RoPE, SwiGLU) and the multi-timescale HRM reasoning loop.
3. **C-ABI Bridge (`hrm_c_api.h`)**: Bridges the two layers using pure C function signatures and zero-copy pointer exchanges with Python's standard `ctypes` library.

## Consequences

### Positive
- Researchers can modify model topology, reasoning depth, and attention mechanisms in Python in seconds.
- Computations execute at full compiled C++ speed without Python interpreter overhead.
- Zero-copy tensor sharing via `wrap_numpy` eliminates costly serialization.
- Independent native CLI binary (`hrm_cli`) allows deployment without Python runtime dependencies.

### Negative / Trade-offs
- Two languages to maintain (C++ for kernels, Python for blueprints).
- C-ABI headers must be kept synchronized when new operators are introduced.
