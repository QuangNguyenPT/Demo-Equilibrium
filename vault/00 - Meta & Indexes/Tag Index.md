---
title: Tag Index & Taxonomy
type: index
tags:
  - meta
  - tags
status: active
created: 2026-09-26
---

# 🏷️ Tag Index & Taxonomy

This vault employs a structured tagging taxonomy to enable quick filtering and graph traversal in Obsidian.

> [!NOTE] Live scope
> The tags below reflect a project that has since moved on. `hrm/` (program-search
> baseline, ADR-003), `hrm_eqb/` (equilibrium baseline, ADR-005/006) and `hrm_x/`
> (current model, ADR-008) are the packages that exist. The C++ engine, arena
> allocator and C-ABI blueprint layer were **never built** and live only in
> `legacy/`; the tags scoped as *legacy* below still describe those historical
> notes and are kept, but nothing in the live project uses them.

---

## System Architecture Tags
- `#architecture` — High-level system design and architectural principles.
- `#blueprint` — **legacy (ADR-001 only)**: Python declarative model graph layout and topology. Never built; exists only in `legacy/python/`.
- `#fusion` — **legacy/ADR-001–002**: Fusion paradigms between HRM and LLM backbones. Documents the unrealised hybrid engine, not the live model.

## Engineering & Compute Tags
- `#cpp` — **legacy (ADR-001/ADR-002 only)**: the native C++ engine, memory allocators and thread pools. Never built; now `legacy/cpp_engine/`. One live exception: the HRM-X e-graph backend in `hrm_x/native/` uses `#cpp-native` below.
- `#kernel` — **legacy (ADR-001/ADR-002 only)**: GEMM, FlashAttention, RoPE, RMSNorm, SwiGLU kernel specifications. Never built; see `vault/02 - C++ Engine & Kernels/`.
- `#memory` — **legacy (ADR-002 only)**: arena allocation, cache lines, zero-allocation paradigms. No arena or custom allocator exists in any live package.
- `#interop` — **legacy (ADR-001/ADR-002 only)**: C-ABI bindings, ctypes, pybind11, pointer exchange. The C-ABI idea survives only as `hrm_x/native/include/hrmx.h` (ADR-008) — tag that `#cpp-native` instead.

## Current Model Tags
- `#hrm-x` — The current model family (HRM-X), package `hrm_x/`. Supersedes `#hrm`.
- `#operator-inference` — The ADR-010 core method: a hypothesis is a law-closed family fitted exactly over ℚ, not a program found by enumeration. (The ADR-008 version of this tag meant the opposite — inferred programs over a hand-written DSL — and remains on the retained v0.4 notes.)
- `#e-graph` — E-graph workspace with provenance (`hrm_x/workspace.py`); equality saturation, rule-based laws. Now serves the v0.4 route inside `hrm_x/`.
- `#spec-v05` — Notes describing the normative [[HRM-X Specification v0.5 - Structural Operator Inference]] (ADR-010). v0.3 and v0.4 are superseded and retained.
- `#adr-010` — Decisions, results and code notes belonging to the current ADR-010 era. **Gates G4 and G5 are not met.**
- `#cpp-native` — Live C++/C-ABI work: the HRM-X e-graph backend in `hrm_x/native/`, behind a differential parity gate. The only native code that exists.

## Baseline Tags
- `#hrm` — Hierarchical Reasoning Model (Wang et al. 2025) **and** the `hrm/` package, which is now the measured program-search baseline (ADR-003), not the model.
- `#hrm-eqb` — The `hrm_eqb/` equilibrium baseline (ADR-005/006): built, measured and theorem-tested (55 tests), superseded as the design target by HRM-X v0.4 and retained. Why it left, and which of its ideas [[ADR-011 Reintroducing EQB Ideas into HRM-X]] proposes to re-take.
- `#recurrence` — Multi-timescale iterative processing ($z_H$ and $z_L$).

## Process & Management Tags
- `#adr` — Architectural Decision Records.
- `#devlog` — Engineering logs, sprint updates, benchmarks.
- `#template` — Standardized note templates for future documentation.
