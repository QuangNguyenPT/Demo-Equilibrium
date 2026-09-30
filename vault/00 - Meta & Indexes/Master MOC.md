---
title: Master Map of Content (MOC)
type: moc
tags:
  - meta
  - moc
  - index
status: active
created: 2026-09-26
updated: 2026-09-29
agent: "opencode/stealth/space-bunny-alpha"
model: "space-bunny-alpha"
---

# 🗺️ Master Map of Content

This Master MOC categorizes and links every major engineering document, architectural specification, and kernel reference in this repository.

> **Start here:** [[How It Works]] — how the system actually works: the loop, both inference routes, the algorithms, and an honest account of what is not finished. Then [[Task Plan]] — the canonical plan: verified state, next gates, and open debt. [[Code Map]] — every code module mapped to the document that governs it. [[00 - Code Subsystems|Code Subsystems]] is the generated companion: one note per subsystem, linked to the specs, ADRs and sessions that govern or changed it (`scripts/knowledge_map.py`).

**Current state (2026-09-29):** three generations coexist, only the third is live. [[HRM-X Specification v0.5 - Structural Operator Inference]] ([[ADR-010 Structural Operator Inference]]) is built in `hrm_x/` as a **measured** model: a hypothesis is a law-closed family fitted exactly over ℚ rather than a program found by enumeration, verification is the dual solve `Aᵀλ = r`, and deriving is a lookup. It is 7.2× cheaper than v0.4 on operators its language covers, and **worse** on operators it does not — 5 confidently wrong out of 6 against 0 wrong and 4 abstentions. **Gates G4 and G5 are not met.** The v0.4 program-search route is kept inside `hrm_x/` as the measured baseline, and [[ADR-011 Reintroducing EQB Ideas into HRM-X]] proposes which of HRM-EQB's ideas would be worth re-taking. `hrm_eqb/` (HRM-Φ, [[ADR-005 Neural HRM-Phi Replaces Program Search]], normative design in [[HRM-EQB Specification v0.2 - Theorems and Proofs]]) is a baseline: it learned patterns but no arithmetic (ADR-006). It was built and measured, then moved out of the live tree to `legacy/hrm_eqb/` on 2026-09-29 — an audit found zero files outside its own test suite importing it (`legacy/README.md`); its 55 tests are in `legacy/tests/` and no longer run by default. `hrm/` stays a live package: `scripts/make_benchmark.py` imports `hrm.learning`.

---

## 🏛️ 01 - Architecture & Philosophy
Concepts, theoretical foundations, and mathematical motivations for the HRM+LLM system.

- [[HRM-X Specification v0.5 - Structural Operator Inference]] — **Current model (ADR-010).** A task is situation + operator + question; the operator is a law-closed family fitted exactly, determinacy and open-world coverage decide when it may answer, verification is the dual solve, and the value-of-computation controller and modulators are unchanged from v0.4.
- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]] — **Superseded (ADR-008), retained.** Program search over a hand-written DSL with an e-graph workspace; now the measured baseline inside `hrm_x/`.
- [[ADR-011 Reintroducing EQB Ideas into HRM-X]] — **Proposed.** Why HRM-EQB left, and the three certifiable ideas (rival margin, leave-one-out provenance, statistical keep gate) that would apply its method to HRM-X's objects.
- [[HRM-X Specification v0.3 - Understanding by Construction]] — **Superseded design (ADR-007).** Define understanding formally (extrapolation, definitions, faithful explanation, honest ignorance), then derive the model: grounded residue-phasor algebra, executable concepts, transformer only for conventions; every theorem a capability.
- [[HRM-EQB Specification v0.2 - Theorems and Proofs]] — **Normative design of the `hrm_eqb/` baseline** (now `legacy/hrm_eqb/`). Every claim a theorem, every theorem a test; theorem IDs (T1–T5, M1, K1–K2, G1, F1–F2, H1) are cited directly in its module docstrings.
- [[HRM-Phi Design Document v0.1]] — Phasor HRM-Φ: plane algebra shared by PEM and the transformer, and the reasoning that replaces program search.
- [[Hybrid System Philosophy]] — Deep explanation of why Python handles the layout blueprint while C++ executes everything computational.
- [[HRM+LLM Fusion Architecture]] — Detailed analysis of architectures A through E, comparing latent loops, separate reasoners, and modular trade-offs.
- [[Tensor Flow & Data Pipelines]] — Complete tensor shape transformations across the entire inference and reasoning pipeline.

---

## ⚙️ 02 - C++ Engine & Kernels
Low-level bare-metal systems engineering, memory arenas, SIMD optimizations, and kernel implementations.

- [[Engine Architecture & Memory Model]] — Memory arena design, cache line alignment, and zero-allocation runtime.
- [[Kernel Specifications]] — Mathematical definitions, blocking strategies, and performance characteristics for GEMM, FlashAttention, RoPE, RMSNorm, and SwiGLU.
- [[HRM Recurrent Stepper & ACT]] — Multi-timescale recurrence loop between High-level Planner ($z_H$) and Low-level Executor ($z_L$), plus learned ACT halting logic.

---

## 🐍 03 - Python Blueprint Layer
The declarative, experimental Python API for building and modifying models.

- [[Model Blueprint & Graph Layout]] — How researchers build and manipulate layers, inspect parameters, and serialize architectures in Python.
- [[Configuration & Hyperparameters]] — Strongly typed configuration schema for experiment control.

---

## 🔌 04 - Interop & C-ABI Bridge
How Python blueprints communicate with the C++ engine at maximum bandwidth.

- [[C-ABI & Interop Specification]] — The exact C function signatures, memory ownership conventions, and ctypes/pybind11 bindings.

---

## 🛠️ 05 - Engineering & Guides
Operational workflows, build instructions, and developer best practices.

- [[Build & Toolchain Guide]] — Compiling the C++ engine on Windows (Clang++ / MSVC) and Linux.
- [[Development Workflow & Conventions]] — Logging tags `[TAG]`, testing requirements, and directory layout conventions.
- [[GPU Safety - VRAM Oversubscription and DPC Watchdog]] — The measured cause of the Windows crash and the hard VRAM cap now enforced in `hrm_eqb/device.py`.
- [[Why v2 Did Not Prove the Model]] — Why benchmark v2's flat ablation ladder falsified the flagship claims, and the four changes that would make them falsifiable.

---

## 📜 06 - Devlogs & Decision Records (ADRs)
Historic decisions and development trajectory.

- [[ADR-001 Hybrid C++ Engine with Python Blueprint]] — Why we separated declarative Python layout from C++ compute execution.
- [[ADR-002 Zero Allocation Memory Arena]] — Rationale for pre-allocated bump arenas over heap allocations.
- [[ADR-003 Unified Cognitive Model]] — The unified model: phasor substrate, hierarchical reasoner, sleep-gated learning.
- [[ADR-004 Discriminating Benchmark and Reasoning Ceiling]] — Benchmark v1/v2, what the instrument can and cannot show.
- [[ADR-005 Neural HRM-Phi Replaces Program Search]] — **The pivot.** Program search becomes a neural HRM-Φ trained core.
- [[Devlog - 2026-09 - Project Foundation]] — Inception and initial milestone delivery.

---

## 🗂️ 08 - Chat Context & Sessions
The decision chain, in order. Start at [[00 - Session Index]].

- [[Session - 2026-09-26 - Foundation & Research]] — Project bootstrap, PEM analysis, Kimi/DeepSeek research.
- [[Session - 2026-09-26 - Knowledge Graph Expansion]] — Vault and information-path design.
- [[Session - 2026-09-27 - Unified Model Build]] — Assembling the unified `hrm/` model.
- [[Session - 2026-09-27 - Benchmark v2 and the Reasoning Ceiling]] — Benchmark v2, the four domain defects, and the ablation ladder.
- [[Session - 2026-09-27 - HRM-EQB Theorems]] — Theorem research and the verified algebra.
- [[Session - 2026-09-27 - HRM-EQB Full Build]] — The N1–N8 build, defects found by measurement.
- [[Session - 2026-09-27 - HRM-Phi Design]] — HRM-Φ design and measured feasibility.

---

## 🔭 Knowledge Graph
- [[00 - Knowledge Graph Index]] — conceptual entities and their relations.
- [[Code Map]] — code ↔ document bridge (the graph cannot do this without a key).

---

## 📋 07 - Templates
Reusable blueprints for future expansion.

- [[Template - Kernel Spec]] — Standard template for documenting a new C++ compute kernel.
- [[Template - Model Blueprint]] — Template for proposing a new architectural model blueprint.
- [[Template - ADR]] — Template for authoring Architecture Decision Records.
- [[Template - Devlog]] — Template for daily/sprint engineering logs.
