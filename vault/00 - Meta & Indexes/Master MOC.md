---
title: Master Map of Content (MOC)
type: moc
tags:
  - meta
  - moc
  - index
status: active
created: 2026-09-26
updated: 2026-09-27
agent: "opencode/big-pickle"
model: "big-pickle"
---

# 🗺️ Master Map of Content

This Master MOC categorizes and links every major engineering document, architectural specification, and kernel reference in this repository.

> **Start here:** [[Code Map]] — every code module mapped to the document that governs it. The knowledge graph cannot link notes to code (markdown extraction needs an LLM key), so this file is the authoritative bridge.

**Current state (2026-09-27):** two architectures coexist. `hrm_eqb/` (HRM-Φ, "reasoning by equilibrium") is the target per [[ADR-005 Neural HRM-Phi Replaces Program Search]]; `hrm/`'s program-search reasoner is the measured baseline. See [[HRM-EQB Specification v0.2 - Theorems and Proofs]] for the normative design.

---

## 🏛️ 01 - Architecture & Philosophy
Concepts, theoretical foundations, and mathematical motivations for the HRM+LLM system.

- [[HRM-EQB Specification v0.2 - Theorems and Proofs]] — **Normative design.** Every claim a theorem, every theorem a test; theorem IDs (T1–T5, M1, K1–K2, G1, F1–F2, H1) are cited directly in `hrm_eqb/` docstrings.
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
