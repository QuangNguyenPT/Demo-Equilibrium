---
title: Master Map of Content (MOC)
type: moc
tags:
  - meta
  - moc
  - index
status: active
created: 2026-09-26
updated: 2026-09-26
---

# 🗺️ Master Map of Content

This Master MOC categorizes and links every major engineering document, architectural specification, and kernel reference in this repository.

---

## 🏛️ 01 - Architecture & Philosophy
Concepts, theoretical foundations, and mathematical motivations for the HRM+LLM system.

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

---

## 📜 06 - Devlogs & Decision Records (ADRs)
Historic decisions and development trajectory.

- [[ADR-001 Hybrid C++ Engine with Python Blueprint]] — Why we separated declarative Python layout from C++ compute execution.
- [[ADR-002 Zero Allocation Memory Arena]] — Rationale for pre-allocated bump arenas over heap allocations.
- [[Devlog - 2026-09 - Project Foundation]] — Inception and initial milestone delivery.

---

## 📋 07 - Templates
Reusable blueprints for future expansion.

- [[Template - Kernel Spec]] — Standard template for documenting a new C++ compute kernel.
- [[Template - Model Blueprint]] — Template for proposing a new architectural model blueprint.
- [[Template - ADR]] — Template for authoring Architecture Decision Records.
- [[Template - Devlog]] — Template for daily/sprint engineering logs.
