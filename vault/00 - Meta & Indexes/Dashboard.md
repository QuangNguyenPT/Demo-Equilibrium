---
title: System Architecture & Knowledge Dashboard
type: dashboard
tags:
  - meta
  - dashboard
  - overview
status: active
created: 2026-09-26
updated: 2026-09-26
---

# 🧠 HRM+LLM Hybrid AI Architecture Dashboard

Welcome to the internal engineering knowledge base for the **Hierarchical Reasoning Model (HRM) + Large Language Model (LLM)** fusion system.

> [!IMPORTANT]
> **Core Architectural Paradigm**:
> - **Python Layer**: Serves as the declarative **blueprint / layout** of the neural network graph. Researchers inspect, mutate, and explore model configurations rapidly without touching compiled code.
> - **C/C++ System Layer**: Executes **all compute operations**: cache-blocked GEMMs, online-softmax FlashAttention, RoPE, RMSNorm, zero-fragmentation memory arenas, and the recursive HRM multi-timescale reasoning loop with ACT halting.

---

## 🧭 Navigation & Maps of Content (MOCs)

| Knowledge Hub | Focus Area | Direct Link |
| :--- | :--- | :--- |
| **Master Index** | Complete vault graph and topic index | [[Master MOC]] |
| **System Philosophy** | Why Python blueprint + C++ system engine | [[Hybrid System Philosophy]] |
| **HRM+LLM Fusion** | Research & architecture design (Architectures A–E) | [[HRM+LLM Fusion Architecture]] |
| **Tensor Data Flow** | Explicit shape mappings $(B, T) \to (B, D) \to (B, V)$ | [[Tensor Flow & Data Pipelines]] |
| **C++ Engine & Kernels** | Low-level C++ engine, memory arena, kernels | [[Engine Architecture & Memory Model]] |
| **Python Blueprint** | Declarative layout API, graph mutators, configs | [[Model Blueprint & Graph Layout]] |
| **C-ABI & Interop** | ctypes & pybind11 zero-copy interop protocol | [[C-ABI & Interop Specification]] |
| **Build & Toolchain** | CMake, Clang++, MSVC build instructions | [[Build & Toolchain Guide]] |
| **Decision Records** | Architectural Decision Records (ADRs) | [[ADR-001 Hybrid C++ Engine with Python Blueprint]] |

---

## ⚡ High-Level System Architecture Diagram

```mermaid
flowchart TD
    subgraph PythonLayer["🐍 Python Blueprint & Layout Layer"]
        CFG["ModelConfig / YAML"] --> BP["ModelBlueprint Graph"]
        BP --> MOD["Dynamic Graph Mutator\n(modify_reasoning_depth)"]
        MOD --> BRG["EngineBridge (ctypes / C-ABI)"]
    end

    subgraph NativeBridge["🔌 C-ABI Interop Boundary"]
        BRG <--> CABI["hrm_c_api.h\n(Zero-Copy Pointer Exchange)"]
    end

    subgraph CPPEngine["⚙️ C++ System & Compute Engine"]
        CABI --> ARENA["ArenaAllocator\n(Zero Fragmentation Bump Arena)"]
        CABI --> KERNELS["Kernels Suite"]
        
        subgraph KernelsDetail["Optimized Kernels"]
            KERNELS --> GEMM["Blocked GEMM / Linear"]
            KERNELS --> ATTN["FlashAttention\n(Tiled Online Softmax)"]
            KERNELS --> ROPE["Rotary Embeddings\n(Precomputed Frequencies)"]
            KERNELS --> NORM["RMSNorm & Fused Residuals"]
            KERNELS --> ACTV["SwiGLU & SiLU"]
            KERNELS --> HRM["HRM Recurrent Stepper\n(High/Low States + ACT Q-Head)"]
        end

        KERNELS --> RUNTIME["Runtime KV Cache & ThreadPool"]
    end

    style PythonLayer fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style NativeBridge fill:#0f172a,stroke:#a855f7,stroke-width:2px,color:#f8fafc
    style CPPEngine fill:#1e1e2e,stroke:#22c55e,stroke-width:2px,color:#f8fafc
```

---

## 📊 Quick System Metrics

- **Backbone Supported**: LLaMA/Mistral-style Decoder Transformer (RMSNorm, SwiGLU, RoPE, GQA/MHA).
- **Reasoning Module**: HRM 2-level Recurrent Engine (High-level Planner $z_H$, Low-level Executor $z_L$).
- **Halting Controller**: Adaptive Computation Time (ACT) Q-Head with dynamic thresholding.
- **Engine Allocation**: 64-byte aligned static memory arena; zero dynamic `malloc` during inference passes.
