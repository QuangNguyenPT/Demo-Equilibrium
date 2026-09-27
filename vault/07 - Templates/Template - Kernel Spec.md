---
title: "Template: Kernel Specification"
type: template
tags:
  - template
  - kernel
  - cpp
status: draft
created: {{date}}
---

# ⚙️ Kernel: {{KernelName}}

## Overview
Brief description of the mathematical operator and its purpose within the HRM or LLM model pipeline.

## Mathematical Formulation
$$
y = f(x, w)
$$

## Input & Output Tensors
| Tensor | Shape | DType | Memory Layout | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `input` | `(B, T, D)` | `Float32` | Contiguous, 64B aligned | Primary activation |
| `weight` | `(D, K)` | `Float32` | Row-major | Projection matrix |
| `output` | `(B, T, K)` | `Float32` | Pre-allocated in Arena | Target buffer |

## Algorithm & Cache Strategy
- **L1/L2 Blocking**: Describe tile sizes ($M_{\text{block}}, N_{\text{block}}, K_{\text{block}}$).
- **SIMD Intrinsics**: AVX2 / AVX-512 vector register utilization.
- **Fusion**: Note any fused activations or residual adds.

## Benchmarks & Latency
- Single-thread latency: `X us`
- Multi-thread scaling (4 cores): `Y us`
