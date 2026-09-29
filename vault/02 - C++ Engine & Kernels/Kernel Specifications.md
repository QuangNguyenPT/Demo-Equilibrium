---
title: Kernel Specifications
type: kernel
tags:
  - kernel
  - cpp
  - gemm
  - attention
  - norm
  - rope
status: superseded-not-implemented
created: 2026-09-26
updated: 2026-09-28
superseded_by: "[[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]"
relates_to:
  - "[[ADR-001 Hybrid C++ Engine with Python Blueprint]]"
  - "[[HRM-X Native Backend]]"
---

# ⚙️ Kernel Specifications

> [!WARNING] Not implemented as described
> This note describes the ADR-001 C++/Python hybrid engine, which was **never built**; ADR-003 moved it to `legacy/`. The live C++ is only `hrm_x/native/` (`egraph.cpp`, `grid.cpp`) behind the `hrmx.h` C ABI. The current model is [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]] (ADR-008), in `hrm_x/`. Kept as design history.

The math below is textbook-correct and the blocking arguments are sound, but **no kernel in this
note was ever written**. There is no `src/kernels/`, no blocked GEMM, no FlashAttention, no
RMSNorm, no RoPE, no SwiGLU in the current tree. Each source path named below is fictional
relative to the live repo; the implementations survive only under `legacy/cpp_engine/src/kernels/`,
which nothing imports.

The reason is structural, not a failure of effort. When the model changed, the hot paths stopped
being dense matrix math and became **pointer-chasing symbolic code**: union-find over an e-graph,
hash-consing, congruence rebuild, e-matching, and flood fill over a grid. Accelerating those is
worth doing; accelerating a GEMM for a model that has no GEMM is not (spec v0.4 §23).

## 🧾 The kernels that do exist

| File | What it does | Speedup vs Python |
| :--- | :--- | :--- |
| `hrm_x/native/src/egraph.cpp` | union-find, hash-consing, congruence rebuild, e-matching, explanations, conflicts | part of the 1.47× arithmetic path |
| `hrm_x/native/src/grid.cpp` | 4-connected components; one object-level program step | 3.79× |
| `hrm_x/native/include/hrmx.h` | the `extern "C"` surface for both | — |

See [[HRM-X Native Backend]] and [[C-ABI & Interop Specification]]. Exact `Fraction` arithmetic
stays in Python deliberately: arbitrary precision *is* the correctness guarantee, and Amdahl's law
caps any arithmetic speedup at about 1.5× anyway (ADR-008, Rejected).

---

## 1. Blocked GEMM (Matrix Multiplication)

- **Source**: `include/hrm/kernels/gemm.hpp`, `src/kernels/gemm.cpp`
- **Equation**: $C = \alpha (A B) + \beta C$
- **Blocking Dimensions**:
  - $M_{\text{block}} = 64$
  - $N_{\text{block}} = 64$
  - $K_{\text{block}} = 64$
- **Cache Optimization**:
  3-level nested loop tiling ensures sub-matrices fit cleanly inside L1 ($32\text{KB}$) and L2 ($512\text{KB}$–$1\text{MB}$) caches. The innermost accumulation loop is decorated with `#pragma omp simd` to compile into 8-wide AVX2 `vfmadd231ps` instructions.

---

## 2. FlashAttention (Tiled Online Softmax)

- **Source**: `include/hrm/kernels/attention.hpp`, `src/kernels/attention.cpp`
- **Core Motivation**: Naive attention materializes the full $T \times T$ attention matrix into RAM, consuming $O(T^2)$ memory bandwidth. FlashAttention fuses matrix multiply and softmax with online reduction.
- **Online Softmax Update Equations**:
  For each row query $q_i$ and key $k_j$:
  $$
  S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}
  $$
  $$
  m_{\text{new}} = \max(m_{\text{old}}, S_{ij}), \quad \alpha = e^{m_{\text{old}} - m_{\text{new}}}
  $$
  $$
  P_{ij} = e^{S_{ij} - m_{\text{new}}}
  $$
  $$
  l_{\text{new}} = l_{\text{old}} \cdot \alpha + P_{ij}
  $$
  $$
  O_i = O_i \cdot \alpha + P_{ij} \cdot v_j
  $$
  After seeing all keys up to sequence length $T_k$, the output vector is scaled once:
  $$
  O_i \leftarrow \frac{O_i}{l_{\text{final}}}
  $$
- **Memory Footprint**: Strict $O(D)$ scratch space per thread instead of $O(T^2)$.

---

## 3. RMSNorm & Fused Residuals

- **Source**: `include/hrm/kernels/norm.hpp`, `src/kernels/norm.cpp`
- **RMSNorm Formulation**:
  $$
  y = \frac{x}{\sqrt{\frac{1}{D}\sum_{d=1}^D x_d^2 + \epsilon}} \odot \gamma
  $$
- **Fused Kernel (`fused_add_rms_norm`)**:
  Performs the in-place residual addition $x \leftarrow x + \text{residual}$ simultaneously with the sum-of-squares reduction in a single memory pass over the cache line, halving RAM read/write cycles.

---

## 4. Rotary Positional Embeddings (RoPE)

- **Source**: `include/hrm/kernels/rope.hpp`, `src/kernels/rope.cpp`
- **Frequency Table**:
  $$
  \theta_i = \text{base}^{-2i / d}, \quad i \in [0, d/2)
  $$
- **Precomputed Cache**: The `RoPECache` precomputes cosine and sine values for all positions up to `max_seq_len`.
- **Complex Rotation**: Two adjacent vector elements $(x_0, x_1)$ are rotated by complex multiplication:
  $$
  \begin{bmatrix} x_0' \\ x_1' \end{bmatrix} = \begin{bmatrix} x_0 \cos(\theta) - x_1 \sin(\theta) \\ x_0 \sin(\theta) + x_1 \cos(\theta) \end{bmatrix}
  $$

---

## 5. SwiGLU Activation

- **Source**: `include/hrm/kernels/activation.hpp`, `src/kernels/activation.cpp`
- **Formulation**:
  $$
  \text{SwiGLU}(G, U) = \left( G \odot \sigma(G) \right) \odot U
  $$
  Where $\sigma(g) = \frac{1}{1 + e^{-g}}$ is the logistic sigmoid.

---

## 🔗 Related Notes
- [[Engine Architecture & Memory Model]] — Memory allocator hosting kernel inputs/outputs. *(never implemented)*
- [[HRM Recurrent Stepper & ACT]] — Recurrent loop combining these kernels. *(never implemented)*
- [[HRM-X Native Backend]] — **the C++ kernels that exist** (e-graph, grid).
- [[Template - Kernel Spec]] — Template for authoring new kernel documentation.
