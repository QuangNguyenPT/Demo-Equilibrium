---
title: "Concept: PEM — Phasor Equilibrium Machine"
type: concept
tags:
  - concept
  - pem
  - vsa
  - fhrr
  - hrm
  - arc
  - reasoning
status: active
created: 2026-09-26
---

# 🌀 Concept: PEM — Phasor Equilibrium Machine

PEM is a VSA-native implementation of the HRM architecture for non-autoregressive grid reasoning (ARC tasks). It validates the H/L halting loop concept using pure algebra — no gradient descent in the substrate layer.

**Location**: `C:\Users\Admin\Desktop\Code\AI\PEM`

---

## FHRR Substrate Algebra

Symbols = D-dimensional unit complex vectors (phasors on the unit circle).

| Operation | Formula | Meaning |
|-----------|---------|---------|
| `bind(a, b)` | `a * b` | Elementwise complex multiply (phase addition) |
| `unbind(a, b)` | `a / b` | Bind with conjugate (exact inverse) |
| `power(x)` | `e^(iθx)` | Fractional binding — Spatial Semantic Pointers |
| `bundle(vecs)` | `sum + torus_project` | Superposition + normalization |
| `similarity` | `Re⟨a,b⟩/D ∈ [-1,1]` | Cosine on complex unit circle |

**Key property**: algebra is zero-parameter. All "knowledge" lives in Engrams (quantized memory), not learned weights.

---

## Engine: Resonator + Equilibrium Loop

### Resonator (f_L = L-module)
Multi-stream soft VSA unbinding. n_streams explore hypothesis space at different temperatures (`log_beta`: 4→24). Scores candidates by similarity to both shared task structure and per-example delta residues.

Small learned params only: `log_beta`, `w_prior`, `gain`.

### BudgetController (Halt/Continue)
4 actions:
- **ACCEPT**: conjunctive halt — kinetic R + semantic E + verification margin all pass simultaneously
- **CONTINUE**: keep iterating
- **RESTART**: false attractor detected → apply annealed phase perturbation σ₀·γᵏ
- **ABSTAIN**: restart budget exhausted → return best-so-far

Adaptive λₜ for KM convergence guarantee.

### QHead (ACT-equivalent)
- Linear Q over **9-feature vector** (not full z_h)
- TD(0) Q-learning update
- Controls only CONTINUE vs RESTART timing
- **Hard rails**: ACCEPT/ABSTAIN are conjunctive conditions — never delegated to learner

### 9 Controller Features

```python
[
    residual,           # ||z_h - z_h_prev|| / D — kinetic convergence
    residual_slope,     # rate of change of residual
    energy,             # ||z_h||² / D — semantic energy level
    diversity,          # variance across hidden dims — state diversity
    memory_confidence,  # similarity to best known candidate
    verification_margin,# margin between top-2 candidates
    plateau_count,      # steps without improvement
    restarts_used,      # restarts consumed so far
    bias                # constant 1.0
]
```

---

## PEM ↔ HRM Architecture Mapping

| PEM Component | HRM Equivalent |
|---------------|----------------|
| Resonator (f_L) — multi-stream VSA factorization | L-module — rapid low-level executor |
| Birkhoff-projected mixing (M_H) — contractive | H-module — slow high-level planner |
| QHead (CONTINUE/RESTART, 9-feature linear Q) | ACT Q-head — halt/continue |
| FHRR phasor vectors (unit complex, zero-param) | Dense token embeddings (learned) |
| KM equilibrium loop (fixed-point iteration) | Nested recurrent loop |
| Conjunctive halt (kin+sem+ver simultaneously) | **Implemented** in hrm_loop.cpp |
| Annealed restart σ₀·γᵏ | **Implemented** in hrm_loop.cpp |
| 9-feature featurization | **Implemented** in hrm_controller.py |

---

## Key Insight: Substrate Independence

PEM proves the H/L halting loop works on ARC **without gradients in the substrate layer** — pure algebra + tiny Q-head. This validates that HRM's architectural structure is substrate-independent:

> Same architecture, different substrate → same emergent behavior

Implications:
- VSA phasors (PEM) vs dense tokens (HRM) → both work for structured reasoning
- The H/L split timing + ACT halting is the critical architectural invariant
- Parameter count can be tiny (9-feature Q = 18 floats) vs typical LLM head (768-dim Q)

---

## Divergences from Dense HRM

| Aspect | PEM | HRM |
|--------|-----|-----|
| Knowledge storage | Engrams (VSA memory) | Learned parameters |
| Symbol algebra | FHRR phase arithmetic | Matrix multiply |
| Similarity | Cosine on unit circle | Dot product |
| Factorization | VSA unbinding | Gradient-based |
| GPU-friendly | No (phase ops irregular) | Yes (matmul-heavy) |

---

## 🔗 Related Knowledge Graph Nodes

- [[Concept - Hierarchical Multi-Timescale Reasoning (HRM)]] — PEM is a VSA-native HRM implementation
- [[Concept - Adaptive Computation Time (ACT)]] — QHead is ACT equivalent
- [[HRM Recurrent Stepper & ACT]] — C++ implementation with PEM features integrated
- [[Session - 2026-09-26 - Foundation & Research]] — PEM analysis session
