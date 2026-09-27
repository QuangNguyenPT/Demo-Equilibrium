---
title: HRM Recurrent Stepper & ACT Halting Logic
type: engine
tags:
  - kernel
  - hrm
  - act
  - recurrence
status: active
created: 2026-09-26
---

# 🔁 HRM Recurrent Stepper & ACT Halting Logic

The Hierarchical Reasoning Model kernel (`include/hrm/kernels/hrm_loop.hpp`) implements the nested multi-timescale recurrence loop and dynamic Adaptive Computation Time (ACT) halting.

---

## 🔄 Algorithm Specification

```mermaid
stateDiagram-v2
    [*] --> InitStates: Receive input_emb
    InitStates --> PrepareInj: z_H = input_emb, z_L = input_emb
    
    state "Reasoning Step (r = 1..R_max)" as Step {
        PrepareInj --> LowLevelLoop: inj = z_H + input_emb
        state "Low-Level Executor (L_sub steps)" as LowLoop {
            LowLevelLoop --> GRU_Low: z_L = GRU(inj, z_L)
            GRU_Low --> LowLevelLoop: sub < L_sub
        }
        LowLoop --> HighLevelUpdate: Pass refined z_L
        HighLevelUpdate --> EvalACT: z_H = GRU(z_L, z_H)
    }

    EvalACT --> CheckHalt: Compute Q = W_q * z_H + b_q
    CheckHalt --> [*]: p_halt >= threshold (Early Exit)
    CheckHalt --> PrepareInj: p_halt < threshold AND r < R_max
    CheckHalt --> [*]: r == R_max (Max Budget Reached)
```

---

## 🧮 Detailed Mathematical Equations

### 1. State Injection
At the start of cycle $r$, the context is injected into the latent recurrence:
$$
\text{inj}_r = z_{H, r-1} + \text{input\_emb}
$$

### 2. Low-Level Sub-Steps
The low-level state $z_L$ performs $L_{\text{sub}}$ fast iterative updates:
For $s = 1 \dots L_{\text{sub}}$:
$$
z_L^{(s)} = \text{GRU}_{\text{low}}\left(\text{inj}_r, z_L^{(s-1)}\right)
$$
Where the GRU cell updates:
$$
r = \sigma(W_r x + U_r h + b_r)
$$
$$
z = \sigma(W_z x + U_z h + b_z)
$$
$$
n = \tanh(W_n x + b_n + r \odot (U_n h + b_{un}))
$$
$$
h' = (1 - z) \odot n + z \odot h
$$

### 3. High-Level Planner Update
Once the low-level executor completes its inner sub-steps, the high-level planner integrates the result:
$$
z_{H, r} = \text{GRU}_{\text{high}}\left(z_L^{(L_{\text{sub}})}, z_{H, r-1}\right)
$$

### 4. ACT $Q$-Head Halting Evaluation
The $Q$-head computes the halting logit $q_{\text{halt}}$ and continuation logit $q_{\text{cont}}$:
$$
\begin{bmatrix} q_{\text{halt}} \\ q_{\text{cont}} \end{bmatrix} = W_Q z_{H, r} + b_Q
$$
The halting probability is:
$$
p_{\text{halt}} = \frac{e^{q_{\text{halt}}}}{e^{q_{\text{halt}}} + e^{q_{\text{cont}}}}
$$
If $p_{\text{halt}} \ge \tau_{\text{ACT}}$, the C++ function breaks immediately and returns:
- `steps_executed`: integer indicating the exact step count taken.
- `halted_early`: boolean flag (1 if halted before `max_steps`).
- `halt_confidence`: float value $p_{\text{halt}}$.

---

## ⚡ Performance Characteristics
- **Execution Speed**: In standalone benchmarks, a full HRM iteration with $D=256$ executes in **$300\mu\text{s}$ – $400\mu\text{s}$** on a single CPU core.
- **Compute Scaling**: On simple inputs, early exit triggers on step 1, saving up to **80% of inference compute** compared to static unrolled loops!

---

## PEM Integration (2026-09-26)

### 9-Feature Q-Head (ControllerFeatures)

```cpp
struct ControllerFeatures {
    float residual;            // ||Δz_h|| / D — kinetic convergence
    float residual_slope;      // dResidual/dt — negative = converging
    float energy;              // 1 - halt_confidence
    float diversity;           // var(z_h) — exploration proxy
    float memory_confidence;   // 0.5 placeholder (overridden by memory module)
    float verification_margin; // halt_confidence - act_threshold
    int   plateau_count;
    int   restarts_used;
};
```

Q-head: `W_q(2,9) @ features + b_q`. Only 18 float weights vs D×2 previously.

### Conjunctive Halt (Hard Rails)

ACCEPT fires only when ALL three hold:
- `residual < eps_kinetic` (kinetic convergence)
- `energy < eps_energy` (semantic convergence)
- `halt_p >= tau_verify` (Q-head verified)

### False-Attractor Detection & Annealed Restart

- kinetic converged but semantic hasn't → RESTART
- sigma = sigma0 × gamma^restart_count, applied to z_h + z_l
- Budget-gated; exhausted → ABSTAIN

### Attention Residuals (RealFormer, in `attention.cpp`)

`score_l += prev_scores_{l-1}` before softmax. Optional `prev_scores` / `scores_out` params on `flash_attention_f32`. Block variant: pass scores only every `attn_residual_block_size` layers.

## 🔗 Related Notes
- [[HRM+LLM Fusion Architecture]] — System design and research foundation.
- [[Engine Architecture & Memory Model]] — Memory allocation for the HRM loop.
- [[Model Blueprint & Graph Layout]] — How this loop is declared in the Python blueprint.
