---
title: "Concept: Adaptive Computation Time (ACT)"
type: concept
tags:
  - concept
  - act
  - dynamic-compute
  - halting
  - q-learning
status: active
created: 2026-09-26
---

# ⏱️ Concept: Adaptive Computation Time (ACT)

**Adaptive Computation Time (ACT)** (originating in Graves, 2015 and adapted into a Q-learning policy in sapientinc/HRM) is an algorithmic mechanism that enables neural networks to dynamically allocate computational depth based on problem difficulty.

Instead of running a fixed number of reasoning layers on every single input, ACT allows the model to "ponder" variable steps: simple questions finish in **1 step**, while complex logical puzzles receive up to **$R_{\text{max}}$ iterations**.

---

## 🎲 The Q-Learning Halting Formulation

In HRM, the halting decision is framed as a reinforcement learning / temporal difference control problem:

```mermaid
flowchart LR
    State["High-Level State z_H"] --> QHead["Linear Q-Head: W_q · z_H + b_q"]
    QHead --> Logits["Q-Values: [q_halt, q_continue]"]
    Logits --> Softmax["Softmax Halting Probability p_halt"]
    Softmax --> Decision{"p_halt ≥ Threshold τ?"}
    
    Decision -- Yes --> Halt["Halt Iteration Immediately\n(Save CPU/GPU cycles)"]
    Decision -- No --> Continue["Execute Next Reasoning Cycle\n(Increase Depth)"]

    style Halt fill:#15803d,stroke:#22c55e,color:#f0fdf4
    style Continue fill:#b45309,stroke:#f59e0b,color:#fffbeb
```

### 1. Halting Logits
At reasoning cycle $r$:
$$
\begin{bmatrix} q_{\text{halt}} \\ q_{\text{continue}} \end{bmatrix} = W_Q z_{H, r} + b_Q
$$

### 2. Halting Probability
The normalized halting confidence is evaluated:
$$
p_{\text{halt}} = \frac{e^{q_{\text{halt}}}}{e^{q_{\text{halt}}} + e^{q_{\text{continue}}}}
$$

### 3. Early Termination Condition
If $p_{\text{halt}} \ge \tau$ (where $\tau \in [0.5, 0.7]$ is user-configurable in `configs.py`), the C++ loop breaks immediately, returning the state and elapsed step count:
- Easy query: Exits at step 1 ($\approx 1\times$ LLM cost).
- Medium query: Exits at step 3 ($\approx 1.2\times$ LLM cost).
- Hard puzzle: Exits at step 8 ($\approx 1.8\times$ LLM cost).

---

## 📈 Economic & Latency Impact

In a production serving environment:
- **80% of user queries** are simple factual or retrieval requests that do not require multi-step reasoning.
- Fixed unrolled loops waste compute on all 80% of queries.
- ACT achieves up to **60%–80% overall inference latency savings** across real-world workloads while retaining peak reasoning depth on challenging queries.

---

## 🔗 Related Knowledge Graph Nodes
- [[Concept - Hierarchical Multi-Timescale Reasoning (HRM)]] — Host architecture for ACT.
- [[HRM Recurrent Stepper & ACT]] — C++ implementation.
- [[Concept - The Four Memory Reductions]] — Synergistic compute/memory optimization.
- [[Semantic Relationship Matrix]] — Ontological links.
