---
title: "Concept: Attention Residuals (RealFormer & AttnRes)"
type: concept
tags:
  - concept
  - attention
  - residual
  - transformer
  - realformer
  - attnres
  - gradient-flow
status: active
created: 2026-09-26
---

# 🔁 Concept: Attention Residuals (RealFormer & AttnRes)

"Attention residuals" — passing attention state **between** transformer layers — is distinct from the standard **within-layer** residual connection `h = h + f(h)`. Multiple related but distinct techniques exist.

---

## 1. RealFormer (He et al., 2021) — Pre-Softmax Score Passthrough

**Core idea**: Layer $l$ receives the raw pre-softmax attention logits from layer $l-1$ as a residual.

$$
\text{score}_l = \frac{Q_l K_l^T}{\sqrt{D}} + \text{score}_{l-1}
$$

**Properties**:
- Creates direct "memory" path through attention maps across layers
- Training stabilizer: gradients flow directly through attention patterns, not just through values
- Sparser attention: once a token attends to a position, it tends to keep doing so
- Zero overhead: `prev_scores` ptr = `nullptr` → no-op, disabled by default
- `scores_out` ptr stores current layer's scores for next layer to consume

**Where used**:
- RealFormer original paper (BERT-scale pretraining)
- Kimi K3 in C: "layers that look back" (section 8 of README)
- **This project**: `src/kernels/attention.cpp` — implemented via optional `prev_scores`/`scores_out` params

**Code (attention.cpp)**:
```cpp
// RealFormer: add prev layer's pre-softmax scores
if (prev_h != nullptr) score += prev_h[i * T_k + j];

// Store for next layer
if (out_s != nullptr) out_s[i * T_k + j] = score;
```

---

## 2. AttnRes (arXiv:2603.15031, Kimi Team, Mar 2026) — softmax attention over depth

*Verified from the paper on 2026-09-27. Earlier project notes wrongly credited the
"1.25×" result to RealFormer-style score passthrough (§1); the two are different
mechanisms. RealFormer adds attention **scores** between layers. AttnRes attends over
layer **outputs**.*

**Core idea.** A PreNorm residual accumulates every layer output with fixed unit weight,
$h_l=\sum_{i<l}v_i$. That makes hidden-state magnitude grow with depth and dilutes early
layers. AttnRes replaces the fixed sum with softmax attention over depth:

$$
h_l=\sum_{i=0}^{l-1}\alpha_{i\to l}\,v_i,\qquad
\alpha_{i\to l}=\frac{\exp\!\big(w_l^{\top}\,\mathrm{RMSNorm}(v_i)\big)}{\sum_{j<l}\exp\!\big(w_l^{\top}\,\mathrm{RMSNorm}(v_j)\big)},\qquad
v_0=\text{token embedding},\;v_i=f_i(h_i)
$$

- $w_l\in\mathbb{R}^d$ is **one learned pseudo-query per layer**. The query is a
  parameter, but the weights are still input-dependent because the keys are the token's
  own intermediate states.
- It **must be zero-initialized**, which makes AttnRes exactly a uniform average at
  initialization (the paper found this necessary for stable training).
- The RMSNorm on keys (no learned scale) stops large-magnitude layers from monopolizing
  the softmax. Ablation: removing it hurts both variants.
- **Time–depth duality:** standard residuals are depth-wise *linear* attention. AttnRes
  completes the linear→softmax transition for depth, as attention did for sequence.

**Block AttnRes.** The $L$ layers are split into $N$ blocks:
- inside a block, outputs are summed into a partial sum $b_n^i$;
- across blocks, each layer attends over $\{b_0=\text{embedding},b_1,\dots,b_{n-1},b_n^{i-1}\}$;
- memory and communication drop from $O(Ld)$ to $O(Nd)$;
- two-phase compute: a batched inter-block pass for all $S$ pseudo-queries of a block,
  then a sequential intra-block pass merged with the online-softmax identity.

**Results.**
- Block AttnRes matches the loss of a baseline trained with **1.25× more compute**.
- In Kimi Linear (48B total / 3B active, 1.4T tokens), output magnitudes stay bounded
  across depth, gradient norms spread more uniformly, and every downstream task
  improves.

**Why it holds (HRM-EQB Lemma A1).** The weights are a convex combination, so
$h_l\in\mathrm{conv}\{v_i\}$ and $\|h_l\|\le\max_i\|v_i\|$. The plain residual sum only
guarantees $\|h_l\|\le l\max_i\|v_i\|$.

**Use in HRM-EQB.** Recurrent depth *is* the thinking chain. AttnRes is applied over the
sequence of segment equilibria $h^*_1,\dots,h^*_s$:
- to aggregate the thought chain for the language coda (the semi-embedder);
- as context for the next segment (injection $c_H$). It cannot be the initializer: the
  equilibrium is independent of its start (E2), so an initializer would receive no gradient.

See [[HRM-EQB Specification v0.2 - Theorems and Proofs]].

---

## 3. ResiDual (PPLN) — Dual Residual Connections

**Core idea**: Both Pre-LN and Post-LN residuals applied simultaneously.

$$
h_l = \underbrace{h_{l-1} + f(\text{Norm}(h_{l-1}))}_{\text{Pre-LN branch}} + \underbrace{\text{Norm}(h_{l-1} + f(h_{l-1}))}_{\text{Post-LN branch}}
$$

**Properties**:
- Avoids gradient vanishing (Post-LN issue) via Pre-LN branch
- Avoids representation collapse (Pre-LN issue) via Post-LN branch
- Drop-in replacement for either standard residual variant

---

## 4. RMT — Residual Matrix Transformer (ICML 2025)

**Core idea**: Replace residual stream with an outer-product memory matrix (associative memory).

$$
M_l = M_{l-1} + k_l \otimes v_l
$$

**Properties**:
- 58% fewer FLOPs than standard attention residual
- 25% fewer parameters
- Based on Kohonen/Anderson associative memory theory

---

## Implementation in This Project

**Chosen**: RealFormer (minimal complexity, zero overhead when disabled)

**Config** (`configs.py`):
```python
use_attn_residuals: bool = False      # enable/disable
attn_residual_block_size: int = 4     # Block AttnRes: layers per block
```

**Blueprint** (`transformer.py`): passes flags to attention hyperparams including `layer_idx`.

**C++ kernel** (`attention.hpp`):
```cpp
void flash_attention_f32(
    // ... core args ...
    const float* prev_scores,   // optional: pre-softmax scores from layer l-1
    float* scores_out           // optional: store pre-softmax scores for l+1
);
```

---

## Comparison Table

| Technique | Complexity | Memory | Compute | Adopted Here |
|-----------|-----------|--------|---------|-------------|
| RealFormer | O(1) extra | +T² per layer pair | ~0% | ✅ (optional) |
| AttnRes | O(L) or O(B) | +L×hidden | +25% → -20% net | Config ready |
| ResiDual | 2× residuals | +hidden | ~+10% | Not yet |
| RMT | Outer product | +D² | -58% | Future |

---

## 🔗 Related Knowledge Graph Nodes

- [[Concept - Hierarchical Multi-Timescale Reasoning (HRM)]] — attention residuals aid HRM loop convergence
- [[Concept - Kimi Delta Attention (KDA)]] — separate mechanism: constant-memory recurrent attention
- [[Concept - Multi-Head Latent Attention (MLA)]] — separate mechanism: KV latent compression
- [[Session - 2026-09-26 - Foundation & Research]] — session where this was researched and implemented
