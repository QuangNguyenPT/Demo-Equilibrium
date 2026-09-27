   
title: "Session: 2026 09 26 — Foundation, PEM Integration & Deep Research"
type: session log
tags:
    session
    decisions
    research
    pem
    kimi
    deepseek
    attention residuals
date: 2026 09 26
status: complete
   

# Session: 2026 09 26 — Foundation, PEM Integration & Deep Research

## Goal

1. Bootstrap project from PDF research doc → CLAUDE.md
2. Analyze PEM codebase for HRM compatibility
3. Deep research: Kimi, DeepSeek, HRM paper, kimi k3 in c
4. Implement PEM insights into C++ kernels + Python blueprints
5. Expand knowledge graph beyond code only nodes

   

## Key Decisions Made (Permanent)

| # | Decision | Rationale | File |
|   |         |           |      |
| 1 | Architecture A (separate HRM module) | Modular, reusable, trains as add on | `configs.py` |
| 2 | 9 feature Q head (not full dim) | PEM validates: 18 floats > 1024 for convergence signal | `hrm_controller.py`, `hrm_loop.cpp` |
| 3 | Conjunctive halt (kin AND sem AND ver) | Prevents reward hacking — kinetic convergence alone is false attractor | `hrm_loop.cpp` |
| 4 | Hard rails for ACCEPT/ABSTAIN | QHead controls only CONTINUE/RESTART timing | `hrm_loop.hpp` |
| 5 | Annealed restart σ₀·γᵏ | Escapes false attractors; anneal prevents oscillation | `hrm_loop.cpp` |
| 6 | RealFormer attention residuals | Optional (+0 overhead via nullptr), sparser attention, better gradient flow | `attention.cpp` |
| 7 | Block AttnRes (block_size=4) | O(N/B) memory instead of O(L) — configured via `attn_residual_block_size` | `configs.py` |
| 8 | R ≤ 10 for interactive use | 10 steps ≈ 2× LLM cost; 50 = 5 6× (impractical) | CLAUDE.md |
| 9 | 6 stage training pipeline | Pretrain → Instruct → Reasoning → Integrated → Align → Compress | CLAUDE.md |

   

## Research Findings

### HRM (arxiv:2506.21734)
  27M params, near perfect Sudoku/ARC, 1000 training samples
  Nested H/L loop at different timescales
  ACT Q head: halt reward = task solved − compute penalty
  Truncated BPTT: only final step gradients
  FlashAttention 2/3 required

### PEM (local: C:\Users\Admin\Desktop\Code\AI\PEM)
  VSA native HRM on FHRR substrate (unit complex phasors)
  Resonator = L module, Birkhoff mixing = H module
  9 feature featurization: `[residual, residual_slope, energy, diversity, memory_confidence, verification_margin, plateau_count, restarts_used, bias]`
  KM equilibrium = guaranteed convergence (contractive operator)
  Surprise gated sparse rule memory

### Kimi k1.5 (arxiv:2501.12599)
  Simple RL (online policy mirror descent) + 128K context = o1 level
  No MCTS, no value functions needed
  77.5 AIME, 96.2 MATH 500
  Long2short: train R=50 → distill R=5

### Kimi VL (arxiv:2504.07491)
  MoE VLM: 16B total / 2.8B activated
  MoonViT (native resolution) + MLP projector + Moonlight MoE
  Kimi VL Thinking 2506: Long CoT SFT+RL

### Kimi K2 (arxiv:2507.20534)
  1T total / 32B activated, 384 experts, 8/token
  MuonClip optimizer, 15.5T tokens, agentic focus
  MLA (Multi head Latent Attention) for KV cache compression

### DeepSeek VL2 (arxiv:2412.10302)
  Dynamic tiling (native res tiles)
  MLA: KV → latent vectors (O(D_latent) vs O(n_heads × D_head))
  3 sizes: 1B/2.8B/4.5B activated

### Attention Residuals Research
  **RealFormer** (He et al. 2021): pass pre softmax scores `score_l = QK^T/√D + scores_{l 1}`
  **AttnRes** (arxiv:2603.15031): replace fixed residual with softmax over preceding layer outputs. 1.25× compute advantage. Fixes Pre LN dilution.
  **ResiDual (PPLN)**: dual Pre LN + Post LN residuals. Drop in replacement.
  **RMT** (ICML 2025): outer product memory matrix. 58% fewer FLOPs.
  **Chose**: RealFormer (simplest, zero overhead via optional ptrs)

### Kimi K3 in C (github.com/FareedKhan dev/kimi k3 in c)
  2.78T params on single CPU in 8.24 GB RAM
  Pure C99, AVX2/FMA, `O_DIRECT`, no BLAS, no GPU
  Four reductions: MXFP4 → MoE sparsity → KDA constant memory → trunk streaming
  675× total memory reduction, byte identical output at all memory budgets
  See: [[Concept   The Four Memory Reductions]], [[Concept   Kimi Delta Attention (KDA)]]
  See: [[Concept   Microscaling MXFP4 & E8M0 Quantization]], [[Concept   Direct IO & NVMe Streaming]]

   

## Files Modified This Session

### Created
  `CLAUDE.md` — project overview, key insights, code conventions
  `vault/` — full Obsidian vault with 25+ notes
  `scripts/update_knowledge_graph.ps1` — auto update script
  `python/hrm_blueprint/configs.py` — PEM fields added to HRMConfig + TransformerConfig
  `python/hrm_blueprint/hrm_controller.py` — 6 layer blueprint (was 4)
  `python/hrm_blueprint/transformer.py` — attn_residual flags in hyperparams
  `vault/Knowledge Graph/` — 10+ concept nodes for Kimi K3, HRM, attention

### Replaced (breaking changes)
  `include/hrm/kernels/hrm_loop.hpp` — ControllerFeatures struct, HRMLoopConfig, updated result
  `src/kernels/hrm_loop.cpp` — 9 feat Q head, conjunctive halt, annealed restart
  `include/hrm/kernels/attention.hpp` — RealFormer prev_scores/scores_out params
  `src/kernels/attention.cpp` — RealFormer attention residual implementation

### Weight Shape Breaking Change
Old `W_q` shape: `(2, D)` — operated on full z_h vector
New `W_q` shape: `(2, 9)` — operates on 9 feature controller vector
**Impact**: Any saved checkpoints with old W_q must be retrained or manually migrated.

   

## Open Questions / Next Steps

  [ ] Implement actual PyTorch model code (not just blueprints)
  [ ] Training loop `train.py` with DDP/ZeRO
  [ ] Data pipeline (tokenizer, dataset builders)
  [ ] Inference `generate.py`
- [ ] Tests: unit + forward pass integration
- [ ] MLA integration (replace MHA in LLM encoder for KV compression)
- [ ] MoE routing in HRM module (different experts per reasoning type)
- [ ] RL training (Stage 5): simple policy gradient, 128K context
- [ ] Long2short distillation: train R=50, serve R=5
- [ ] Port kimi k3 in c memory techniques for inference optimization
- [ ] Project owner adds the extraction key to the shared `.env` (no personal key) -> re-run graphify deep mode on vault docs

   

## Artifact

Published research report artifact: https://claude.ai/artifact/6Ayashqr1SqTpnmvb1QAkU
Contains: HRM deep analysis, PEM↔HRM mapping table, Kimi/DeepSeek research synthesis

   

## Related Nodes

  [[Concept   Hierarchical Multi Timescale Reasoning (HRM)]]
  [[Concept   Adaptive Computation Time (ACT)]]
  [[Concept   Attention Residuals (RealFormer & AttnRes)]]
  [[Concept   The Four Memory Reductions]]
  [[HRM Recurrent Stepper & ACT]]
  [[HRM+LLM Fusion Architecture]]
  [[ADR 001 Hybrid C++ Engine with Python Blueprint]]
