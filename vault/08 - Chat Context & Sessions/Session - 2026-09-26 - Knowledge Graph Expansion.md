---
title: "Session: 2026-09-26 — Knowledge Graph Expansion"
type: session-log
tags:
  - session
  - knowledge-graph
  - chat-context
  - kimi-k3
  - documentation
date: 2026-09-26
status: complete
---

# Session: 2026-09-26 — Knowledge Graph Expansion

## Goal

Expand knowledge graph from code-only to full information coverage:
- Chat session context (plans, decisions, research paths)
- Research papers as concept nodes
- Kimi K3 deep architecture research
- "Information path" tracking across sessions

---

## Key Decisions Made

| # | Decision | Rationale |
|---|---------|-----------|
| 1 | Chat sessions → vault/08 folder | Sessions are knowledge nodes, not just logs |
| 2 | Session Index tracks decision chain | "Information path" = traceability from question → decision → implementation |
| 3 | Concept nodes for all researched papers | Papers become queryable graph nodes linked to code |
| 4 | graphify indexes vault/ markdown | With the project-scoped key in `.env`, deep mode extracts semantic content from session logs |

---

## Kimi K3 Deep Architecture (from README)

### Stats
- 2.78T parameters, 93 transformer layers, 896 routed experts per layer
- 16 experts selected per token (+ 2 always-resident shared experts)
- Checkpoint: 1.56 TB on disk (1,560 GB)
- Peak RSS: 8.24 GB (675× reduction from uncompressed BF16)

### Four Reductions
1. **MXFP4** (Reduction 1, 3.56×): Experts ship at 4-bit; 0.53125 bytes/param (includes shared E8M0 scale)
2. **MoE Sparsity** (Reduction 2, 13.7×): 896 experts per layer, only 16 fire per token. 82,432 routed experts never in RAM. Dense trunk 108.81 GB + embeddings 4.70 GB = 113.49 GB.
3. **KDA + MLA** (Reduction 3): Constant-memory attention (O(1) state, no T² score matrix). MLA: KV → 512-dim latent.
4. **Trunk Streaming** (Reduction 4, 13.8× combined): 93 layers packed into 109 GB file with known offsets. `O_DIRECT` reads bypass OS page cache. Stream only active layers. Memory budget = dial.

### Invariant: Byte-Identical Output
Same token sequence at 8 GB and 224 GB. More RAM = faster, never different. Proven via gate ladder:
- Gate 1: teacher forcing 32/32 positions match
- Gate 2: greedy decode 20/20 tokens match
- Gate 3: incremental + KV cache + KDA state 20/20 tokens match

### Architecture Sections (Part II of README)
- §1: Checkpoint reading from headers (safetensors)
- §2: Config reader (refuses to guess)
- §3: Tokenizer (byte-exact)
- §4: MXFP4 - zero-copy unpack
- §5: Floating-point contract (`-ffp-contract=off`)
- §6: KDA - constant-memory attention
- §7: MLA - latent KV
- §8: **Attention residuals** ("layers that look back" = RealFormer-style)
- §9: Top-16 expert selection from 896
- §10: Trunk packing (93 layers, one contiguous file)
- §11: Streaming trunk = memory dial
- §12: LRU expert cache
- §13: Cache sizing from trace replay

### Codebase Structure
```
kimi-k3-in-c/
├── src/
│   ├── 7 C files (176 KB total engine)
│   ├── io/k3_portable_io.h  (O_DIRECT, posix_memalign, getrusage)
├── tests/fixtures/expert_trace.bin  (100,096 expert requests for cache sim)
├── tools/sim_cache.py  (trace-driven cache sizing)
├── scripts/
│   ├── k3-doctor.sh  (machine check)
│   ├── download-model.sh  (resumable, per-shard verification)
│   └── pack-trunk.sh  (rewrite layers → contiguous file)
├── docs/data/  (all measurements)
```

### Performance Table (same machine, 124 cores)
| RAM | Time/Token | Mechanism |
|-----|-----------|-----------|
| 8 GB | 26.5 s | Full model streams from NVMe each step |
| 32 GB | 24.2 s | Some trunk resident |
| 64 GB | 19.8 s | Larger expert LRU cache |
| 128 GB+ | 5.6 s | Full trunk resident, no disk wait |

### Relevance to This Project
- **KDA**: Implement in HRM recurrent loop for O(1) state memory (long contexts)
- **MXFP4**: Stage 6 compression - quantize HRM state + experts to 4-bit
- **Expert streaming**: If HRM uses MoE reasoning experts, stream from disk at low memory
- **Trunk packing**: HRM checkpoints can use contiguous packing for faster load
- **Attention residuals**: Already implemented in `attention.cpp` (RealFormer style)

---

## Information Path System Design

### What "Information Path" Means
Traceability from question → research → decision → implementation:

```
Question/Hypothesis
    → Research (papers, repos, experiments)
    → Decision Node (ADR or session decision table)
    → Implementation (code files changed)
    → Validation (tests, benchmarks)
    → Knowledge Graph Node (permanent reference)
```

### Path Example: "How to halt HRM efficiently?"
1. **Question**: ACT halting - reward hacking risk?
2. **Research**: PEM QHead (9 features), BudgetController conjunctive halt
3. **Decision**: Conjunctive halt + hard rails (session 2026-09-26, decision #3-4)
4. **Implementation**: `hrm_loop.cpp` eval_qhead_9feat(), false-attractor detection
5. **Validation**: Pending (no tests yet)
6. **Node**: [[Concept - Adaptive Computation Time (ACT)]], [[HRM Recurrent Stepper & ACT]]

---

## Files Created This Session

- `vault/08 - Chat Context & Sessions/00 - Session Index.md`
- `vault/08 - Chat Context & Sessions/Session - 2026-09-26 - Foundation & Research.md`
- `vault/08 - Chat Context & Sessions/Session - 2026-09-26 - Knowledge Graph Expansion.md` (this file)
- `vault/Knowledge Graph/Entities & Concepts/Concept - Attention Residuals (RealFormer & AttnRes).md`
- `vault/Knowledge Graph/Entities & Concepts/Concept - PEM Phasor Equilibrium Machine.md`
- Updated `vault/Knowledge Graph/00 - Knowledge Graph Index.md`

---

## Open Questions

- [ ] Fetch actual kimi-k3-in-c source files (src/attention.c, src/mla.c, src/kda.c) when GitHub fetch available
- [ ] Port KDA to `src/kernels/attention.cpp` as alternative to FlashAttention for long HRM sequences
- [ ] Implement MXFP4 quantization for inference (Stage 6 compression)
- [ ] Experiment: does attention residual improve HRM convergence speed?
- [ ] Run graphify with `--mode deep` once the project owner has put the key in `.env`

---

## Related Nodes

- [[Session - 2026-09-26 - Foundation & Research]]
- [[Concept - Kimi Delta Attention (KDA)]]
- [[Concept - The Four Memory Reductions]]
- [[Concept - Attention Residuals (RealFormer & AttnRes)]]
