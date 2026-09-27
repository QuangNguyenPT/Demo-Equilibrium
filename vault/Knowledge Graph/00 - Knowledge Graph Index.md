---
title: Conceptual Knowledge Graph Index
type: knowledge-graph
tags:
  - knowledge-graph
  - ontology
  - concepts
  - information
status: active
created: 2026-09-26
---

# 🌎 Conceptual Knowledge Graph: Information, Systems & Reasoning

This Knowledge Graph captures the **theoretical foundations, conceptual relationships, mathematical principles, and systems mechanisms** underlying the hybrid architecture, drawing deep insights from:
1. **Hierarchical Reasoning Model (HRM)** (Wang et al., 2025; sapientinc/HRM)
2. **Kimi K3 & Extreme Systems Inference in C** (Fareed Khan, 2026, `kimi-k3-in-c`)
3. **Agentic Knowledge Graphs & Non-Parametric Grounding** (Fareed Khan, 2026)

---

## 🗺️ Visual Knowledge Graph (Obsidian Canvas)

> [!TIP]
> Open the hierarchical spatial graph in Obsidian:
> **[[vault/Knowledge Graph/System Knowledge Graph.canvas|Open System Knowledge Graph Canvas]]**
> It maps out all entity relationships, mitigation edges, and architectural cascades on a 2D zoomable plane.

---

## 🔭 Entity & Concept Nodes

### 1. Architectural Paradigms & Attention Mechanics
- [[Concept - Kimi Delta Attention (KDA)]] — Recurrent linear attention, $O(1)$ constant memory footprint, depthwise causal conv, bounded state decay.
- [[Concept - Multi-Head Latent Attention (MLA)]] — Low-rank KV projection into a single latent vector (512-dim), NoPE positional embedding.
- [[Concept - Block Sparse Mixture of Experts (MoE)]] — 896 routed experts, top-16 active per token, 2 residual shared experts, routing bias vs unbiased combination.
- [[Concept - Hierarchical Multi-Timescale Reasoning (HRM)]] — Nested recurrence between Low-Level Executor ($z_L$) and High-Level Planner ($z_H$).
- [[Concept - Adaptive Computation Time (ACT)]] — $Q$-learning dynamic halting mechanism, early exit confidence, compute allocation.

### 2. Systems, Storage & Precision Engineering
- [[Concept - The Four Memory Reductions]] — the mathematical cascade taking a 2.78T model from $5,560\text{ GB} \to 8.24\text{ GB}$ (675× reduction).
- [[Concept - Microscaling MXFP4 & E8M0 Quantization]] — Sub-byte weight representation ($0.53125$ bytes/param) and zero-copy unpack.
- [[Concept - Direct IO & NVMe Streaming]] — Why `O_DIRECT` beats operating system page caches, turning memory floors into dollars.
- [[Concept - Bit-Identical Floating Point Contracts]] — the role of `-ffp-contract=off` in cross-hardware and cross-compiler determinism.

### 3. Agentic Knowledge & Grounding
- [[Concept - Agentic Knowledge Graphs & Grounding]] — Structuring millions of entities into verified graphs, longest-match lookup, deterministic traversal without LLM hallucination.

### 4. Attention Residuals & Cross-Layer Communication
- [[Concept - Attention Residuals (RealFormer & AttnRes)]] — Pre-softmax score passthrough between layers (RealFormer), softmax over preceding layer outputs (AttnRes/Kimi Linear), dual residuals (ResiDual), associative memory (RMT).

### 5. VSA & Non-Parametric Reasoning
- [[Concept - PEM Phasor Equilibrium Machine]] — FHRR phasor algebra, Resonator (L-module), Birkhoff mixing (H-module), 9-feature QHead, conjunctive halting, annealed reshare. PEM = VSA-native HRM.

---

## 🔗 Ontological Relationships

- [[Semantic Relationship Matrix]] — Formal RDF/OWL-style triples connecting all conceptual nodes (`ENABLES`, `MITIGATES`, `COMPRESSES`, `DISPATCHES`, `GROUNDS`).
- [[Information Flow & Reasoning Graph]] — Step-by-step trace of knowledge from input query to reasoning state and output generation.

---

## 💬 Chat Session Logs (Information Path)

Session logs capture decisions, research, file changes, and open questions per session.
These are knowledge nodes — graphify indexes them semantically in deep mode, using the project-scoped key in `.env` (see `AGENTS.md`).

- [[Session - 2026-09-26 - Foundation & Research]] — CLAUDE.md creation, PEM analysis, Kimi/DeepSeek research, PEM↔HRM integration, attention residuals implementation
- [[Session - 2026-09-26 - Knowledge Graph Expansion]] — KG expanded to chat context, Kimi K3 deep architecture research, information path system design

**Index**: [[vault/08 - Chat Context & Sessions/00 - Session Index|Session Index]] — full registry + decision chain
