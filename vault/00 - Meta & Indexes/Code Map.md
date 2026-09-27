---
title: "Code Map — every module to the document that governs it"
type: moc
tags:
  - moc
  - index
  - code-map
  - graph
status: current
date: 2026-09-27
agent: "opencode/big-pickle"
model: "big-pickle"
related:
  - "[[ADR-005 Neural HRM-Phi Replaces Program Search]]"
  - "[[HRM-EQB Specification v0.2 - Theorems and Proofs]]"
  - "[[ADR-004 Discriminating Benchmark and Reasoning Ceiling]]"
---

# Code Map

The knowledge graph cannot do this job on its own. `graphify` indexes code with
local AST parsing, but **markdown requires an LLM key** — so with no key the
graph contains 1,665 code nodes and **0** note nodes, and the vault looks
unconnected to the code no matter how the graph is drawn. This file is the
deterministic bridge: every module, and the document that is authoritative for
it. It is maintained by hand because it is the part that must not depend on a
credential.

Two live architectures exist in parallel. **ADR-005** replaced program search
with a neural design, so `hrm_eqb/` is now the target and `hrm/`'s program-search
reasoner is the baseline being measured against.

## 1. `hrm_eqb/` — HRM-Φ, the current model

Authoritative: [[HRM-EQB Specification v0.2 - Theorems and Proofs]] ·
[[ADR-005 Neural HRM-Phi Replaces Program Search]] ·
[[Session - 2026-09-27 - HRM-EQB Theorems]] ·
[[Session - 2026-09-27 - HRM-EQB Full Build]]

The design rule is *every claim a theorem, every theorem a test*, so the module
docstrings cite the spec ID they implement.

| module | implements | spec ref |
|---|---|---|
| `hrm_eqb/algebra.py` | one state, two algebras: 2k complex planes ≡ k quaternions | §2, T1–T4 |
| `hrm_eqb/equilibrium.py` | two-level equilibrium core + convergence certificate | §3 |
| `hrm_eqb/certified.py` | operators with exactly known Lipschitz constants | T5, S1, N1 |
| `hrm_eqb/gate.py` | self-improvement keep gate, provable false-keep rate | §8, G1 |
| `hrm_eqb/neuromodulation.py` | four global hormones, projected EMA | §6, H1 |
| `hrm_eqb/product_keys.py` | exact top-k over S1×S2 from two top-k searches | M1 |
| `hrm_eqb/memory.py` | product-key addressing over quaternion values, bounded reads | M1, K1, K2 |
| `hrm_eqb/knowledge.py` | `knowledge.pt` archive, rewritten in place | F2 |
| `hrm_eqb/hrmc.py` | the whole trained core, certificate recomputed on load | F1 |
| `hrm_eqb/attnres.py` | attention residuals over a chain of states | A1 |
| `hrm_eqb/field.py` | task field: content/row/col/slot tokenisation | §1 |
| `hrm_eqb/coda.py` | language coda: thinking chain → byte-level decoder | [[HRM-Phi Design Document v0.1]] |
| `hrm_eqb/model.py` | prelude → certified equilibrium segments → heads + coda | §0 |
| `hrm_eqb/mind.py` | the running model: core + knowledge + hormones | — |
| `hrm_eqb/train.py` | deep-supervised loss over segments | — |
| `hrm_eqb/tasks.py` | generated task families with exact answers | — |
| `hrm_eqb/bench.py` | benchmark suites incl. ARC-AGI-1 | — |
| `hrm_eqb/device.py` | device selection with a hard VRAM cap | [[GPU Safety - VRAM Oversubscription and DPC Watchdog]] |
| `hrm_eqb/config.py`, `curiosity.py`, `study.py` | configuration, curiosity, study loop | — |

## 2. `hrm/` — the program-search baseline

Authoritative: [[ADR-003 Unified Cognitive Model]] ·
[[ADR-004 Discriminating Benchmark and Reasoning Ceiling]] ·
[[Why v2 Did Not Prove the Model]] ·
[[Session - 2026-09-27 - Benchmark v2 and the Reasoning Ceiling]]

| module | role | evidence |
|---|---|---|
| `hrm/reasoning/reasoner.py` | fast / H-engine / synthesis routing; exactness gate | [[Why v2 Did Not Prove the Model]] |
| `hrm/reasoning/equilibrium.py` | H-refresh over `refresh_top_k=6` | same — 0 accepts in 108/108 |
| `hrm/reasoning/synthesis.py` | beam search over the single-op candidates | same — carries 61/84 solves |
| `hrm/reasoning/resonator.py`, `qhead.py`, `priors.py` | z_h perturbation, halting, priors | [[Concept - PEM Phasor Equilibrium Machine]] |
| `hrm/reasoning/controller.py` | budget/restart control | [[Session - 2026-09-26 - Foundation & Research]] |
| `hrm/verify/verifier.py` | `energy` oracle + complexity cap | ADR-004 |
| `hrm/perception/vector3d.py` | 3-D domain, vocabulary, verified depth | ADR-004 |
| `hrm/perception/grid/*`, `sequence.py` | grid + sequence domains | ADR-004 |
| `hrm/perception/base.py` | domain protocol, complexity/redundancy policy | ADR-004 |
| `hrm/learning/harness.py` | scorer, per-depth accounting, harness lock | ADR-004 |
| `hrm/learning/generators.py` | verified task generation | ADR-004 |
| `hrm/learning/sleep.py`, `study.py`, `online.py` | the only writers of knowledge | ADR-003 |
| `hrm/knowledge/store.py` | journal store, torn-write recovery | ADR-003 |
| `hrm/substrate/fhrr.py`, `codebook.py`, `pool.py`, `quantize.py` | FHRR phasor substrate | [[Concept - PEM Phasor Equilibrium Machine]] |
| `hrm/model.py`, `core.py`, `config.py`, `trace.py` | wiring, config, traces | [[Configuration & Hyperparameters]] |
| `hrm/metacognition/*` | curiosity, intent, self-model | ADR-003 |

## 3. `hrm/language/` — native language section

Authoritative: [[HRM-Phi Design Document v0.1]] · [[ADR-005 Neural HRM-Phi Replaces Program Search]]

| module | role |
|---|---|
| `hrm/language/base.py` | the language section: native ability, not a bolted-on translator |
| `hrm/language/text.py` | text task notation: words → structure |
| `hrm/language/speech.py` | inner speech while reasoning |
| `hrm/language/backends.py` | swappable backends (`HashingLanguage` is dependency-free) |

## 4. Tooling

| path | governs |
|---|---|
| `scripts/make_benchmark.py` | frozen benchmark generation (refuses overwrite) |
| `scripts/simplify_graph.py` | prunes the raw graph to a readable view |
| `scripts/update_knowledge_graph.ps1` | the only sanctioned graph update path |
| `AGENTS.md` | credentials + attribution rules for all agents |

## Related

- [[Master MOC]] · [[00 - Session Index]] · [[00 - Knowledge Graph Index]]
