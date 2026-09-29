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
local AST parsing, and markdown needs an LLM backend — the default `claude-cli`
backend needs **no key** and works, but a full semantic pass over the corpus is
slow enough to look like a hang, so routine runs are `-Mode code-only` and the
graph holds 2,933 code nodes and **0** note nodes. This file is the
deterministic bridge: every module, and the document that is authoritative for
it. It is maintained by hand because it is the part that must not depend on a
credential or on an extraction that can hang.

Three generations exist. **ADR-008** superseded `hrm_eqb/`, so `hrm_x/` is now
the current model; `hrm_eqb/` is a retained baseline and `hrm/` is the
program-search reasoner it is measured against.

## 0. `hrm_x/` — HRM-X, the current model

Authoritative: [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]] ·
[[ADR-008 Operator Inference over Structured Situations]] ·
[[ADR-009 Free Text Is Not Yet Supported]] ·
[[Task Plan: verified state and next gates]]

| module | role | evidence |
|---|---|---|
| `hrm_x/mind.py` | the loop: operator hypotheses, VOC controller, learned `Patience` λ, persistence | spec §14, §19; `tests/test_x_patience.py` |
| `hrm_x/workspace.py` | e-graph with provenance, congruence closure, blame | [[ADR-008 Operator Inference over Structured Situations]] |
| `hrm_x/rules.py` | law rules and saturation | spec §8 |
| `hrm_x/induction.py` | Bayesian program induction, observational equivalence | spec §8 |
| `hrm_x/library.py` | DreamCoder-style MDL primitive mining, exact expand, **measured** search credit | `experiments/x_dsl.json` — compresses correctly, costs more search; off by default |
| `hrm_x/domains/arith.py` | arithmetic concepts, held-out search, opt-in library bound | spec §9; `tests/test_x_dsl.py` |
| `hrm_x/domains/story.py` | learned tagger + exact e-graph story situations, `reading_defect` guard | [[ADR-009 Free Text Is Not Yet Supported]] — closed templates only, not free text |
| `hrm_x/domains/grid.py` | grid object programs, conditions verifier | spec §10 |
| `hrm_x/native/` | C++ e-graph and grid kernels behind a C ABI | `experiments/x_native.json` — parity identical, 3.79× / 1.47× |
| `experiments/x_perception.py` | held-out free text, scored by an independent reader | `experiments/results/x_perception.json` |
| `experiments/x_patience.py` | learned λ judged against a fixed sweep over its own reachable range | `experiments/results/x_patience.json` — the learned price lost; the constant stays |

## 1. `hrm_eqb/` — HRM-Φ, the superseded model, kept as a baseline

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
| `scripts/simplify_graph.py` | prunes the raw graph to a readable view; wired into the update script |
| `scripts/knowledge_map.py` | generates the curated `vault/09 - Code Map/` subsystem notes |
| `scripts/update_knowledge_graph.ps1` | the only sanctioned graph update path: update → prune → code map → derived `vault/_graph-view/` |
| `AGENTS.md` | credentials + attribution rules for all agents |

## Related

- [[Master MOC]] · [[00 - Session Index]] · [[00 - Knowledge Graph Index]]
