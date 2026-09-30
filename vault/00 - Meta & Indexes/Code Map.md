---
title: "Code Map — every module to the document that governs it"
type: moc
tags:
  - moc
  - index
  - code-map
  - graph
status: current
date: 2026-09-29
agent: "opencode/stealth/space-bunny-alpha"
model: "space-bunny-alpha"
related:
  - "[[ADR-010 Structural Operator Inference]]"
  - "[[HRM-X Specification v0.5 - Structural Operator Inference]]"
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

Three generations exist. **ADR-008** superseded `hrm_eqb/` and **ADR-010** changed what a
hypothesis is, so `hrm_x/` is now the current model; `hrm_eqb/` is a retained baseline and
`hrm/` is the program-search reasoner it is measured against.

## 0. `hrm_x/` — HRM-X, the current model

Authoritative: [[HRM-X Specification v0.5 - Structural Operator Inference]] ·
[[ADR-010 Structural Operator Inference]] ·
[[ADR-011 Reintroducing EQB Ideas into HRM-X]] (proposed) ·
[[ADR-009 Free Text Is Not Yet Supported]] ·
[[Task Plan]]

### 0a. The v0.5 concept core — structural operator inference

| module | role | evidence |
|---|---|---|
| `hrm_x/qlinalg.py` | exact RREF / rank / solve / row-space membership over ℚ (stdlib only) | `tests/test_x_concept.py` — the determinacy calculus |
| `hrm_x/concepts.py` | 68 law-closed families, exact fit, Prop. 4 weighting, Def. 3 BOTTOM, Thm 3 determinacy, the open-world coverage rule | spec v0.5 §2–§3; `experiments/results/x_v05.json` |
| `hrm_x/domains/concept.py` | the same mathematics as a `Domain`, driven by the existing `Mind.solve` | ADR-010 §4; `tests/test_x_concept.py` |
| `hrm_x/consolidation.py` | corpus-MDL macro admission; the library becomes features | `tests/test_x_concept.py` — a single solved task is **rejected** (negative gain) |
| `hrm_x/situation.py` | `Q^(E×I)` state, verb frames learned one example each, conservation check | `tests/test_x_concept.py` — refuses an untaught frame |
| `hrm_x/algebra.py` | FHRR binding, exact unbind, characters, Prop. 8 cleanup bound | `tests/test_x_concept.py` |
| `hrm_x/operators.py` | `D4 × Sym(colours)` global and object operators, solved not searched | `tests/test_x_concept.py` — Thm 6 composition exact |
| `hrm_x/reader.py` | Place-Value Reader (Thm 5). **Requires torch; not imported by `__init__`** | `tests/test_x_concept.py` (torch-gated) |

### 0b. The shared mind, and the v0.4 route it also serves

| module | role | evidence |
|---|---|---|
| `hrm_x/mind.py` | the loop: operator hypotheses, VOC controller, learned `Patience` λ, persistence. **One mind for both routes.** | spec §14, §19; `tests/test_x_patience.py` |
| `hrm_x/workspace.py` | e-graph with provenance, congruence closure, blame | [[ADR-008 Operator Inference over Structured Situations]] |
| `hrm_x/rules.py` | law rules and saturation | spec v0.4 §8 |
| `hrm_x/induction.py` | Bayesian program induction, observational equivalence (v0.4 route) | spec v0.4 §8 |
| `hrm_x/library.py` | DreamCoder-style MDL **primitive** mining, exact expand, measured search credit (v0.4 route; program level, distinct from `consolidation.py`) | `experiments/x_dsl.json` — compresses correctly, costs more search; off by default |
| `hrm_x/domains/arith.py` | arithmetic concepts by program search, laws, residue verifier (the baseline `x_v05` measures against) | `experiments/results/x_v05.json` |
| `hrm_x/domains/story.py` | learned tagger + exact e-graph story situations, `reading_defect` guard | [[ADR-009 Free Text Is Not Yet Supported]] — closed templates only, not free text |
| `hrm_x/domains/grid.py` | grid object programs, conditions verifier | spec v0.4 §10 |
| `hrm_x/writer.py` | trace → answer + explanation | — |
| `hrm_x/native/` | C++ e-graph and grid kernels behind a C ABI | `experiments/x_native.json` — parity identical, 3.79× / 1.47× |

### 0c. Experiments

| driver | measures | evidence |
|---|---|---|
| `experiments/x_v05.py` | **E13** — the v0.5 concept route vs the v0.4 arith route, same `Mind`, six conditions, two task lists | `experiments/results/x_v05.json` — in-language 13/15 at 593 ops vs 7/15 at 4,265; out-of-language 5 confidently wrong of 6 vs 0 wrong and 4 abstentions |
| `experiments/x_perception.py` | held-out free text, scored by an independent reader | `experiments/results/x_perception.json` |
| `experiments/x_patience.py` | learned λ against a fixed sweep over its own reachable range | `experiments/results/x_patience.json` — the learned price lost; the constant stays |

## 1. `legacy/hrm_eqb/` — HRM-Φ, built and measured, then moved out

> **Moved to `legacy/hrm_eqb/` on 2026-09-29.** An audit found that **zero** files outside its
> own test suite imported it. `legacy/` here means *superseded and unused*; the never-built
> thing in that folder is `legacy/cpp_engine/` (ADR-001/002), which is a different claim.
> `hrm/` (§2) was audited the same way and **stayed**, because `scripts/make_benchmark.py`
> imports `hrm.learning`. The module inventory is now `legacy/README.md`.

Authoritative: [[HRM-EQB Specification v0.2 - Theorems and Proofs]] ·
[[ADR-005 Neural HRM-Phi Replaces Program Search]] ·
[[ADR-011 Reintroducing EQB Ideas into HRM-X]] ·
[[Session - 2026-09-27 - HRM-EQB Theorems]] ·
[[Session - 2026-09-27 - HRM-EQB Full Build]]

The design rule is *every claim a theorem, every theorem a test*, so the module
docstrings cite the spec ID they implement.

| module | implements | spec ref |
|---|---|---|
| `algebra.py` | one state, two algebras: 2k complex planes ≡ k quaternions | §2, T1–T4 |
| `equilibrium.py` | two-level equilibrium core + convergence certificate | §3 |
| `certified.py` | operators with exactly known Lipschitz constants | T5, S1, N1 |
| `gate.py` | self-improvement keep gate, provable false-keep rate | §8, G1 |
| `neuromodulation.py` | four global hormones, projected EMA | §6, H1 |
| `product_keys.py` | exact top-k over S1×S2 from two top-k searches | M1 |
| `memory.py` | product-key addressing over quaternion values, bounded reads | M1, K1, K2 |
| `knowledge.py` | `knowledge.pt` archive, rewritten in place | F2 |
| `hrmc.py` | the whole trained core, certificate recomputed on load | F1 |
| `attnres.py` | attention residuals over a chain of states | A1 |
| `field.py` | task field: content/row/col/slot tokenisation | §1 |
| `coda.py` | language coda: thinking chain → byte-level decoder | [[HRM-Phi Design Document v0.1]] |
| `model.py` | prelude → certified equilibrium segments → heads + coda | §0 |
| `mind.py` | the running model: core + knowledge + hormones | — |
| `train.py` | deep-supervised loss over segments | — |
| `tasks.py` | generated task families with exact answers | — |
| `bench.py` | benchmark suites incl. ARC-AGI-1 | — |
| `device.py` | device selection with a hard VRAM cap | [[GPU Safety - VRAM Oversubscription and DPC Watchdog]] |
| `config.py`, `curiosity.py`, `study.py` | configuration, curiosity, study loop | — |

Its 55 tests are in `legacy/tests/` and are **not** collected by `python -m pytest`
(`testpaths = ["tests"]`). Run them with `$env:PYTHONPATH = "legacy"; python -m pytest
legacy/tests -q`.

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
