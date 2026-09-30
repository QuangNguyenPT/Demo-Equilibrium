---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# Tooling & Benchmarks

> Benchmark fetch and generation, graph and code-map tooling.

## Modules

| module | what it does | lines |
|---|---|---|
| `scripts/build_native.ps1` | Build the HRM-X native backend (C ABI, loaded by hrm_x/native via ctypes). | 30 |
| `scripts/fetch_benchmarks.py` | Download ARC-AGI-1 and Sudoku-Extreme (test split) into data/external/ (gitignored). | 41 |
| `scripts/knowledge_map.py` | Curated Obsidian code map: one note per subsystem, linked to the notes that govern and changed it. | 419 |
| `scripts/make_benchmark.py` | Build a frozen benchmark (run once, keep the output). | 62 |
| `scripts/simplify_graph.py` | Prune graphify's raw graph down to a human-reviewable graph. | 315 |
| `scripts/update_knowledge_graph.ps1` | Update knowledge graph after modifying the project. | 178 |

## Depends on

- [[HRM Learning Loops]]

## Used by

- none

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-008 Operator Inference over Structured Situations]]
- [[Configuration & Hyperparameters]]
- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]

## Changed or discussed in sessions

- [[Session - 2026-09-26 - Foundation & Research]] — named in the log
- [[Session - 2026-09-27 - Benchmark v2 and the Reasoning Ceiling]] — named in the log
- [[Session - 2026-09-27 - Graph Rebuild and Benchmarks]] — commits `6e892b3`
- [[Session - 2026-09-27 - HRM-EQB Full Build]] — commits `483c0c1`
- [[Session - 2026-09-27 - Understanding by Construction]] — commits `6b28b26`
- [[Session - 2026-09-28 - DSL Growth, Free Text, and the Graph Pipeline]] — named in the log
- [[Session - 2026-09-28 - HRM-X v0.4 PoC and Native Backend]] — commits `d5bacb1`
- [[Session - 2026-09-29 - HRM-X v0.5 Structural Induction and Vault Cleanup]] — commits `f0dc2ae`

## Tests

none

## Recent history

- 2026-09-29 f0dc2ae HRM-X v0.5: structural operator inference (ADR-010) + vault cleanup
- 2026-09-28 d5bacb1 HRM-X v0.4: operator inference over structured situations + C++ backend (ADR-008)
- 2026-09-27 6b28b26 HRM-X design (ADR-007, proposed) + curated code map replaces graph blobs
- 2026-09-27 6e892b3 Graph rebuild: deep semantic extraction, benchmarks complete (ADR-006)
- 2026-09-27 483c0c1 HRM-EQB N1-N7: task field, model, file formats, runtime mind, self-study, CLI
- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
