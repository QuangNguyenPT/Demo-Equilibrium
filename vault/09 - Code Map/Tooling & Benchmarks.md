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
| `scripts/fetch_benchmarks.py` | Download ARC-AGI-1 and Sudoku-Extreme (test split) into data/external/ (gitignored). | 41 |
| `scripts/knowledge_map.py` | Curated Obsidian code map: one note per subsystem, linked to the notes that govern and changed it. | 405 |
| `scripts/make_benchmark.py` | Build a frozen benchmark (run once, keep the output). | 62 |
| `scripts/simplify_graph.py` | Prune graphify's raw graph down to a human-reviewable graph. | 180 |
| `scripts/update_knowledge_graph.ps1` | Update knowledge graph after modifying the project. | 129 |

## Depends on

- [[HRM Learning Loops]]

## Used by

- none

## Governed by (specs, ADRs, concepts, guides)

- none

## Changed or discussed in sessions

- [[Session - 2026-09-26 - Foundation & Research]] — named in the log
- [[Session - 2026-09-27 - Benchmark v2 and the Reasoning Ceiling]] — named in the log
- [[Session - 2026-09-27 - Graph Rebuild and Benchmarks]] — commits `6e892b3`
- [[Session - 2026-09-27 - HRM-EQB Full Build]] — commits `483c0c1`
- [[Session - 2026-09-27 - Understanding by Construction]] — commits `6b28b26`

## Tests

none

## Recent history

- 2026-09-27 6b28b26 HRM-X design (ADR-007, proposed) + curated code map replaces graph blobs
- 2026-09-27 6e892b3 Graph rebuild: deep semantic extraction, benchmarks complete (ADR-006)
- 2026-09-27 483c0c1 HRM-EQB N1-N7: task field, model, file formats, runtime mind, self-study, CLI
- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
