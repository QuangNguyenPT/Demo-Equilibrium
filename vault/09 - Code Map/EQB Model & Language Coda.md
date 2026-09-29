---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# EQB Model & Language Coda

> Prelude transformer -> certified segments -> answer and halt heads; byte-level coda.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_eqb/__init__.py` | HRM-EQB: hierarchical reasoning by certified equilibrium (spec v0.2, torch). | 27 |
| `hrm_eqb/coda.py` | Language coda: a semi-embedder from the thinking chain into a byte-level transformer decoder. | 88 |
| `hrm_eqb/config.py` | Architecture of one HRM-EQB model: a frozen dataclass with a strict JSON round-trip. | 77 |
| `hrm_eqb/model.py` | HRM-EQB: prelude -> segments of certified equilibrium -> structured head, halting head, coda. | 180 |

## Depends on

- [[EQB Algebra & Certified Maps]]
- [[EQB Equilibrium Core]]
- [[EQB Knowledge & Model Files]]
- [[EQB Runtime Mind]]
- [[EQB Task Field & Generators]]

## Used by

- [[EQB Knowledge & Model Files]]
- [[EQB Runtime Mind]]
- [[EQB Training, Study & Bench]]

## Governed by (specs, ADRs, concepts, guides)

- [[Configuration & Hyperparameters]]
- [[Development Workflow & Conventions]]
- [[Hybrid System Philosophy]]
- [[Model Blueprint & Graph Layout]]

## Changed or discussed in sessions

- [[Session - 2026-09-27 - HRM-EQB Full Build]] — commits `1553230`, `483c0c1`

## Tests

`tests/test_eqb_model.py`, `tests/test_eqb_runtime.py`, `tests/test_eqb_theorems.py`

## Recent history

- 2026-09-27 1553230 HRM-EQB: bounded training memory, checkpoint/resume, demo-implied canvas rule
- 2026-09-27 483c0c1 HRM-EQB N1-N7: task field, model, file formats, runtime mind, self-study, CLI
- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
