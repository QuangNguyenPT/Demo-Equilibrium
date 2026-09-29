---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# EQB Equilibrium Core

> Two-level fixed-point solver whose convergence certificate is computed from its weights.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_eqb/attnres.py` | Attention Residuals (arXiv:2603.15031) over a chain of states: softmax over depth (spec A1). | 27 |
| `hrm_eqb/equilibrium.py` | Two-level equilibrium core with a convergence certificate computed from its weights (spec §3). | 199 |

## Depends on

- [[EQB Algebra & Certified Maps]]

## Used by

- [[EQB Model & Language Coda]]
- [[HRM-X Experiments]]

## Governed by (specs, ADRs, concepts, guides)

- [[Concept - Attention Residuals (RealFormer & AttnRes)]]

## Changed or discussed in sessions

- [[Session - 2026-09-27 - HRM-EQB Full Build]] — commits `483c0c1`
- [[Session - 2026-09-27 - HRM-EQB Theorems]] — named in the log

## Tests

`tests/test_eqb_model.py`, `tests/test_eqb_theorems.py`

## Recent history

- 2026-09-27 483c0c1 HRM-EQB N1-N7: task field, model, file formats, runtime mind, self-study, CLI
- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
