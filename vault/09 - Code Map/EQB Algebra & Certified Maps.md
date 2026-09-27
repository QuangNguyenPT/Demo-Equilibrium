---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# EQB Algebra & Certified Maps

> Rotation-vector text, quaternion knowledge, operators with exactly known Lipschitz constants.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_eqb/algebra.py` | One state, two algebras: 2k complex planes (text, positions) = k quaternions (knowledge). | 79 |
| `hrm_eqb/certified.py` | Operators whose Lipschitz constants are known exactly from their parameters (T5, S1, N1). | 66 |
| `hrm_eqb/product_keys.py` | Product-key retrieval: exact top-k over S1 x S2 slots from two top-k searches (spec M1). | 20 |

## Depends on

- none

## Used by

- [[EQB Equilibrium Core]]
- [[EQB Knowledge & Model Files]]
- [[EQB Model & Language Coda]]

## Governed by (specs, ADRs, concepts, guides)

- none

## Changed or discussed in sessions

- [[Session - 2026-09-27 - HRM-EQB Full Build]] — commits `483c0c1`
- [[Session - 2026-09-27 - HRM-EQB Theorems]] — named in the log

## Tests

`tests/test_eqb_model.py`

## Recent history

- 2026-09-27 483c0c1 HRM-EQB N1-N7: task field, model, file formats, runtime mind, self-study, CLI
- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
