---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM-X Concept Induction (v0.5)

> Law-closed hypothesis families fitted exactly over Q, the determinacy test of Theorem 3, the open-world coverage rule and BOTTOM, corpus-MDL consolidation, phasor binding, and the group-theoretic grid and place-value layers.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_x/algebra.py` | Phasor (FHRR) binding algebra and character codes (doc 03, sections 4 and 10). | 99 |
| `hrm_x/concepts.py` | Typed concept induction for binary operators (doc 03, sections 1-2). | 413 |
| `hrm_x/consolidation.py` | Library growth by MDL gain (doc 03, section 7): verified concepts become new basis features. | 109 |
| `hrm_x/operators.py` | Group-action operators on grids (doc 03, section 4). | 262 |
| `hrm_x/qlinalg.py` | Exact linear algebra over Q (fractions.Fraction). Reference implementation; sizes here are tiny. | 89 |
| `hrm_x/reader.py` | Place-Value Reader (PVR): a learned layer numeral string -> exact rational (doc 03, section 3). | 253 |
| `hrm_x/situation.py` | Situation model: quantities as a state vector, events as group actions learned one-shot (doc 03, s.1, 4). | 165 |

## Depends on

- [[HRM-X Domains]]

## Used by

- [[HRM-X Concept Construction (v0.6)]]
- [[HRM-X Domains]]
- [[HRM-X Experiments]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-010 Structural Operator Inference]]
- [[ADR-012 EQB and HRM-X Audit - Linear Core, Recalled Language, Live BOTTOM]]
- [[HRM-X Specification v0.5 - Structural Operator Inference]]
- [[architecture_spec]]
- [[research_plan]]

## Changed or discussed in sessions

- [[Session - 2026-09-29 - EQB and HRM-X Audit]] — commits `734792f`
- [[Session - 2026-09-29 - HRM-X v0.5 Structural Induction and Vault Cleanup]] — commits `f0dc2ae`
- [[Session - 2026-09-30 - HRM-X v0.6 Constructive Concept Induction]] — named in the log

## Tests

`tests/test_x_concept.py`, `tests/test_x_construct.py`

## Recent history

- 2026-09-29 734792f Audit: EQB learning path and HRM-X BOTTOM hypothesis (ADR-012)
- 2026-09-29 f0dc2ae HRM-X v0.5: structural operator inference (ADR-010) + vault cleanup
