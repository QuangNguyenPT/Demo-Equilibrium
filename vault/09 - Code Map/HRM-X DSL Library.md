---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM-X DSL Library

> Structural MDL primitive mining, exact abstraction/expansion, and the measured credit a library earns in search.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_x/library.py` | Learned primitives: the DSL grows by compressing the programs that are already known. | 407 |

## Depends on

- none

## Used by

- [[HRM-X Domains]]
- [[HRM-X Experiments]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-010 Structural Operator Inference]]
- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]
- [[HRM-X Specification v0.5 - Structural Operator Inference]]

## Changed or discussed in sessions

- [[Session - 2026-09-28 - DSL Growth, Free Text, and the Graph Pipeline]] — named in the log
- [[Session - 2026-09-29 - HRM-X v0.5 Structural Induction and Vault Cleanup]] — commits `f0dc2ae`

## Tests

`tests/test_x_dsl.py`, `tests/test_x_library.py`

## Recent history

- 2026-09-29 f0dc2ae HRM-X v0.5: structural operator inference (ADR-010) + vault cleanup
