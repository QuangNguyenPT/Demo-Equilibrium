---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM-X Domains

> Arithmetic concepts, word-problem stories, grid object programs; their verifiers.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_x/domains/__init__.py` | Domains: each supplies perception, an operator language, L-level derivation and an independent verifier. | 2 |
| `hrm_x/domains/arith.py` | Arithmetic: numbers are exact rationals; an unknown operator symbol is a concept to identify. | 938 |
| `hrm_x/domains/concept.py` | The v0.5 concept route: structural operator inference behind the v0.4 mind. | 365 |
| `hrm_x/domains/grid.py` | Grids: pixels -> objects -> an object-level program induced from the demos -> exact rendering. | 317 |
| `hrm_x/domains/story.py` | Stories: sentences -> a situation that changes over time -> answers derived, not guessed. | 443 |

## Depends on

- [[HRM-X Concept Induction (v0.5)]]
- [[HRM-X DSL Library]]
- [[HRM-X Mind & Induction]]
- [[HRM-X Native Backend]]
- [[HRM-X Workspace & Rules]]

## Used by

- [[HRM-X Concept Induction (v0.5)]]
- [[HRM-X Experiments]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-009 Free Text Is Not Yet Supported]]
- [[ADR-010 Structural Operator Inference]]
- [[HRM+LLM Fusion Architecture]]
- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]
- [[HRM-X Specification v0.5 - Structural Operator Inference]]
- [[Model Blueprint & Graph Layout]]
- [[Tensor Flow & Data Pipelines]]

## Changed or discussed in sessions

- [[Session - 2026-09-28 - DSL Growth, Free Text, and the Graph Pipeline]] — named in the log
- [[Session - 2026-09-28 - HRM-X v0.4 PoC and Native Backend]] — commits `d5bacb1`
- [[Session - 2026-09-29 - HRM-X v0.5 Structural Induction and Vault Cleanup]] — named in the log

## Tests

`tests/test_x_concept.py`, `tests/test_x_domains.py`, `tests/test_x_dsl.py`, `tests/test_x_library.py`, `tests/test_x_mind.py`, `tests/test_x_native.py`, `tests/test_x_patience.py`, `tests/test_x_perception.py`

## Recent history

- 2026-09-28 d5bacb1 HRM-X v0.4: operator inference over structured situations + C++ backend (ADR-008)
