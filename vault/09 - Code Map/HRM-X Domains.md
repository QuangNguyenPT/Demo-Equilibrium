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
| `hrm_x/domains/arith.py` | Arithmetic: numbers are exact rationals; an unknown operator symbol is a concept to identify. | 813 |
| `hrm_x/domains/grid.py` | Grids: pixels -> objects -> an object-level program induced from the demos -> exact rendering. | 317 |
| `hrm_x/domains/story.py` | Stories: sentences -> a situation that changes over time -> answers derived, not guessed. | 394 |

## Depends on

- [[HRM-X Mind & Induction]]
- [[HRM-X Native Backend]]
- [[HRM-X Workspace & Rules]]

## Used by

- [[HRM-X Experiments]]

## Governed by (specs, ADRs, concepts, guides)

- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]

## Changed or discussed in sessions

- none recorded

## Tests

`tests/test_x_domains.py`, `tests/test_x_mind.py`, `tests/test_x_native.py`

## Recent history

- none
