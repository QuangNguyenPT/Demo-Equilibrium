---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM-X Workspace & Rules

> E-graph with provenance: congruence closure, conflicts with blame, rule saturation, extraction.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_x/rules.py` | Rules: pattern-matching modulo equality (e-matching) and bounded saturation (the fast, L-level step). | 206 |
| `hrm_x/workspace.py` | Workspace: the situation model, as typed terms in an e-graph where every equality has a reason. | 289 |

## Depends on

- [[HRM-X Mind & Induction]]
- [[HRM-X Native Backend]]

## Used by

- [[HRM-X Domains]]
- [[HRM-X Mind & Induction]]
- [[HRM-X Native Backend]]

## Governed by (specs, ADRs, concepts, guides)

- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]

## Changed or discussed in sessions

- none recorded

## Tests

`tests/test_x_core.py`, `tests/test_x_native.py`

## Recent history

- none
