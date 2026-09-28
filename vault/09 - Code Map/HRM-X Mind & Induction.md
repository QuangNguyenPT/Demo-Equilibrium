---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM-X Mind & Induction

> Operator hypotheses as programs, law discovery, the VOC controller, modulators, trace writer.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_x/__init__.py` | HRM-X v0.4: operator inference over structured situations, under value-of-computation control. | 5 |
| `hrm_x/induction.py` | Induction: hypotheses about an unknown operator, and laws about a known one (spec v0.4 §8). | 121 |
| `hrm_x/mind.py` | Mind: the cognitive loop. It keeps a belief over hypotheses and spends computation only where it pays. | 350 |
| `hrm_x/writer.py` | Writer: the answer and its explanation, rendered from the actual computation (spec v0.4 §8). | 45 |

## Depends on

- [[HRM-X Workspace & Rules]]

## Used by

- [[HRM-X Domains]]
- [[HRM-X Experiments]]
- [[HRM-X Workspace & Rules]]

## Governed by (specs, ADRs, concepts, guides)

- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]

## Changed or discussed in sessions

- none recorded

## Tests

`tests/test_x_core.py`, `tests/test_x_domains.py`, `tests/test_x_mind.py`, `tests/test_x_native.py`

## Recent history

- none
