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
| `hrm_x/__init__.py` | HRM-X v0.5: structural operator inference, under value-of-computation control. | 14 |
| `hrm_x/induction.py` | Induction: hypotheses about an unknown operator, and laws about a known one (spec v0.4 §8). | 121 |
| `hrm_x/mind.py` | Mind: the cognitive loop. It keeps a belief over hypotheses and spends computation only where it pays. | 473 |
| `hrm_x/writer.py` | Writer: the answer and its explanation, rendered from the actual computation (spec v0.4 §8). | 45 |

## Depends on

- [[HRM-X Workspace & Rules]]

## Used by

- [[HRM-X Domains]]
- [[HRM-X Experiments]]
- [[HRM-X Workspace & Rules]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-010 Structural Operator Inference]]
- [[Configuration & Hyperparameters]]
- [[HRM+LLM Fusion Architecture]]
- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]
- [[HRM-X Specification v0.5 - Structural Operator Inference]]
- [[Tensor Flow & Data Pipelines]]

## Changed or discussed in sessions

- [[Session - 2026-09-28 - DSL Growth, Free Text, and the Graph Pipeline]] — named in the log
- [[Session - 2026-09-28 - HRM-X v0.4 PoC and Native Backend]] — commits `d5bacb1`
- [[Session - 2026-09-29 - HRM-X v0.5 Structural Induction and Vault Cleanup]] — commits `f0dc2ae`

## Tests

`tests/test_x_concept.py`, `tests/test_x_core.py`, `tests/test_x_domains.py`, `tests/test_x_dsl.py`, `tests/test_x_mind.py`, `tests/test_x_native.py`, `tests/test_x_patience.py`, `tests/test_x_perception.py`

## Recent history

- 2026-09-29 f0dc2ae HRM-X v0.5: structural operator inference (ADR-010) + vault cleanup
- 2026-09-28 d5bacb1 HRM-X v0.4: operator inference over structured situations + C++ backend (ADR-008)
