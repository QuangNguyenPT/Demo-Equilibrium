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
| `hrm_x/__init__.py` | HRM-X v0.6: constructive concept induction on top of v0.5's structural operator inference. | 22 |
| `hrm_x/induction.py` | Induction: hypotheses about an unknown operator, and laws about a known one (spec v0.4 §8). | 121 |
| `hrm_x/mind.py` | Mind: the cognitive loop. It keeps a belief over hypotheses and spends computation only where it pays. | 512 |
| `hrm_x/writer.py` | Writer: the answer and its explanation, rendered from the actual computation (spec v0.4 §8). | 45 |

## Depends on

- [[HRM-X Workspace & Rules]]

## Used by

- [[HRM-X Concept Construction (v0.6)]]
- [[HRM-X Domains]]
- [[HRM-X Experiments]]
- [[HRM-X Workspace & Rules]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-010 Structural Operator Inference]]
- [[ADR-012 EQB and HRM-X Audit - Linear Core, Recalled Language, Live BOTTOM]]
- [[Configuration & Hyperparameters]]
- [[HRM+LLM Fusion Architecture]]
- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]
- [[HRM-X Specification v0.5 - Structural Operator Inference]]
- [[Tensor Flow & Data Pipelines]]
- [[architecture_spec]]

## Changed or discussed in sessions

- [[Session - 2026-09-28 - DSL Growth, Free Text, and the Graph Pipeline]] — named in the log
- [[Session - 2026-09-28 - HRM-X v0.4 PoC and Native Backend]] — commits `d5bacb1`
- [[Session - 2026-09-29 - EQB and HRM-X Audit]] — commits `734792f`
- [[Session - 2026-09-29 - HRM-X v0.5 Structural Induction and Vault Cleanup]] — commits `f0dc2ae`
- [[Session - 2026-09-30 - HRM-X v0.6 Constructive Concept Induction]] — named in the log

## Tests

`tests/test_x_concept.py`, `tests/test_x_construct.py`, `tests/test_x_core.py`, `tests/test_x_domains.py`, `tests/test_x_dsl.py`, `tests/test_x_mind.py`, `tests/test_x_native.py`, `tests/test_x_patience.py`, `tests/test_x_perception.py`

## Recent history

- 2026-09-30 5a40a66 HRM-X v0.6 phase 1: concept construction, concept memory, language interface, native enumerator
- 2026-09-29 734792f Audit: EQB learning path and HRM-X BOTTOM hypothesis (ADR-012)
- 2026-09-29 f0dc2ae HRM-X v0.5: structural operator inference (ADR-010) + vault cleanup
- 2026-09-28 d5bacb1 HRM-X v0.4: operator inference over structured situations + C++ backend (ADR-008)
