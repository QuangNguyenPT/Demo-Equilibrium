---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM Metacognition

> Self-model (competence), curiosity, intent regression.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm/metacognition/__init__.py` |  | 7 |
| `hrm/metacognition/curiosity.py` | Curiosity: the model's own choice of what to study next. | 45 |
| `hrm/metacognition/intent.py` | Intent: what does the sender want from this input? | 104 |
| `hrm/metacognition/self_model.py` | The self-model: what the model knows it knows, and what it knows it does not. | 95 |

## Depends on

- [[HRM Model Facade & CLI]]
- [[HRM Perception]]

## Used by

- [[HRM Learning Loops]]
- [[HRM Model Facade & CLI]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-003 Unified Cognitive Model]]

## Changed or discussed in sessions

- [[Session - 2026-09-27 - Unified Model Build]] — named in the log

## Tests

`tests/test_language.py`, `tests/test_metacognition.py`

## Recent history

- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
