---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM Language

> Language section backends, inner speech, text notation.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm/language/__init__.py` |  | 24 |
| `hrm/language/backends.py` | Language backends. | 119 |
| `hrm/language/base.py` | The language section: the model's native language ability. | 40 |
| `hrm/language/speech.py` | Inner speech: the model thinking in language while it reasons. | 80 |
| `hrm/language/text.py` | Task notation for text: how a task written in words becomes structure. | 82 |

## Depends on

- [[HRM Model Facade & CLI]]
- [[HRM Perception]]

## Used by

- [[HRM Model Facade & CLI]]
- [[HRM Reasoning]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-003 Unified Cognitive Model]]

## Changed or discussed in sessions

- [[Session - 2026-09-27 - Unified Model Build]] — named in the log

## Tests

`tests/test_language.py`

## Recent history

- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
