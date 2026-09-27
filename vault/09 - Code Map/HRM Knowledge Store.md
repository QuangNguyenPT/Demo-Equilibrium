---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM Knowledge Store

> SSD store: memmap fp16 keys plus a sqlite journal.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm/knowledge/__init__.py` |  | 7 |
| `hrm/knowledge/store.py` | SSD-resident knowledge: the dynamic part of the model file. | 694 |

## Depends on

- [[HRM Model Facade & CLI]]
- [[HRM Substrate (FHRR)]]

## Used by

- [[HRM Learning Loops]]
- [[HRM Model Facade & CLI]]
- [[HRM Reasoning]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-003 Unified Cognitive Model]]

## Changed or discussed in sessions

- [[Session - 2026-09-27 - Unified Model Build]] — named in the log

## Tests

`tests/test_knowledge_store.py`, `tests/test_reasoning.py`

## Recent history

- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
