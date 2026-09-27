---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM Verification

> Independent judge: exact replay, determinism, anti-memorization.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm/verify/__init__.py` |  | 4 |
| `hrm/verify/verifier.py` | The independent judge. | 118 |

## Depends on

- [[HRM Model Facade & CLI]]
- [[HRM Perception]]

## Used by

- [[HRM Learning Loops]]
- [[HRM Reasoning]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-003 Unified Cognitive Model]]
- [[HRM-X Specification v0.3 - Understanding by Construction]]

## Changed or discussed in sessions

- [[Session - 2026-09-27 - Benchmark v2 and the Reasoning Ceiling]] — named in the log
- [[Session - 2026-09-27 - Unified Model Build]] — named in the log

## Tests

`tests/test_perception_verify.py`, `tests/test_reasoning.py`, `tests/test_vector3d.py`

## Recent history

- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
