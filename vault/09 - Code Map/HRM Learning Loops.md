---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM Learning Loops

> Online learner, generators, frozen harness, sleep cycle, self-study.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm/learning/__init__.py` |  | 11 |
| `hrm/learning/generators.py` | Task generators: practice material with ground truth, per domain and family. | 317 |
| `hrm/learning/harness.py` | The frozen judge of the learning loop: a fixed benchmark plus a deterministic scorer. | 237 |
| `hrm/learning/online.py` | Online learning (loop A): runs after every interaction. | 117 |
| `hrm/learning/sleep.py` | Sleep cycle (loop B): consolidate proven knowledge into the core, behind a frozen gate. | 119 |
| `hrm/learning/study.py` | Self-study (loop B driver): the model studies its own knowledge gaps. | 107 |

## Depends on

- [[HRM Knowledge Store]]
- [[HRM Metacognition]]
- [[HRM Model Facade & CLI]]
- [[HRM Perception]]
- [[HRM Reasoning]]
- [[HRM Verification]]

## Used by

- [[HRM Model Facade & CLI]]
- [[Tooling & Benchmarks]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-003 Unified Cognitive Model]]
- [[HRM-Phi Design Document v0.1]]
- [[HRM-X Specification v0.3 - Understanding by Construction]]

## Changed or discussed in sessions

- [[Session - 2026-09-27 - Benchmark v2 and the Reasoning Ceiling]] — named in the log
- [[Session - 2026-09-27 - Unified Model Build]] — named in the log

## Tests

`tests/test_cli.py`, `tests/test_metacognition.py`, `tests/test_sleep.py`, `tests/test_vector3d.py`

## Recent history

- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
