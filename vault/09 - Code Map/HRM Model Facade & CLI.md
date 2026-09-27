---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM Model Facade & CLI

> Bundle directory, core weights, config, traces and the CLI.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm/__init__.py` | Unified cognitive model: reasoning (PEM-derived) + language + SSD knowledge + metacognition. | 7 |
| `hrm/__main__.py` | Command line for a model bundle. | 161 |
| `hrm/config.py` | Model configuration: the single source of truth for every tunable. | 271 |
| `hrm/core.py` | The core's weights: everything learned that lives in RAM. | 53 |
| `hrm/model.py` | The model: one bundle directory on SSD, loaded section by section. | 363 |
| `hrm/trace.py` | Reasoning trace: the legible record of how an answer was (or was not) reached. | 44 |

## Depends on

- [[HRM Knowledge Store]]
- [[HRM Language]]
- [[HRM Learning Loops]]
- [[HRM Metacognition]]
- [[HRM Perception]]
- [[HRM Reasoning]]
- [[HRM Substrate (FHRR)]]

## Used by

- [[HRM Knowledge Store]]
- [[HRM Language]]
- [[HRM Learning Loops]]
- [[HRM Metacognition]]
- [[HRM Perception]]
- [[HRM Reasoning]]
- [[HRM Substrate (FHRR)]]
- [[HRM Verification]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-003 Unified Cognitive Model]]

## Changed or discussed in sessions

- [[Session - 2026-09-27 - Unified Model Build]] — named in the log

## Tests

`tests/test_cli.py`, `tests/test_config.py`, `tests/test_knowledge_store.py`, `tests/test_language.py`, `tests/test_metacognition.py`, `tests/test_model_online.py`, `tests/test_perception_verify.py`, `tests/test_reasoning.py`, `tests/test_sleep.py`, `tests/test_substrate.py`, `tests/test_vector3d.py`

## Recent history

- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
