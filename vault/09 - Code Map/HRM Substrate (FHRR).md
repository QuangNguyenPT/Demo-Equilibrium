---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM Substrate (FHRR)

> Phasor algebra, name-derived codebook, backend validation gate.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm/substrate/__init__.py` |  | 7 |
| `hrm/substrate/backend.py` | Kernel backend dispatch. | 117 |
| `hrm/substrate/codebook.py` | Deterministic symbol store and cleanup memory. | 87 |
| `hrm/substrate/fhrr.py` | FHRR phasor algebra. | 123 |
| `hrm/substrate/pool.py` | Byte-bounded caches for the substrate. | 43 |
| `hrm/substrate/quantize.py` | Phase quantization: unit phasors stored as uint16 angle codes. | 30 |

## Depends on

- [[HRM Model Facade & CLI]]

## Used by

- [[HRM Knowledge Store]]
- [[HRM Model Facade & CLI]]
- [[HRM Perception]]
- [[HRM Reasoning]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-003 Unified Cognitive Model]]
- [[HRM-X Specification v0.3 - Understanding by Construction]]

## Changed or discussed in sessions

- [[Session - 2026-09-27 - Unified Model Build]] — named in the log

## Tests

`tests/test_knowledge_store.py`, `tests/test_language.py`, `tests/test_perception_verify.py`, `tests/test_reasoning.py`, `tests/test_substrate.py`, `tests/test_vector3d.py`

## Recent history

- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
