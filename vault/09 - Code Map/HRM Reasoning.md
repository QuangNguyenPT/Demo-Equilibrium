---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM Reasoning

> Fast path, H-equilibrium, synthesis search, halting and restarts.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm/reasoning/__init__.py` |  | 13 |
| `hrm/reasoning/controller.py` | Budget controller: ACCEPT / CONTINUE / RESTART / ABSTAIN plus the KM coefficient. | 138 |
| `hrm/reasoning/equilibrium.py` | H-level (deep): the contractive equilibrium operator. | 166 |
| `hrm/reasoning/priors.py` | Consolidated knowledge held in the core (RAM). | 65 |
| `hrm/reasoning/qhead.py` | Q-learning ACT head over controller features, confined to discretionary actions. | 76 |
| `hrm/reasoning/reasoner.py` | The reasoning section: fast recognition, then deep equilibrium, then synthesis. | 230 |
| `hrm/reasoning/resonator.py` | L-level (fast): one resonator tick of soft VSA factorization. | 107 |
| `hrm/reasoning/synthesis.py` | Deepest level: compose new programs when no single hypothesis explains the task. | 57 |

## Depends on

- [[HRM Knowledge Store]]
- [[HRM Language]]
- [[HRM Model Facade & CLI]]
- [[HRM Perception]]
- [[HRM Substrate (FHRR)]]
- [[HRM Verification]]

## Used by

- [[HRM Learning Loops]]
- [[HRM Model Facade & CLI]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-003 Unified Cognitive Model]]
- [[ADR-004 Discriminating Benchmark and Reasoning Ceiling]]

## Changed or discussed in sessions

- [[Session - 2026-09-27 - Benchmark v2 and the Reasoning Ceiling]] — named in the log
- [[Session - 2026-09-27 - Unified Model Build]] — named in the log

## Tests

`tests/test_language.py`, `tests/test_reasoning.py`

## Recent history

- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
