---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# EQB Runtime Mind

> The running model: answer-or-question, verified writes, curiosity, hormones, keep gate.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_eqb/curiosity.py` | Curiosity: the most informative question is where the committee disagrees most (spec §7, Q1). | 41 |
| `hrm_eqb/gate.py` | Self-improvement keep gate with a provable false-keep rate (spec §8, G1). | 20 |
| `hrm_eqb/mind.py` | The running model: core (.hrmc) + knowledge (knowledge.pt) + hormones, answering or asking. | 292 |
| `hrm_eqb/neuromodulation.py` | Hormone system: four global neuromodulators in a box, updated by a projected EMA (spec §6, H1). | 85 |

## Depends on

- [[EQB Knowledge & Model Files]]
- [[EQB Model & Language Coda]]
- [[EQB Task Field & Generators]]
- [[EQB Training, Study & Bench]]

## Used by

- [[EQB Model & Language Coda]]
- [[EQB Training, Study & Bench]]
- [[HRM-X Experiments]]

## Governed by (specs, ADRs, concepts, guides)

- none

## Changed or discussed in sessions

- [[Session - 2026-09-27 - HRM-EQB Full Build]] — commits `483c0c1`
- [[Session - 2026-09-27 - HRM-EQB Theorems]] — named in the log

## Tests

`tests/test_eqb_runtime.py`

## Recent history

- 2026-09-27 483c0c1 HRM-EQB N1-N7: task field, model, file formats, runtime mind, self-study, CLI
- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
