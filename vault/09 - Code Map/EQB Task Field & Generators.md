---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# EQB Task Field & Generators

> Tasks as token fields; generated families, ARC and Sudoku loaders, augmentation group.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_eqb/field.py` | The task field: every task becomes one set of tokens with (content, row, col, slot). | 251 |
| `hrm_eqb/tasks.py` | Tasks the model learns from and is judged on. | 362 |

## Depends on

- none

## Used by

- [[EQB Knowledge & Model Files]]
- [[EQB Model & Language Coda]]
- [[EQB Runtime Mind]]
- [[EQB Training, Study & Bench]]
- [[HRM-X Experiments]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-012 EQB and HRM-X Audit - Linear Core, Recalled Language, Live BOTTOM]]
- [[HRM-X Specification v0.3 - Understanding by Construction]]

## Changed or discussed in sessions

- [[Session - 2026-09-27 - HRM-EQB Full Build]] — commits `1553230`, `483c0c1`
- [[Session - 2026-09-29 - EQB and HRM-X Audit]] — named in the log

## Tests

`tests/test_eqb_model.py`, `tests/test_eqb_runtime.py`

## Recent history

- 2026-09-27 1553230 HRM-EQB: bounded training memory, checkpoint/resume, demo-implied canvas rule
- 2026-09-27 483c0c1 HRM-EQB N1-N7: task field, model, file formats, runtime mind, self-study, CLI
