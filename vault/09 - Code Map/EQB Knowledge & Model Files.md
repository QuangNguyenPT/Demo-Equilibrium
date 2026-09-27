---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# EQB Knowledge & Model Files

> Product-key quaternion memory, the knowledge.pt archive and the model.hrmc format.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_eqb/hrmc.py` | model.hrmc: the whole trained core in one pickle-free file whose certificate is recomputed on load (F1). | 150 |
| `hrm_eqb/knowledge.py` | knowledge.pt: the model's knowledge as a normal PyTorch archive, rewritten in place on SSD (F2). | 196 |
| `hrm_eqb/memory.py` | Knowledge memory: product-key addressing (M1) over quaternion values (K1), bounded reads (K2). | 77 |

## Depends on

- [[EQB Algebra & Certified Maps]]
- [[EQB Model & Language Coda]]
- [[EQB Task Field & Generators]]

## Used by

- [[EQB Model & Language Coda]]
- [[EQB Runtime Mind]]
- [[EQB Training, Study & Bench]]

## Governed by (specs, ADRs, concepts, guides)

- none

## Changed or discussed in sessions

- [[Session - 2026-09-27 - HRM-EQB Full Build]] — commits `483c0c1`

## Tests

`tests/test_eqb_runtime.py`

## Recent history

- 2026-09-27 483c0c1 HRM-EQB N1-N7: task field, model, file formats, runtime mind, self-study, CLI
