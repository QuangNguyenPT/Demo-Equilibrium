---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM-X Concept Construction (v0.6)

> New basis features built from primitives when the written language runs out, exact fitting of their linear closure, concept memory (candidate -> admitted -> retired), the language interface with a learned numeral reader, and the measured-but-unused FHRR signature / mushroom-body router.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_x/__main__.py` | CLI for the HRM-X v0.6 model. | 68 |
| `hrm_x/concept_memory.py` | Concept memory: constructed features that earned a place, stored as reusable operators. | 167 |
| `hrm_x/construct.py` | Concept construction: new basis features built from primitives when the language runs out. | 379 |
| `hrm_x/curriculum.py` | The task field HRM-X v0.6 trains on: operators over rationals, from written to constructible. | 109 |
| `hrm_x/language.py` | Language interface: surface text -> structured evidence -> the mind -> an explained answer. | 206 |
| `hrm_x/mbi.py` | Mushroom-Body Index (MBI): a learned router that orders constructed features, never answers. | 137 |
| `hrm_x/model.py` | HRM-X v0.6 as one trainable, runnable model: config, train, checkpoint, answer, talk, evaluate. | 182 |
| `hrm_x/signature.py` | FHRR evidence signatures: a set of (x, y) -> z observations as one relational phasor vector. | 94 |

## Depends on

- [[HRM-X Concept Induction (v0.5)]]
- [[HRM-X Domains]]
- [[HRM-X Experiments]]
- [[HRM-X Mind & Induction]]
- [[HRM-X Native Backend]]

## Used by

- [[HRM-X Domains]]
- [[HRM-X Experiments]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-013 Constructive Concept Induction]]
- [[architecture_spec]]
- [[evaluation]]
- [[research_plan]]

## Changed or discussed in sessions

- [[Session - 2026-09-30 - HRM-X v0.6 Constructive Concept Induction]] — named in the log

## Tests

`tests/test_x_construct.py`, `tests/test_x_native.py`

## Recent history

- 2026-09-30 5a40a66 HRM-X v0.6 phase 1: concept construction, concept memory, language interface, native enumerator
