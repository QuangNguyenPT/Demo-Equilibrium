---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# EQB Training, Study & Bench

> Deep-supervised training, keep/discard self-study, benchmark suites, VRAM-capped device, CLI.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_eqb/__main__.py` | CLI: python -m hrm_eqb create \| info \| ask \| learn \| study \| bench \| seal. | 194 |
| `hrm_eqb/bench.py` | Benchmarks: generated families, language (text answers + coda descriptions), Sudoku, ARC-AGI-1. | 168 |
| `hrm_eqb/device.py` | Device selection with a hard VRAM cap. | 20 |
| `hrm_eqb/study.py` | Self-study, influenced by karpathy/autoresearch: fixed budget, one frozen metric, keep or discard. | 134 |
| `hrm_eqb/train.py` | Training and evaluation. | 234 |

## Depends on

- [[EQB Knowledge & Model Files]]
- [[EQB Model & Language Coda]]
- [[EQB Runtime Mind]]
- [[EQB Task Field & Generators]]

## Used by

- [[EQB Runtime Mind]]
- [[HRM-X Experiments]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-012 EQB and HRM-X Audit - Linear Core, Recalled Language, Live BOTTOM]]
- [[GPU Safety - VRAM Oversubscription and DPC Watchdog]]
- [[HRM-X Specification v0.3 - Understanding by Construction]]

## Changed or discussed in sessions

- [[Session - 2026-09-27 - HRM-EQB Full Build]] — commits `0afeafc`, `1553230`, `2c82d7e`, `483c0c1`
- [[Session - 2026-09-29 - EQB and HRM-X Audit]] — named in the log

## Tests

`tests/test_eqb_model.py`, `tests/test_eqb_runtime.py`

## Recent history

- 2026-09-27 0afeafc HRM-EQB: --pause duty cycle for GPU thermal headroom (88 C observed); session log draft
- 2026-09-27 2c82d7e HRM-EQB: bench OOM-as-skip, CLI test isolation, hormone drive test, docs
- 2026-09-27 1553230 HRM-EQB: bounded training memory, checkpoint/resume, demo-implied canvas rule
- 2026-09-27 483c0c1 HRM-EQB N1-N7: task field, model, file formats, runtime mind, self-study, CLI
