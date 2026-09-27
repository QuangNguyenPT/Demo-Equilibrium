---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM Perception

> Domain protocol; grid, sequence and 3-D domains; the program language.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm/perception/__init__.py` |  | 9 |
| `hrm/perception/base.py` | Perception: the Domain protocol. | 258 |
| `hrm/perception/grid/__init__.py` |  | 7 |
| `hrm/perception/grid/domain.py` | Grid domain: ARC-style 2-D grids and voxel volumes, colors 0..9 (0 = background). | 180 |
| `hrm/perception/grid/programs.py` | Grid programs: compositions of primitives with an executable and an algebraic face. | 198 |
| `hrm/perception/grid/segmentation.py` | Segmentation hypotheses over voxel grids (flood fills in pure numpy). | 106 |
| `hrm/perception/grid/symmetry.py` | Symmetry operators on 2-D grids and (Z, H, W) volumes. | 59 |
| `hrm/perception/grid/volume.py` | Grids are native 3-D: shape (Z, H, W), coordinates written (x, y, z). | 27 |
| `hrm/perception/sequence.py` | Sequence domain: integer sequences in, integer sequences out. | 302 |
| `hrm/perception/vector3d.py` | Vector3D domain: dense 3-vectors in, dense 3-vectors out. | 424 |

## Depends on

- [[HRM Model Facade & CLI]]
- [[HRM Substrate (FHRR)]]

## Used by

- [[HRM Language]]
- [[HRM Learning Loops]]
- [[HRM Metacognition]]
- [[HRM Model Facade & CLI]]
- [[HRM Reasoning]]
- [[HRM Verification]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-003 Unified Cognitive Model]]

## Changed or discussed in sessions

- [[Session - 2026-09-27 - Benchmark v2 and the Reasoning Ceiling]] — named in the log
- [[Session - 2026-09-27 - Unified Model Build]] — named in the log

## Tests

`tests/test_language.py`, `tests/test_model_online.py`, `tests/test_perception_verify.py`, `tests/test_reasoning.py`, `tests/test_sleep.py`, `tests/test_vector3d.py`

## Recent history

- 2026-09-27 2c2675c Baseline: hrm/ unified model (ADR-003), hrm_eqb N0 certified core, vault, benchmarks
