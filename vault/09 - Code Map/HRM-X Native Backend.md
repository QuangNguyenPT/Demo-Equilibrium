---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM-X Native Backend

> C++ e-graph, grid and construction-enumeration kernels behind a C ABI, parity-tested against the Python reference.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_x/native/__init__.py` | Native backend (C ABI over ctypes) for the measured hot paths; Python modules stay the reference. | 287 |
| `hrm_x/native/include/hrmx.h` | hrmx native backend: C ABI for the measured hot paths of HRM-X (spec v0.4 §23). | 88 |
| `hrm_x/native/src/construct.cpp` | Bottom-up enumeration of feature expressions with observational equivalence (hrm_x/construct.py). | 189 |
| `hrm_x/native/src/egraph.cpp` | E-graph core; reference semantics: hrm_x/workspace.py (Workspace) and hrm_x/rules.py (_match, matches). | 399 |
| `hrm_x/native/src/grid.cpp` | Grid perception and object-level steps; reference semantics: hrm_x/domains/grid.py (objects, apply_step). | 130 |

## Depends on

- [[HRM-X Workspace & Rules]]

## Used by

- [[HRM-X Concept Construction (v0.6)]]
- [[HRM-X Domains]]
- [[HRM-X Experiments]]
- [[HRM-X Workspace & Rules]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-001 Hybrid C++ Engine with Python Blueprint]]
- [[ADR-003 Unified Cognitive Model]]
- [[Build & Toolchain Guide]]
- [[C-ABI & Interop Specification]]
- [[Configuration & Hyperparameters]]
- [[Development Workflow & Conventions]]
- [[Devlog - 2026-09 - Project Foundation]]
- [[Engine Architecture & Memory Model]]
- [[HRM Recurrent Stepper & ACT]]
- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]
- [[Hybrid System Philosophy]]
- [[Kernel Specifications]]
- [[Model Blueprint & Graph Layout]]
- [[Tensor Flow & Data Pipelines]]

## Changed or discussed in sessions

- [[Session - 2026-09-28 - HRM-X v0.4 PoC and Native Backend]] — commits `d5bacb1`
- [[Session - 2026-09-30 - HRM-X v0.6 Constructive Concept Induction]] — named in the log

## Tests

`tests/test_x_native.py`

## Recent history

- 2026-09-30 5a40a66 HRM-X v0.6 phase 1: concept construction, concept memory, language interface, native enumerator
- 2026-09-28 d5bacb1 HRM-X v0.4: operator inference over structured situations + C++ backend (ADR-008)
