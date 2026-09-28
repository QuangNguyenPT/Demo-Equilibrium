---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM-X Native Backend

> C++ e-graph and grid kernels behind a C ABI, parity-tested against the Python reference.

## Modules

| module | what it does | lines |
|---|---|---|
| `hrm_x/native/__init__.py` | Native backend (C ABI over ctypes) for the measured hot paths; Python modules stay the reference. | 262 |
| `hrm_x/native/include/hrmx.h` | hrmx native backend: C ABI for the measured hot paths of HRM-X (spec v0.4 §23). | 78 |
| `hrm_x/native/src/egraph.cpp` | E-graph core; reference semantics: hrm_x/workspace.py (Workspace) and hrm_x/rules.py (_match, matches). | 399 |
| `hrm_x/native/src/grid.cpp` | Grid perception and object-level steps; reference semantics: hrm_x/domains/grid.py (objects, apply_step). | 130 |

## Depends on

- [[HRM-X Workspace & Rules]]

## Used by

- [[HRM-X Domains]]
- [[HRM-X Experiments]]
- [[HRM-X Workspace & Rules]]

## Governed by (specs, ADRs, concepts, guides)

- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]

## Changed or discussed in sessions

- [[Session - 2026-09-28 - HRM-X v0.4 PoC and Native Backend]] — commits `d5bacb1`

## Tests

`tests/test_x_native.py`

## Recent history

- 2026-09-28 d5bacb1 HRM-X v0.4: operator inference over structured situations + C++ backend (ADR-008)
