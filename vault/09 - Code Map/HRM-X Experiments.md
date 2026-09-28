---
type: code-subsystem
generated: knowledge_map
tags:
  - code-map
---

# HRM-X Experiments

> E1-E12 runners with baselines and ablations; results recorded as JSON.

## Modules

| module | what it does | lines |
|---|---|---|
| `experiments/__init__.py` |  | 1 |
| `experiments/common.py` | Shared experiment plumbing: seeded runs, result records, and a results directory under version control. | 30 |
| `experiments/x_arith.py` | E1 numeric extrapolation, E2 compositional arithmetic, E12 efficiency from discovered laws. | 171 |
| `experiments/x_control.py` | E9 adaptive computation, E10 verification effectiveness, and the hierarchy (H vs L) test. | 228 |
| `experiments/x_grid.py` | E6 rule composition, E7 structural transformation on new arrangements, E11 symbol permutation, | 334 |
| `experiments/x_metacog.py` | Curiosity and volatility: does self-directed study pay, and does the NE-like signal adapt the mind? | 125 |
| `experiments/x_native.py` | Native backend speedup on the profiled workloads (the justification for C++, spec v0.4 §23). | 85 |
| `experiments/x_story.py` | E3 variable binding, E4 context-dependent roles, E5 novel relation combinations, E8 counterfactuals, | 274 |

## Depends on

- [[HRM-X Domains]]
- [[HRM-X Mind & Induction]]
- [[HRM-X Native Backend]]

## Used by

- none

## Governed by (specs, ADRs, concepts, guides)

- none

## Changed or discussed in sessions

- none recorded

## Tests

none

## Recent history

- none
