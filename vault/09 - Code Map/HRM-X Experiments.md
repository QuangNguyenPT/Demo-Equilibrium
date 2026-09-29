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
| `experiments/x_dsl.py` | Does a learned DSL actually make concept search cheaper, or only make it sound tidier? | 195 |
| `experiments/x_grid.py` | E6 rule composition, E7 structural transformation on new arrangements, E11 symbol permutation, | 334 |
| `experiments/x_metacog.py` | Curiosity and volatility: does self-directed study pay, and does the NE-like signal adapt the mind? | 125 |
| `experiments/x_native.py` | Native backend speedup on the profiled workloads (the justification for C++, spec v0.4 §23). | 85 |
| `experiments/x_patience.py` | E9b: is a *learned* price of computation better than a fixed one? | 210 |
| `experiments/x_perception.py` | Held-out free text: does the learned perception transfer to surface forms it never saw? | 385 |
| `experiments/x_story.py` | E3 variable binding, E4 context-dependent roles, E5 novel relation combinations, E8 counterfactuals, | 274 |
| `experiments/x_v05.py` | E13 — the v0.5 concept route against the v0.4 program route, one controller. | 212 |

## Depends on

- [[HRM-X Concept Induction (v0.5)]]
- [[HRM-X DSL Library]]
- [[HRM-X Domains]]
- [[HRM-X Mind & Induction]]
- [[HRM-X Native Backend]]

## Used by

- none

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-009 Free Text Is Not Yet Supported]]
- [[ADR-010 Structural Operator Inference]]
- [[ADR-011 Reintroducing EQB Ideas into HRM-X]]
- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]
- [[HRM-X Specification v0.5 - Structural Operator Inference]]

## Changed or discussed in sessions

- [[Session - 2026-09-28 - DSL Growth, Free Text, and the Graph Pipeline]] — named in the log
- [[Session - 2026-09-28 - HRM-X v0.4 PoC and Native Backend]] — commits `d5bacb1`
- [[Session - 2026-09-29 - HRM-X v0.5 Structural Induction and Vault Cleanup]] — commits `f0dc2ae`

## Tests

`tests/test_x_patience.py`, `tests/test_x_perception.py`

## Recent history

- 2026-09-29 f0dc2ae HRM-X v0.5: structural operator inference (ADR-010) + vault cleanup
- 2026-09-28 d5bacb1 HRM-X v0.4: operator inference over structured situations + C++ backend (ADR-008)
