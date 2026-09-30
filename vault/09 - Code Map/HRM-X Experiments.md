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
| `experiments/connectome_malecns.py` | B1 — MaleCNS v1.0: re-measure the delivery's connectome principles and add one new one. | 161 |
| `experiments/eqb_probe.py` | EQB-P — what the trained HRM-EQB actually learned from its task field, measured. | 238 |
| `experiments/eqb_text.py` | EQB-T — can HRM-EQB learn a text *rule* when the training words cannot be memorised? | 99 |
| `experiments/x_arith.py` | E1 numeric extrapolation, E2 compositional arithmetic, E12 efficiency from discovered laws. | 171 |
| `experiments/x_control.py` | E9 adaptive computation, E10 verification effectiveness, and the hierarchy (H vs L) test. | 228 |
| `experiments/x_dsl.py` | Does a learned DSL actually make concept search cheaper, or only make it sound tidier? | 195 |
| `experiments/x_grid.py` | E6 rule composition, E7 structural transformation on new arrangements, E11 symbol permutation, | 334 |
| `experiments/x_margin.py` | E14 — ADR-011 step A: does the rival margin separate right answers from wrong ones? | 151 |
| `experiments/x_metacog.py` | Curiosity and volatility: does self-directed study pay, and does the NE-like signal adapt the mind? | 125 |
| `experiments/x_native.py` | Native backend speedup on the profiled workloads (the justification for C++, spec v0.4 §23). | 85 |
| `experiments/x_patience.py` | E9b: is a *learned* price of computation better than a fixed one? | 210 |
| `experiments/x_perception.py` | Held-out free text: does the learned perception transfer to surface forms it never saw? | 385 |
| `experiments/x_router.py` | E16 — does a learned index (FHRR signature -> mushroom-body code -> readouts) order features well? | 72 |
| `experiments/x_sampling.py` | E15b — reaching features beyond the exhaustive budget: random sampling vs enumeration vs library. | 115 |
| `experiments/x_story.py` | E3 variable binding, E4 context-dependent roles, E5 novel relation combinations, E8 counterfactuals, | 274 |
| `experiments/x_transfer.py` | E17 — does a stored abstraction lower the evidence needed to acquire a related concept? | 86 |
| `experiments/x_understanding.py` | E19 — the understanding suite: operational properties U1-U10, with causal ablations. | 245 |
| `experiments/x_v05.py` | E13 — the v0.5 concept route against the v0.4 program route, one controller. | 212 |

## Depends on

- [[HRM-X Concept Construction (v0.6)]]
- [[HRM-X Concept Induction (v0.5)]]
- [[HRM-X DSL Library]]
- [[HRM-X Domains]]
- [[HRM-X Mind & Induction]]
- [[HRM-X Native Backend]]

## Used by

- [[HRM-X Concept Construction (v0.6)]]

## Governed by (specs, ADRs, concepts, guides)

- [[ADR-009 Free Text Is Not Yet Supported]]
- [[ADR-010 Structural Operator Inference]]
- [[ADR-011 Reintroducing EQB Ideas into HRM-X]]
- [[ADR-012 EQB and HRM-X Audit - Linear Core, Recalled Language, Live BOTTOM]]
- [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]
- [[HRM-X Specification v0.5 - Structural Operator Inference]]
- [[biological_inspiration]]
- [[experiment_plan]]

## Changed or discussed in sessions

- [[Session - 2026-09-28 - DSL Growth, Free Text, and the Graph Pipeline]] — named in the log
- [[Session - 2026-09-28 - HRM-X v0.4 PoC and Native Backend]] — commits `d5bacb1`
- [[Session - 2026-09-29 - EQB and HRM-X Audit]] — commits `734792f`
- [[Session - 2026-09-29 - HRM-X v0.5 Structural Induction and Vault Cleanup]] — commits `f0dc2ae`
- [[Session - 2026-09-30 - HRM-X v0.6 Constructive Concept Induction]] — named in the log

## Tests

`tests/test_x_patience.py`, `tests/test_x_perception.py`

## Recent history

- 2026-09-30 5a40a66 HRM-X v0.6 phase 1: concept construction, concept memory, language interface, native enumerator
- 2026-09-30 657f59f Merge feat/hrm-x-v04 (hrm_eqb moved to legacy/) into the audit branch
- 2026-09-29 734792f Audit: EQB learning path and HRM-X BOTTOM hypothesis (ADR-012)
- 2026-09-29 f0dc2ae HRM-X v0.5: structural operator inference (ADR-010) + vault cleanup
- 2026-09-28 d5bacb1 HRM-X v0.4: operator inference over structured situations + C++ backend (ADR-008)
