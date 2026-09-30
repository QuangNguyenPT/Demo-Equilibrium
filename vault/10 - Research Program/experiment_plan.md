---
title: Experiment plan (v0.6) — what each run decides
type: research-experiments
tags: [research, experiments]
created: 2026-09-30
related: ["[[hypotheses]]", "[[evaluation]]", "[[ablation_results]]"]
---

# Experiments, in the order they decide something

| id | question | runner → result | decides | controls |
|---|---|---|---|---|
| B1 | do doc 14's connectome numbers hold on the raw tables; how is the KC code read out? | `experiments/connectome_malecns.py` (needs pyarrow) → `connectome_malecns.json` | which biological principles are facts here | configuration null (analytic) |
| E13 | does construction change the v0.5 head-to-head? | `x_v05.py` → `x_v05.json` | construction inside the unchanged harness | closed world, no teacher, L = 4, v0.4 route |
| E14 | is the rival margin a gate? (ADR-011 A) | `x_margin.py` → `x_margin.json` | margin vs BOTTOM | seeded held-out operators, `bits_bottom` sweep |
| E15b | M2 random sampling vs enumeration vs M3 library at equal budget | `x_sampling.py` → `x_sampling.json` | how to reach beyond the exhaustive budget | same budget, 200 random 9-token targets + 4 named composites |
| E16 | M4 learned ordering (FHRR → MBI) | `x_router.py` → `x_router.json` | whether learning belongs in ordering | chance rank, kWTA vs dense, cost vs exhaustive prefilter |
| E17 | does a stored abstraction lower the evidence needed? (U7) | `x_transfer.py` → `x_transfer.json` | concept memory's value | empty memory, memory switched off, an unrelated target |
| E19 | U1–U10 on held-out tasks, with causal ablations | `x_understanding.py` → `x_understanding.json` | the architecture as a whole | fresh model, 6 ablations, 2 pattern-matching baselines |
| E-native | native enumeration parity and speed | `tests/test_x_native.py` | C++ backend correctness | bit-identity with the Python reference |

Run everything (≈ 30 min on one CPU; B1 ≈ 4 min with pyarrow):

```
python -m pytest -q
python -m experiments.x_v05
python -m experiments.x_margin
python -m experiments.x_sampling
python -m experiments.x_router
python -m experiments.x_transfer
python -m experiments.x_understanding
<python-with-pyarrow> -m experiments.connectome_malecns
```

Confounds controlled: every comparison inside one runner uses the same tasks (seeded), the same
controller and price λ, and — for ordering and sampling — the same evaluation budget. Held-out
evaluation restores memory before every task, so the test set cannot teach itself.
