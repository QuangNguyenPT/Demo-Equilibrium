---
title: "Session: 2026-09-28 — HRM-X v0.4 PoC and Native Backend"
type: session-log
tags:
  - session
  - hrm-x
  - native
  - metacognition
agent: "claude-code"
model: "claude-opus-5-5"
date: 2026-09-28
status: built-and-measured
commits: []
---

# Session: 2026-09-28 — HRM-X v0.4 PoC and Native Backend

## Request

- Run the full research prompt (/ecc:loop-design-check): a model that shows it understands
  structure, meaning, context and mathematics. From 2×2=4 and 2×3=6 it should get 10×3.1=31
  through a reusable operation.
- It must be one cognitive theory, not modules plus a transformer, with 26 deliverables and
  demos A–E.
- Then continue:
  - build a C++ backend with Python as the blueprint;
  - re-check the aim against the prompt;
  - read the hand-drawn architecture and `docs/`;
  - address the hormone system and curiosity. The owner warned that it "is just a layout plan…
    don't add random nonsense".
  - update the vault through graphify.

## Decisions → [[ADR-008 Operator Inference over Structured Situations]]

- A task is **situation + operator + question**. Each part of the owner's sketch maps to a
  computation; the mapping table is in spec §5.
  - cognitive matrix = e-graph workspace with provenance;
  - EQB matrix = rule saturation to a least fixpoint;
  - H level = the value-of-computation controller;
  - self-loop = sleep (curiosity study, law consolidation, re-testing laws on imagined values).
- **Hormones:** only four signals were kept, each with a defined computation. The rest of the
  layout was *not* implemented.
  - ACh-like = reliability ρ;
  - DA-like = feedback;
  - NE-like = volatility, meaning surprise in excess of the expected entropy H(ρ), which
    discounts stale evidence;
  - 5-HT-like = λ, left fixed [U].
- **Curiosity** = expected information gain over a concept's version space, times usage.
- **C++:** only the profiled hot paths moved, behind a C ABI, with differential parity tests:
  - grid components and steps;
  - the e-graph (union-find, hash-cons, rebuild, e-matching, explanations).

  Exact Fraction arithmetic stays in Python.

## Bugs found and fixed (with regression tests)

- **Union-by-size swap.** It moved the roots but not the constants, so a value was dropped.
  Found by porting to C++.
- **Congruence conflicts** did not name their argument premises.
- **`is_nat` seen-set.** Memo bug.
- **Grid verifier too strict.** It demanded that every demo exercise every step. Now it needs
  support in at least one demo, plus applicability on the test.
- **Myopic VOC.** Two agreeing 50% readings looked worthless. Fixed with the derive-group
  macro-action.
- **"undefined" counted as an answer.** Fixed by answerability conditioning: the answer mass
  covers only the readings that answer.
- **Groups normalized over the total mass.** Their shares included readings predicted
  undefined, so a lone answering group never looked worth deriving. Now normalized over the
  answering mass.
- **Predictions by evaluating the question.** Each one was a derivation without the laws, so it
  erased the laws' savings (1000×3.1 cost 1,217 ops, not 388).
  - Replaced by abstract interpretation over kinds plus probe fingerprints.
  - A 200-op cap was tried first. It left big-integer readings ungrouped (T2 0.00) and mixed
    exact and analogy guesses (E2 depth 3 0.70).
- **Bad plan `f(f(10,9906),405)` (99,070 ops)** from a guessed loop count. Fixed with exact
  folding of cheap applications before extraction.
- **Volatility read 1.0 at once.** Its baseline started at 0. It is now referenced to the
  expected surprise H(ρ).
- **The E10 false law never fired.** Its guard required naturals. Now `f(x, z) = add(x, z)`.
- **False-law tests.** They planted the law only on the best reading, so the second reading
  answered correctly. They now plant it on every reading.

## Results (measured; `experiments/results/`)

| Experiment | Result |
|---|---|
| E1 extrapolation | HRM-X 1.000 on every split (T2 large integers now 7,784 mean ops, was 13,802); kNN 0.41 → 0; MLPs 0 |
| E2 composition | 1.000 at depths 1–3 |
| E12 laws | 1000×3.1: 1,021 → 192 ops; 10⁴: 10,020 → 411; 10⁵ refused without laws, 779 with |
| E9 control | VOC easy 1.000 / 248 ops, hard 0.975 / 4,346; fixed 3,836 / 10,182; greedy hard 0.75 |
| E10 faults | false law: 0 wrong (verifier on) vs 9 (off); perception role swap: 10 wrong either way |
| curiosity | 1.0 question vs 4.8 random; later accuracy 0.86 vs 0.71 (none 0.67, 33% wrong) |
| volatility | 2 wrong vs 6 without the signal |
| stories | 1.000 on iid, unseen names, large numbers, long chains, contradictions; baseline 0.21 / 0 / 0.03 |
| grids | seed 0: 0.984 / 0.984 / 0.931; seed 1: 1.000 / 1.000 / 0.833 |
| native | identical results; grid 3.8×, arithmetic 1.5× |

Tests: 34 HRM-X tests; the full suite passes.

## Files

- `hrm_x/`: `workspace.py`, `rules.py`, `induction.py`, `mind.py`, `writer.py`,
  `domains/{arith,story,grid}.py`, `native/` (`include/hrmx.h`, `src/egraph.cpp`,
  `src/grid.cpp`, `__init__.py`).
- `experiments/`: x_arith, x_story, x_grid, x_control, x_metacog, x_native, common.
- `tests/`: test_x_core, test_x_domains, test_x_native, test_x_mind.
- `scripts/build_native.ps1`. `scripts/knowledge_map.py` gains 5 HRM-X subsystems and C++
  sources.
- Vault:
  - [[HRM-X Specification v0.4 - Operator Inference over Structured Situations]];
  - [[ADR-008 Operator Inference over Structured Situations]];
  - ADR-007 marked superseded.

## Open questions

- Growing the hypothesis languages (DreamCoder-style compression on ARC-AGI) is the most
  dangerous assumption and the next experiment.
- The perception blind spot: the story verifier replays the same reading.
- A learning rule for patience (λ). Non-myopic value of computation.
