---
title: "ADR-011: Why HRM-EQB Left, and What It Would Take to Bring It Back into HRM-X"
type: adr
tags:
  - adr
  - hrm-x
  - hrm-eqb
  - certification
  - proposal
status: proposed
date: 2026-09-29
agent: "opencode/stealth/space-bunny-alpha"
model: "space-bunny-alpha"
related:
  - "[[ADR-005 Neural HRM-Phi Replaces Program Search]]"
  - "[[ADR-006 HRM-EQB Training and Benchmark Results]]"
  - "[[ADR-007 Grounded Algebra and Executable Concepts]]"
  - "[[ADR-008 Operator Inference over Structured Situations]]"
  - "[[ADR-010 Structural Operator Inference]]"
  - "[[HRM-X Specification v0.5 - Structural Operator Inference]]"
  - "[[HRM-EQB Specification v0.2 - Theorems and Proofs]]"
---

# ADR-011: Why HRM-EQB Left, and What It Would Take to Bring It Back into HRM-X

## Status

**Proposed. Nothing here is implemented.** This ADR exists to answer two
questions that keep recurring — *why did we abandon the certified model?* and
*is there anything in it worth keeping?* — and to answer the second with a
concrete, falsifiable plan rather than a mood.

## 1. Why it left: the diagnosis, not the scoreboard

`hrm_eqb/` scored 0.298 on generated tasks, 0.983 on language, 1.000 on schedule
invariance — and **0.000 on ARC (158 tasks) and 0.000 on Sudoku-Extreme (200
tasks)**. The scores are not the interesting part. ADR-007 gave the causes, and
they are three representation facts, none of which is about capacity:

**(a) The integer representation is a magnitude code, and it has a ceiling.**
Cross-validated `R² = 0.91` for `|v|`, `0.48` for `sign(v)`, **0.09** for signed
`v`. Cosine similarity correlates with `−‖|a|−|b||` at `r = 0.60`, not with
`−|a−b|` at `r = −0.10`. Re-measured additive-consistency error **1.34** against a
shuffled-label null of **1.40 ± 0.015** over 20 permutations — about four standard
deviations better than chance, nowhere near the `0` of an additive code. (ADR-006
recorded 1.39 against 1.41 on the 2026-09-27 run; the conclusion is unchanged and
the re-measurement is the one to quote.) Integer tokens are limited to ±64, so
`no data creates a token for 65`.

**(b) A global contraction has one fixed point, so it cannot search or allocate
compute.** Every solve converges in about five cycles *whatever the task*. The
certificate is exact and the consequence is fatal: convergence is provable,
therefore the number of steps is a property of the architecture rather than of
the question. Audit finding F8 names this — "the contraction certificate is what
makes compute non-adaptive" — against HRM-X's own measured 248 ops on easy
arithmetic versus 4,346 on hard, the opposite behaviour and the reason VOC
control exists.

**(c) No spatial inductive bias.** Every grid transform scored 0.000 across all
families.

### The deepest reason, and the one that transfers

ADR-007 put it in a single sentence:

> `hrm_eqb/` learns from data but only patterns (ADR-006). The v0.2 theorems are
> all *safety* theorems. **None states what the model can compute.**

Spec v0.2 §11 item 1 is explicit that the proofs "hold for *any* weights,
including useless ones". So the package was **certified and useless at the same
time** — and nothing in 21 theorems, a contraction certificate recomputed from
the weights at every load, and a forgery-resistant file format could detect that,
because those properties are all about *not being wrong*, never about *being
able*.

## 2. The uncomfortable part: we retired the certified model and kept an uncertified one

Here is what the audit of HRM-X v0.4 measured, and it is the same disease in a
milder form:

| | `hrm_eqb` (certified) | `hrm_x` v0.5 (uncertified) |
|---|---|---|
| halting | E6: residual `‖z_{t+1} − z*‖ ≤ r_n/(1−r_n)·‖z_{t+1} − z_t‖`, an error bound **computed from the weights** | `Mind.solve` halts when `VOC ≤ 0`. **No bound on the answer's error.** |
| per-answer certainty | `E_loo = 0`: a write is only kept if leave-one-out re-derivation still succeeds | `Concept.verify` certifies the **algebra exactly** (the dual `Aᵀλ = r` agrees with the primal) — and says nothing about whether the *fit* was right |
| refusal | the certificate forces contraction | posterior threshold `q > L/(1+L)` — a confidence, not a proof |

So the honest reading of ADR-008 is not "the certified model lost". It is
**"we set aside the only component in the project that could compute a bound, and
nothing replaced it."** `hrm_x` traded away the certificate and bought exactness
and adaptivity — good trades, both real — but the certificate did not come back.

This is what makes the question worth reopening. Not "should we go back to
`hrm_eqb`" — cause (b) says no — but "**can we take EQB's method and point it at
HRM-X's objects**".

## 3. The method, stated once

EQB never certified a model. It certified a **quantity**, from parameters that
were available at the moment of the claim:

- `C1/E6` — from the operators' Lipschitz constants: *how far is this iterate
  from the fixed point?*
- `V1/E_loo` — from the demonstrations: *would this write have survived without
  the datum it was learned from?*
- `G1` — from paired held-out differences: *is this change real, at confidence
  δ?*

Each turns "it worked" into "it worked, and here is the number that says how
much to trust it". HRM-X v0.5 has the raw material for all three and computes
none of them. That is the gap this ADR proposes to close.

## 4. Three proposals, cheapest first

### A. Certify the **number** — the rival-margin bound (targets G4 and G5)

**The gap.** v0.5 answers when `U > 0`, i.e. when the posterior is confident. On
the operators its own language cannot express, ADR-010 measured **five confidently
wrong answers out of six**. The determinacy test (Theorem 3) stops a *reading*
extrapolating outside the evidence span; it cannot stop a *different reading that
fits the evidence exactly* from being confirmed. That is doc 13's Q1 — the
coincidental-fit channel — and the delivery's own README lists it as an honest
limit.

**The proposal.** Stop answering on posterior mass alone. Carry two numbers on
every `Result` and answer only when both hold:

1. the existing **determinacy certificate** (exact, already computed): the query
   row lies in the row space of the evidence;
2. a **rival margin in bits** — `log2 q(top reading) − log2 q(best rival that
   disagrees at the query)` — against the evidence's own surplus count.

This is doc 13's **P3**, the rival-margin rule, which the delivery named and did
not implement. It is EQB's E6 move pointed at a different object: instead of
turning "it converged" into "it converged to within ε", turn "it answered" into
"it answered, and the best reading that disagrees is *d* bits behind".

**Evidence required.** Run the margin over `experiments/results/x_v05.json` and
report the margin distribution separately for correct and wrong answers.

**Falsifier.** If the five wrong out-of-language answers carry margins as large as
the thirteen correct in-language ones, the margin does not discriminate, and the
channel is not a calibration problem — it needs a different mechanism entirely.
*This measurement is the first thing to do, and it is cheap.*

### B. Verify the **write** — leave-one-out provenance (targets the ADR-009 blind spot)

**The gap.** ADR-009 established that `Story.verify` "cannot see" a misreading,
because it replays the tagger's own `Reading` through a different representation.
The shape of that failure generalises: verification that replays the model's own
intermediate object cannot falsify it. `Concept.remember` currently stores a
version space and its parameters with no check at all.

**The proposal.** Every stored concept keeps its evidence, and on load it is
re-derived **leave-one-out**: drop each datum in turn, refit, and require the
stored reading to still be in the posterior. A concept that only survives with a
particular datum in the set has memorised that datum. This is EQB's `V1` /
`E_loo = 0` verbatim, and it is cheap here for a reason EQB did not have: **the
fit is exact**, so one LOO check is one exact RREF, and a reading with `k`
parameters needs `k+1` of them before the check can fail.

**Evidence required.** Run LOO over every concept in the current corpora.

**Falsifier.** If nothing fails, the check is **untested, not passing** — the
repo's own rule, "a benchmark that cannot fail proves nothing". It must first be
seeded with a deliberately poisoned concept and shown to catch it. A check that
has never fired is not evidence of safety.

### C. A keep gate for consolidation (targets the most dangerous assumption)

**The gap.** `consolidation.consolidate` admits a macro when its corpus MDL gain
is positive — measured on the corpus it was derived from. Doc 13 states the
falsifier itself: *"If adding macros does not reduce N_verify on held-out tasks
drawn from the same generator … then consolidation is memorising rather than
abstracting."* That test has not been run.

**The proposal.** Admission becomes a statistical claim. A macro is kept only if
it reduces verification cost on a **held-out** split at the claimed δ, using EQB's
`G1` bound on paired per-task differences (Hoeffding + Bonferroni, false-keep
rate ≤ δ). The measurement harness already exists — `experiments/x_v05.py` plus
the E8-style study/held-out split.

**Evidence required.** Consolidate on the study corpus with 30% held out; report
the G1 bound next to the MDL gain.

**Falsifier.** If the held-out gain is not positive at δ, `consolidate` is
memorising, and `MacroLibrary` must not ship. This is the one proposal that can
retract an existing component rather than add one.

### Already present — do not import twice

**D. The committee.** EQB's `Q1` ("the best question is the committee's most
divided input", expected information gain about *which member is right*) is
already implemented in `hrm_x` as VOI over the question pool: `Concept.questions`
offers exactly the pairs where the surviving readings disagree, and E13a measures
`mean_asks = 0.667`. Nothing to do. Recorded here so a future agent does not
re-port it.

## 5. Explicitly not proposed

| idea | why not |
|---|---|
| restore the contraction core | it is cause (b): one fixed point cannot search or allocate compute |
| restore the ±64 integer token field | it is cause (a) |
| scale `dim` to ≥512 and train longer | ADR-007 rejected it as a *strategy*: "D1 and D3 are representational: no data creates a token for 65 or a variable-depth fixed point" |
| AttnRes, product-key 4-D memory, quaternion knowledge, the hormone box | never the binding constraint. The measured failures were representation and adaptivity, and both have since been replaced by something that works |

## 6. What the architecture looks like if A, B and C land

```
propose ─ exact fit over law-closed families          (already: concepts.py)
   │
derive ─ row-space determinacy test                    (already: Thm 3)
   │
verify ─ dual solve Aᵀλ = r  [exact, ALREADY IMPLEMENTED]
   │
answer ─ determinacy certificate  AND  rival margin ≥ threshold   ◄── A
   │
remember ─ version space + parameters + evidence
            └─ load → leave-one-out re-derivation                 ◄── B
consolidate ─ corpus MDL gain AND held-out reduction at δ         ◄── C
```

Nothing in that diagram is a neural network, and nothing replaces the exact
`Fraction` core. The three additions are **three numbers on objects that already
exist**: a margin, a LOO result, and a bound.

## 7. Sequencing, and what each step costs

| step | cost | unblocks |
|---|---|---|
| A. Measure the margin distribution on `x_v05.json` | one experiment, no new code | decides whether A is even a calibration problem |
| B. LOO re-derivation + a poisoned-concept canary | one function + one test | closes the ADR-009-shaped gap everywhere |
| C. Held-out consolidation with the G1 bound | one experiment | the project's most dangerous assumption |
| the neural baseline (Abacus, arXiv 2405.17399) | separate | **not** an EQB revival; the falsifier that was never run |

Order matters only in that **A must be measured before it is implemented**. If
the margin does not discriminate, A is the wrong fix and proposing it anyway
would be the "registry lookup dressed up as reasoning" the README rules out.

## 8. What this does not change

`hrm_eqb/` remains a baseline and not the target. Its 55 tests keep running, its
theorem corpus keeps its file, and nothing above resurrects the package or its
architecture. What is proposed is EQB's *method* — certify the quantity, not the
model — applied to the objects HRM-X already computes.

## Attribution

- opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
- Measurements cited: `experiments/results/x_v05.json` (ADR-010), ADR-006
  benchmark table, ADR-007 diagnosis, doc 01 and doc 13 of the v0.5 delivery.
