---
title: "ADR-008: Operator Inference over Structured Situations (HRM-X v0.4)"
type: adr
tags:
  - adr
  - hrm-x
  - architecture
  - native
status: accepted-poc
date: 2026-09-28
superseded_by: "[[ADR-010 Structural Operator Inference]]"
related:
  - "[[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]"
  - "[[ADR-007 Grounded Algebra and Executable Concepts]]"
  - "[[ADR-006 HRM-EQB Training and Benchmark Results]]"
  - "[[ADR-001 Hybrid C++ Engine with Python Blueprint]]"
  - "[[ADR-009 Free Text Is Not Yet Supported]]"
  - "[[Session - 2026-09-28 - HRM-X v0.4 PoC and Native Backend]]"
---

# ADR-008: Operator Inference over Structured Situations (HRM-X v0.4)

## Status

**Accepted as a proof of concept.** It is built in `hrm_x/`, tested (34 HRM-X tests at the
time; `tests/test_x_*.py` now collects 128) and measured in `experiments/results/`.
**Superseded as the design target 2026-09-29** by
[[ADR-010 Structural Operator Inference]], which changed what a hypothesis is rather than how
it is found. The program-search route below remains in `hrm_x/` as the measured baseline; this
record is kept verbatim as the decision record.
- It supersedes the ADR-007 design (v0.3): grounded algebra survives as the *verifier's*
  representation, and executable concepts become programs inferred from evidence.
- `hrm_eqb/` and `hrm/` stay as baselines and references.

Labels: [E] established, [R] measured here, [H] hypothesis, [U] unresolved.

## Context

The research prompt asks for a model that shows it understands structure, meaning, context and
mathematics. The example: from 2×2=4 and 2×3=6 it should get 10×3.1=31 through a reusable
operation.

The prompt also asks for one cognitive theory rather than "modules plus a transformer".
- ADR-006 showed that HRM-EQB learned patterns but no arithmetic.
- ADR-007 proposed grounded algebra, but its concepts were hand-written and its "compiler"
  could hide memorization.

The owner's hand-drawn architecture includes three things that were not yet defined as
computations:
- a cognitive matrix;
- an EQB matrix with H and L levels;
- a self-loop with imagination, a hormone layout, and curiosity.

## Decision

A task is **situation + operator + question**. Understanding means inferring the operator and
executing it exactly on a structured situation, with a check that uses a different
representation.

1. **Workspace = e-graph with provenance** [E: egglog]. This is the "cognitive matrix".
   - Invariants, tested: I1 congruence; I2 constant uniqueness, where a conflict is refused
     and blamed; I3 explanations.
   - "Follows from" means sharing a class in the least fixpoint of saturation.
   - Similarity never creates an equality.
2. **Operators are programs; belief is a posterior over behaviour classes** (observational
   equivalence, prior 2^-size) [E: program induction].
   - Concepts are stored as **version spaces**, not winners.
3. **Laws by theory exploration** [E: QuickSpec].
   - Laws are proposed by testing and pruned by congruence closure, with domain guards.
   - Planning rewrites with laws and place value, folds cheap applications exactly, then
     extracts the cheapest term.
4. **H/L as computation, not as two networks** [E: ARC-Prize HRM analysis, TRM].
   - L = saturation, extraction and exact execution (the sketch's "EQB matrix": a least
     fixpoint).
   - H = a value-of-computation controller over {propose, derive-group, verify, ask, stop}. It
     answers only when U > 0 and no action is worth its cost.
   - [R] Additions: the derive-group macro-action, plan-then-cost, answerability
     conditioning, and kind-analogy predictions.
5. **Independent verifiers.**
   - Arithmetic: the definition evaluated in residue-phasor space (A2 exponentiation).
   - Stories: dictionary replay plus conservation.
   - Grids: applicability conditions.
6. **Transformer only for convention** (the surface → role tagger, 40,908 parameters).
   Names and numbers are copied by role.
7. **Hormones only where they compute** [E: Doya 2002; Yu & Dayan 2005].
   - ACh-like: reliability ρ.
   - DA-like: outcome feedback.
   - NE-like: volatility, meaning surprise in excess of the expected entropy H(ρ). It discounts
     stale evidence.
   - 5-HT-like: λ, kept fixed [U].
   - No other hormone is added.
8. **Curiosity = expected information gain** over a concept's version space, times usage.
   - Used in sleep, together with law consolidation and re-testing laws on imagined values.
9. **C++ backend behind a C ABI** (ADR-001 conventions). Python is the blueprint and the
   reference; C++ runs the profiled hot paths:
   - grid components and object steps;
   - the e-graph (union-find, hash-cons, rebuild, e-matching, explanations).

   Parity is tested differentially. Exact Fraction arithmetic stays in Python on purpose.

## Evidence (all [R], `experiments/results/*.json`)

| Claim | Result |
|---|---|
| exact extrapolation (E1) | 1.000 on in-range, 10²–10⁴ integers, large decimals, 10×3.1; kNN 0.41→0, MLP and log-MLP 0 |
| composition (E2) | 1.000 at depths 1–3 |
| abstraction pays (E12) | 1000×3.1: 1,021 → 192 ops; 10⁴×3.1: 10,020 → 411; 10⁵×3.1 refused without laws, 779 ops with them |
| adaptive compute (E9) | VOC 1.000 / 248 ops easy, 0.975 / 4,346 hard; fixed policy 3,836 / 10,182; greedy 0.75 on hard |
| verification (E10) | false law: 0 wrong with the verifier, 9 without; perception role swap: 10 wrong either way |
| curiosity | × identified with 1.0 question vs 4.8 random; later accuracy 0.86 vs 0.71 (none: 0.67, 33% wrong) |
| volatility | 2 wrong answers after a regime change vs 6 without the signal |
| native | identical results; grid 3.8×, arithmetic 1.5× |

Added 2026-09-28, after the ADR was accepted. These qualify its claims rather
than overturn it, and the decision stands; the gates they touch stay off.

| Claim | Result |
|---|---|
| DSL library growth | real MDL compression (40→38) and exact expansion, but a merely *present* library made search 9.4× worse; the mined step never pays at one use per program and only pays on a high-redundancy corpus. `use_library_bound` defaults off |
| learned price of computation (patience) | measured against a fixed lambda swept over patience's own reachable range: −0.90 / −0.36 accuracy. The fixed constant stays; the shipped E9 baseline is unchanged |
| free-text perception | held-out transfer fails — 1.00 exact on seen forms, 0.50 unseen verb, **0.00** changed clause order, 76–93% of held-out stories misread. See [[ADR-009 Free Text Is Not Yet Supported]]; closed templates only |

## Consequences

- Answers are exact *relative to the hypothesis languages*. When no reading answers, the mind
  says "I don't know" and gives the reason.
- Grouping predictions are abstract (kinds and probe fingerprints) [H]. When they are wrong,
  the mind spends extra work; they cannot make an answer wrong, because answers come only from
  derivation and verification.
- The story verifier replays the same reading, so perception errors pass it (E10). This is a
  stated blind spot, not a solved problem.
- The CLAUDE.md invariants for `hrm_eqb/` (certified maps, the VRAM cap) do not apply to
  `hrm_x/`. It is CPU-only and needs no CUDA.

## Rejected

- **End-to-end neural reasoning (HRM-EQB as the reasoner).** It measured no arithmetic (ADR-006).
- **Hormones as free scalars in a box.** They had no ablation and no computational role
  (ADR-007 finding). They were kept only where the controller consumes them.
- **Separate H/L networks.** They are not load-bearing on small puzzles [E]. The hierarchy is
  kept as a separation of computation.
- **Porting exact arithmetic to C++.** Arbitrary precision is the guarantee. Amdahl's law caps
  the arithmetic speedup at about 1.5×, and that is accepted.

## Falsifiers (spec §26)

The design is refuted if any of these hold:
- **(a)** A transformer with structure-aware embeddings and a comparable budget matches
  extrapolation, counterfactual consistency and contradiction refusal.
- **(b)** Library learning cannot grow the DSLs needed for a new domain.
- **(c)** Perception errors dominate on free text.

**Most dangerous assumption:** that the hypothesis languages can be *grown* rather than given.
The next experiment is DreamCoder-style compression on ARC-AGI training tasks.
