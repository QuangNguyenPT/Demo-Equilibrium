---
title: "ADR-013: Constructive Concept Induction — build the family, verify it, store it, reuse it"
type: adr
tags: [adr, hrm-x, v0.6, construction, concept-memory, connectome, negative-result]
status: accepted
date: 2026-09-30
related:
  - "[[ADR-010 Structural Operator Inference]]"
  - "[[ADR-011 Reintroducing EQB Ideas into HRM-X]]"
  - "[[ADR-012 EQB and HRM-X Audit - Linear Core, Recalled Language, Live BOTTOM]]"
  - "[[architecture_spec]]"
  - "[[evaluation]]"
  - "[[ablation_results]]"
  - "[[biological_inspiration]]"
---

# ADR-013: Constructive Concept Induction

## Status

Accepted and implemented as HRM-X v0.6 (`hrm_x/construct.py`, `concept_memory.py`, `language.py`,
`model.py`, `curriculum.py`, `__main__.py`, native `hx_enumerate`). Spec: [[architecture_spec]].

## Context

After ADR-012, v0.5 refuses what its 68 written families cannot express (BOTTOM is live), but it
can never *learn* such a concept: the hypothesis language is fixed, and the stored version spaces
are never read back. EQB, the only learned model, memorised its training table (ADR-012 E3).
The research question: can the system construct a family it was not given, verify it, keep it, and
reuse it — without re-opening the confidently-wrong channel ADR-012 closed?

## Decision

1. **Construct features, not programs.** Bottom-up enumeration of postfix expressions over
   {x, y, 1, 2, +, −, ×, ÷, ifgt} and stored concepts; deduplicated by observational equivalence and
   modulo span{1, x, y, xy}; a family is a feature's linear closure, fitted by v0.5's exact solve.
   Only skeletons are searched; coefficients are solved (v0.5's lesson, one level up).
2. **Construction is a propose depth under value-of-computation control**, reached only when
   OUTSIDE holds the mass; depth budgets 3 / 5 / 7 tokens; the level is exhaustive.
3. **Concept memory**: a verified construction is a *candidate*; a second verification on a
   different task *admits* it; admitted concepts are depth-0 features (one token) and construction
   primitives. Identity is behaviour, not symbol.
4. **Calibrated evidence**: a predicted row is worth log₂|Y| bits with |Y| estimated from the
   evidence's own values (not a constant 10); surplus counts distinct feature rows; asymmetric
   readings may not extrapolate across an unseen sign(x − y) regime. All three were forced by
   measurement (E17/E19 wrong answers), not chosen in advance, and each is pinned by a test.
5. **Learning where it measurably works**: the place-value reader (numerals from written sums,
   snapped and verified) is the input-learning path; a learned router is *not* wired in.
6. **Native backend** for enumeration (bit-identical, 4–6× faster); exact rationals stay in Python;
   row-space tests are cached (5.7× on a construction-heavy solve).

## Evidence (all in `experiments/results/`)

- **E19** held-out (81 tasks, 26 operators, 9 never trained): v0.6 trained **60 / 0 / 21**
  (correct / wrong / abstain) against **18 / 0 / 63** for v0.5 + ADR-012; memory off 44 / 0 / 37;
  library tokens off drops composites 14 → 0; closed world 63 / 12 / 6. Surface variants 8/8
  identical, revision 6/6. Details: [[evaluation]], [[ablation_results]].
- **E17** with `max` stored, rows for ≥ 75 % correct: |a−b| >7 → 5, relu >7 → 5, 2max+1 7 → 4,
  max(a,b²) 7 → 5; the unrelated a²+b stays at 4; memory off reproduces empty memory exactly.
- **E13** (2–3-row v0.5 harness): 0 wrong (v0.5 as ported: 10; ADR-012: 2), at lower recall.
- **E15b** random (mushroom-body-style) sampling loses to enumeration at equal budget (9 % vs 55.5 %
  at 80k); library compression reaches composites neither reaches at 80k.
- **E16** learned FHRR→MBI router: top-1 5–7.5 % vs chance 2.5 %; kWTA no better than dense;
  routing costs ≈ ⅓ of exhaustively prefiltering the level.
- **B1** MaleCNS: doc 14's numbers reproduce from the raw tables; new M3 (one KC code, ~9 gated
  readouts per KC) motivates memory-as-readouts.

## Rejected (with the measurement that rejected it)

| idea | why rejected |
|---|---|
| random conjunctive sampling (M2) | E15b: fewer targets and 4.6× fewer distinct classes at equal budget |
| learned ordering / MBI router in the loop (M4) | E16: weak signal; ordering cannot change an exhaustive level |
| anytime stop with a sound bound | never fired: the remaining ceilings always exceed the best reading |
| fixed |Y| = 2¹⁰ with construction | E19: 3 construction-caused wrong answers (e.g. gcd → 1 via ifgt(x, y − 2, y, 1)) |
| reviving the EQB equilibrium core | ADR-012 E1–E2; TRM (arXiv 2510.04871) |
| more hand-written families | moves the wall; the null v0.6 is measured against |

## Consequences

- G4 (refuse outside the language) holds for construction: 0 of 9 out-of-primitive tasks answered.
- G3 is no longer "the language covers it": concepts outside the written language are acquired,
  at an evidence price set by their description length, lowered by memory.
- The primitive alphabet is still given; one nonlinear feature per family; language stays
  template-bound (ADR-009). These are the next research targets, in that order.
