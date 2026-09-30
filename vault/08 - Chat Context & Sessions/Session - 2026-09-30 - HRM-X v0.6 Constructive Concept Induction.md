---
title: "Session - 2026-09-30 - HRM-X v0.6 Constructive Concept Induction"
type: session
tags: [session, hrm-x, v0.6, construction, connectome, research-program]
date: 2026-09-30
commits: [5a40a66, 130a797]
code:
  - hrm_x/construct.py
  - hrm_x/concept_memory.py
  - hrm_x/language.py
  - hrm_x/model.py
  - hrm_x/curriculum.py
  - hrm_x/__main__.py
  - hrm_x/signature.py
  - hrm_x/mbi.py
  - hrm_x/mind.py
  - hrm_x/concepts.py
  - hrm_x/qlinalg.py
  - hrm_x/domains/concept.py
  - hrm_x/native/src/construct.cpp
  - experiments/connectome_malecns.py
  - experiments/x_sampling.py
  - experiments/x_router.py
  - experiments/x_transfer.py
  - experiments/x_understanding.py
  - tests/test_x_construct.py
---

# Session — 2026-09-30: HRM-X v0.6, constructive concept induction

**Goal.** Move HRM-X from *selecting among written families* to *constructing, verifying, storing
and reusing* concepts; integrate what survives of EQB; inventory and use the local MaleCNS
connectome; build a trainable, runnable model with a C++ hot path; test understanding
operationally. Decision: [[ADR-013 Constructive Concept Induction]]. Spec: [[architecture_spec]].
Research record: `vault/10 - Research Program/`.

Started by merging `feat/hrm-x-v04` (hrm_eqb moved to `legacy/`), so this work builds on it.

## Information path

| question | evidence | decision | code | result |
|---|---|---|---|---|
| can a missing family be built? | M1 in [[hypotheses]] | construct features from primitives, fit their linear closure exactly | `construct.py`, `Concept._construct` | E19 constructible 0 → 20 of 24 held-out |
| does stored structure transfer? | MaleCNS M3: one KC code, ~9 gated readouts per KC | concept memory, two-task admission, concepts as primitives | `concept_memory.py` | E17 rows needed fall only for related targets; memory off 60 → 44 |
| should search sample randomly like KCs? | M2 | no | `x_sampling.py` | enumeration 55.5 % vs random 9 % at 80k |
| should a learned index order candidates? | M4, Kimi K3 router invariant | no (kept as a measured module) | `signature.py`, `mbi.py`, `x_router.py` | top-1 5–7.5 % vs chance 2.5 %; ordering cannot change an exhaustive level |
| why did wrong answers appear? | E17/E19 diagnoses | regime coverage, empirical \|Y\|, distinct surplus | `concepts.py` | E19 wrong 5 → 0 |
| where does learning pay? | PVR learns a base-7 notation from sums | learned numeral reader in the language path | `language.py` | test passes; base identified and verified exact |
| is the native path worth it? | profile | enumeration native (bit-identical); exact RREF stays Python but cached | `construct.cpp`, `qlinalg.RowSpace` | 4–6× enumeration; 5.7× on a construction-heavy solve |

## Key results (all [RESULT])

- E19 held-out, 81 tasks: v0.6 **60 / 0 / 21**; v0.5 + ADR-012 18 / 0 / 63; memory off 44 / 0 / 37;
  closed world 63 / 12 / 6; nearest row 5 / 76; least squares 15 / 66.
- MaleCNS: doc 14's reciprocity (0.299 vs 0.0055) and claw counts reproduce from the raw tables;
  new: each KC is read by a median 9 of 37 MBON types, each MBON type is DAN-gated.

## Open questions / next

1. The primitive alphabet is given — can primitives themselves be induced (e.g. from a stored
   concept's failure modes)?
2. One nonlinear feature per family — two-feature families multiply the search; library growth may
   be the cheaper route.
3. Language remains template-bound (ADR-009).
4. Thin-evidence recall (E13, 2–3 rows) fell: calibrated evidence asks for more rows or questions.
