---
title: "ADR-007: Grounded Algebra and Executable Concepts (HRM-X)"
type: adr
tags:
  - adr
  - hrm-x
  - architecture
  - mathematics
status: proposed
date: 2026-09-27
related:
  - "[[HRM-X Specification v0.3 - Understanding by Construction]]"
  - "[[ADR-006 HRM-EQB Training and Benchmark Results]]"
  - "[[ADR-005 Neural HRM-Phi Replaces Program Search]]"
  - "[[ADR-004 Discriminating Benchmark and Reasoning Ceiling]]"
---

# ADR-007: Grounded Algebra and Executable Concepts (HRM-X)

## Status

**Proposed — awaiting the owner's approval before any code (phase X0).**

## Context

The trained HRM-EQB was checked on 2026-09-27. Evidence is in Part I of the spec.
- It learned per-position symbol maps (`text/*` ≈ 1.00) and a fuzzy number line.
- It learned **no arithmetic**. The additive-consistency error of its integer embeddings is
  1.39, where random vectors score 1.41.
- Integers are atomic tokens limited to ±64.
- Every solve converges in about 5 cycles whatever the task. A global contraction has one
  fixed point, so it cannot search.
- Grid families, ARC and Sudoku are at chance or 0.

Two models now each hold half of understanding:
- `hrm/` is exact but enumerative: a depth-2 ceiling, and a phasor substrate that is
  decoration (ADR-004).
- `hrm_eqb/` learns from data but only patterns (ADR-006).

The v0.2 theorems are all *safety* theorems. None states what the model can compute. That is
why each fix felt like a patch.

## Decision (proposed)

Adopt **HRM-X** as specified in
[[HRM-X Specification v0.3 - Understanding by Construction]]:
1. **Grounded algebra** (residue phasors) where addition *is* binding and multiplication *is*
   repeated addition, by theorems A1–A4. It has 0 learned parameters and was checked
   numerically before this ADR.
2. **Executable concepts.** Knowledge is a well-founded library of definitions. An H-level
   executor unfolds them, so depth equals the problem's own cost. Compiled skills are the
   L-level, trusted only through the keep gate.
3. **Transformers only where the knowledge is conventional.** A Reader turns English into
   programs, and a Writer turns traces into English. About 0.83M learned parameters in total.
4. **Judge and questions.** Exact checks and inverse checks. Ambiguity or unknown concepts
   produce a question, never a guess.
5. **One benchmark defines success:** English Understanding Suite v1 (U1 extrapolation,
   U2 definitions, U3 faithful explanations, U4 honest ignorance). Its ablations make each
   part load-bearing by measurement.

## Consequences

- `hrm_eqb/` becomes the measured baseline. Nothing is deleted.
- These are retired from the design, with evidence:
  - quaternion memory (0 measured contribution);
  - hormones and the AttnRes chain (no ablation, no capability claim);
  - the coda (replaced by the shared decoder);
  - global contraction as the reasoning guarantee (its mathematics moves to local cleanup).
- `hrm/` contributes its verifier, sleep gate and FHRR substrate. Its enumerative synthesis
  is replaced by learned, guided search.
- New code goes in a new package (proposed `hrm_x/`), built in phases X0–X6. Each phase ends
  with the suite green, the graph updated and a vault note.

## Rejected

- **Scale HRM-EQB** (wider model, longer training). D1 and D3 are representational: no data
  creates a token for 65 or a variable-depth fixed point.
- **Add an arithmetic module to HRM-EQB.** This would be another patch. The concept still
  could not enter by definition (U2), and explanations would not be traces (U3).
- **Pure program synthesis without a learned Reader.** It cannot read English, and
  enumerative search already hit its ceiling (ADR-004).

## Open decisions for the owner

1. Approve HRM-X as the direction (or amend the spec).
2. Package name and location (`hrm_x/` proposed).
3. First English scope: templated arithmetic, days and text operations (Suite v1), or broader.
