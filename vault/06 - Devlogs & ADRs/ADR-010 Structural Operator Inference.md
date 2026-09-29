---
title: "ADR-010: Structural Operator Inference, and the Wrong-Answer Channel It Moves Rather Than Closes"
type: adr
tags:
  - adr
  - hrm-x
  - concept-induction
  - negative-result
status: accepted
date: 2026-09-29
agent: "opencode/stealth/space-bunny-alpha"
model: "space-bunny-alpha"
related:
  - "[[ADR-008 Operator Inference over Structured Situations]]"
  - "[[ADR-009 Free Text Is Not Yet Supported]]"
  - "[[HRM-X Specification v0.5 - Structural Operator Inference]]"
---

# ADR-010: Structural Operator Inference, and the Wrong-Answer Channel It Moves Rather Than Closes

## Status

Accepted. This ADR records **a port, a measurement, and a negative result**. The
measurement is better than ADR-008 on the axis the route was built for, and
worse on the axis ADR-008 was already good at. Both are recorded.

## Context

ADR-008 built a mind that infers an operator by **searching** for it: enumerate
programs up to a size bound, evaluate on shared probes, group the
observationally equivalent ones, and keep the version space. It works, and it was
measured working. What it does not do is decide *what kind of thing* an operator
is. The hypothesis language is a hand-written DSL, and the audit that accompanied
the v0.5 research delivery (doc 01) measured the consequence:

> From `2⊙2=4, 2⊙3=8` it answers `3⊙2 = 4`. From `3⊘1=3, 6⊘2=3` it selects the
> constant program and answers `12⊘3 = 3` (truth 4). **3 of the 4 out-of-DSL
> probes gave wrong answers with status `answer`.** (finding F3, labelled "the
> most important negative result of the audit")

The same audit's summary of every positive v0.4 result is the sentence this ADR
takes as its problem statement:

> Every positive HRM-X result is explained by: *exact execution + enumeration
> over a DSL that contains the target concept + tests whose distribution the DSL
> was built for.*

The v0.5 delivery (docs 01-14, `poc/`) supplies the alternative: a hypothesis is
a **family** `m(x,y) = Σ_j p_j ψ_j(x,y)` that is the exact solution set of a law
set, its rational parameters are fitted exactly by linear solve, and Theorem 3
says it may answer a query only where `ψ(query)` lies in the row space of the
evidence. Add the open-world coverage rule and the reserved hypothesis BOTTOM, and
there is a principled "I don't know".

## Considered options

1. **Keep the delivery as a side-by-side `hrmx_poc` package.** Two code paths for
   one idea, two results registries (`poc/results/EXP-*` and
   `experiments/results/*.json`), and the head-to-head driven by two different
   H-level loops, so no comparison is like-for-like.
2. **Port the mathematics and run it under the existing `Mind`.** One controller,
   one trace format, one results contract. The delivery's standalone
   `concepts.decide()` loop is not used; it is the reference the upstream numbers
   came from, not a second mind.
3. **Replace the v0.4 route.** Rejected: it is the baseline every claim is
   measured against, and deleting it would delete the evidence.

## Decision Outcome

Chosen option: **2**, with 3 kept as the comparison arm.

- `hrm_x/qlinalg.py`, `hrm_x/concepts.py`, `hrm_x/consolidation.py`,
  `hrm_x/situation.py`, `hrm_x/algebra.py`, `hrm_x/operators.py`,
  `hrm_x/reader.py` — the delivery's mathematics, ported.
- `hrm_x/domains/concept.py` — the same mathematics as a `Domain`, driven by the
  existing `Mind.solve`, the existing value-of-computation controller, the
  existing modulators and patience, the existing writer.
- `hrm_x/library.py` and `hrm_x/domains/arith.py` stay exactly where they were.
  They are the v0.4 route and the thing the new one is measured against.
- `experiments/x_v05.py` runs both routes over the same operators with the same
  controller and records `experiments/results/x_v05.json`.

One deviation from the delivery: `poc/operators.py` used
`scipy.ndimage.label`, and scipy is not installed here. The component finder is
`hrm_x.domains.grid.objects`, the repository's own, so there is one component
finder rather than two. The delivery also contained a live bug —
`np.nindices(g.shape)` does not exist, so `apply_global` raised `AttributeError`
on every call and the whole global route could not execute. It is `np.indices`,
which returns the identical arrays.

## Measurement

`python -m experiments.x_v05`, 26 solves per condition, both routes through one
`Mind`. Correct / wrong / abstain, and mean ops:

### E13a, the delivery's ten operators (21 queries)

| condition | correct | wrong | abstain | mean ops |
|---|---|---|---|---|
| **concept v0.5**, open world + teacher | **14** | 7 | 0 | **751.5** |
| concept v0.5, closed world + teacher | 14 | 7 | 0 | 751.5 |
| concept v0.5, no teacher | 9 | 8 | 4 | 643.1 |
| concept v0.5, open world + L=4 | 0 | 0 | 21 | 502.0 |
| arith v0.4, teacher | 9 | 4 | 8 | 4247.2 |
| arith v0.4, no teacher | 9 | 4 | 8 | 4247.2 |

### E13a split by whether the language covers the operator

| scope | concept v0.5 | arith v0.4 |
|---|---|---|
| in-language (15) | **13 / 2 / 0** at 593 ops | 7 / 4 / 4 at 4265 ops |
| out-of-language (6) | 1 / **5** / 0 at 1147 ops | 2 / **0** / 4 at 4202 ops |

### E13b, second argument pinned in the evidence (5 queries)

| condition | correct | wrong | abstain | mean ops |
|---|---|---|---|---|
| concept v0.5, open world + teacher | 2 | 3 | 0 | 1618.2 |
| concept v0.5, closed world + teacher | 2 | 3 | 0 | 1604.4 |
| arith v0.4, teacher | 3 | 0 | 2 | 4243.2 |

## What this measures, honestly

**Positive.** On operators the hypothesis language can express, the structural
route answers 13 of 15 against 7 of 15 for the program route, with **7.2x fewer
ops** (593 against 4265), and it abstains four times less. That is the axis the
route was built for and it holds.

**Negative, and larger than expected.** On operators the language does *not*
contain, the structural route is **worse than the program route**: 5 confidently
wrong answers out of 6, against 0 wrong and 4 abstentions. The delivery's README
reports the opposite direction (0 wrong against 10 for v0.4). Two reasons this
repository measures differently, both recorded:

1. **A different controller.** These numbers come from `Mind.solve`, not from the
   delivery's standalone loop. That is deliberate — it is the only way both
   routes are driven identically — and it changes what "the answer threshold"
   means (see the L=4 row below).
2. **Different task lists.** The operators here are this repository's.

**The mechanism, and it is pinned in the tests.** On `a^2 + b` with evidence
`(2,3) -> 7, (4,3) -> 19`, the full bi-affine family `3/2 xy + 3/2 x - 2y + 1`
fits both rows exactly. It is not symmetric, so the coverage rule refuses it at
`(5, 7)`. Then the controller **asks one question**, the answer brings a second
value of `y` into the evidence, `y` has now "varied", and the coverage rule stops
firing. The dual solve confirms the answer, and the mind returns 47 where the
truth is 32. So:

> The coverage rule asks "did this argument ever take a different value?", and
> the act of asking is what makes it take one.

`tests/test_x_concept.py::test_one_question_destroys_the_coverage_guard` and
`::test_a_coincidental_symmetric_fit_is_still_confidently_wrong` pin both halves
of this. A change that claims to close the channel has to fail them.

This is doc 13's Q1, and this ADR does not claim to have solved it: the
open-world posterior removes errors caused by a *missing* hypothesis, not errors
caused by a *simpler wrong* one that happens to fit.

**The closed-world ablation is nearly vacuous here, and the run shows it.** With
`y` varying in the evidence, coverage never fires and `bits_bottom = inf` changes
nothing (E13a is identical to four significant figures). E13b pins the argument
and moves the numbers by 14 ops out of 1618. The machinery is present, correct,
and, on this task list, close to inert. That is the honest reading, and it is why
E13b exists at all.

**L = 4 is incompatible with this controller's priors.** The delivery calibrated
its answer threshold at `L = 4`, i.e. answer only when `q > 0.8`; `Controller`
ships `L = 1`, i.e. `q > 0.5`. At `L = 4` the route abstains on **21 of 21**:
before any verification has been observed, `rho` is its Beta(4,1) prior of 0.8,
and `U = (1 + 4)(0.8) - 4 = 0` exactly, so no action has positive value. The two
constants are not independently choosable. This is doc 13's Q7 in concrete form.

## Consequences

### Positive

- The hypothesis language is now law-closed rather than program-shaped: the
  families are the exact solution sets of their law sets, so a fitted reading is
  not "a program that happened to fit".
- Verification is a genuinely different computation — the dual `A^T λ = r`
  rather than the primal `A p = b` — and it fails when it should.
- Deriving is a lookup. The expensive part moved into `propose`, which is the
  asymmetry that makes 593 ops against 4265 possible.
- `hrm_x/situation.py`, `hrm_x/algebra.py`, `hrm_x/operators.py` and
  `hrm_x/reader.py` arrive as unused-but-working components; their property
  tests are live, so they are not scaffolding.

### Negative / Trade-offs

- **A regression on the safety axis**, measured above. Out-of-language operators
  now produce confident wrong answers instead of abstentions. If F3 was the
  defect, the fix moved the defect rather than removing it.
- The language is still *given*. `Concept.propose` returning `None` on the second
  call is the honest statement of that, and it is the most dangerous assumption
  the project has (doc 13's "library growth can create new concept *kinds*").
  `consolidation.py` admits a macro only when its corpus MDL gain is positive,
  and on a single solved task that gain is **negative** — pinned in the tests.
- `reader.py` requires torch at import and must not be wired into
  `hrm_x/__init__.py`; the package is torch-free otherwise, and
  `operators.py` is numpy-only.

### Not done, deliberately

- The rival-margin rule (doc 13's P3) is the identified fix for the
  coincidental-fit channel and is **not implemented**. It is a design with its
  own evidence requirement, and its own ADR.
- The strongest neural baseline (Abacus, arXiv 2405.17399) has still not been run.
  Every claim here compares two symbolic routes.

## Attribution

- Code: `hrm_x/qlinalg.py`, `hrm_x/concepts.py`, `hrm_x/consolidation.py`,
  `hrm_x/situation.py`, `hrm_x/algebra.py`, `hrm_x/operators.py`,
  `hrm_x/reader.py`, `hrm_x/domains/concept.py`, `experiments/x_v05.py`,
  `tests/test_x_concept.py`.
- opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
