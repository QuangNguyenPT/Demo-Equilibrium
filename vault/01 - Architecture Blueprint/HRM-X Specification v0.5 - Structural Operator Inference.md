---
title: "HRM-X Specification v0.5 - Structural Operator Inference"
type: specification
tags:
  - hrm-x
  - specification
  - concept-induction
status: active
version: "0.5"
supersedes: "HRM-X Specification v0.4 - Operator Inference over Structured Situations"
date: 2026-09-29
agent: "opencode/stealth/space-bunny-alpha"
model: "space-bunny-alpha"
related:
  - "[[ADR-010 Structural Operator Inference]]"
  - "[[ADR-008 Operator Inference over Structured Situations]]"
---

# HRM-X Specification v0.5 — Structural Operator Inference

Labels used throughout: **[E]** established in the literature, **[R]** an
empirical result in this repository with the file that records it, **[H]** a
hypothesis, **[U]** unresolved.

Theorem, proposition and definition numbers are those of the v0.5 research
delivery (`docs/03_mathematical_foundations.md`) and are quoted, not restated.

## 1. What changed, and why

v0.4 inferred an operator by **searching** for it: enumerate programs up to a
size bound, group the observationally equivalent ones, keep the version space
[[ADR-008 Operator Inference over Structured Situations]]. The audit that accompanied the
v0.5 delivery measured the cost of that choice — finding **F3**, "the most important
negative result of the audit":

> 3 of the 4 out-of-DSL probes gave wrong answers with status `answer`. (doc 01)

and summarised every positive result in one line:

> Every positive HRM-X result is explained by: *exact execution + enumeration
> over a DSL that contains the target concept + tests whose distribution the DSL
> was built for.* (doc 01)

v0.5 changes the *type* of a hypothesis. A hypothesis is no longer a program to
be found; it is a **family** — the exact solution set of a law set — whose
rational parameters are fitted exactly. There is no enumeration and therefore no
size bound to grow.

## 2. The representation

A binary operator over ℚ is hypothesised by a **family** (Definition 2)

    m(x, y) = Σ_j p_j ψ_j(x, y),      p ∈ ℚ^k

whose members are exactly the solutions of a law set, so no member is an
arbitrary program. `hrm_x/concepts.py` builds 68 of them:

| family | laws | closed form | k |
|---|---|---|---|
| bi-affine supports over {xy, x, y, 1} | affine in each argument (Jensen ×2) | δxy + αx + βy + γ | ≤ 4 |
| `x^a y^b`, a, b ∈ [−3, 3] | multiplicative in each argument (Cauchy ×2) | c·x^a·y^b | 1 |
| `x^(a y)`, a ∈ {1, 2} | exponential in y, power in x | c·x^(a·y) | 1 |
| `c b^x`, `c b^y`, b ∈ {2, 3, 10} | exponential in one argument | | 1 |

Each fit is an exact RREF over `fractions.Fraction`. `hrm_x/qlinalg.py` is the
linear algebra; it is stdlib-only and the whole of `hrm_x.concepts` is
stdlib-only, because exact rational arithmetic stays in Python (Amdahl, ADR-008).

## 3. The three quantities that decide an answer

**Consistency.** `A p = b` has a solution. A family that cannot fit the evidence
is not a reading. **[E]**

**Evidence weight (Prop. 4).** "The probability that h fits the remaining
n − k points *by chance* is |Y|^{−(n−k)}." So

    log2 w(h) = −bits(h) − param_bits(p) + (n − rank(A)) · log2|Y| + symmetry bonus

Only the **surplus** rows buy evidence; the fitted parameters buy none.
`tests/test_x_concept.py::test_fitted_parameters_buy_no_evidence` asserts the
increment is exactly `log2|Y|` per predicted row. **[R]**

**Determinacy (Theorem 3).** "If the evidence is consistent, φ(g) is uniquely
determined **iff** g ∈ span_ℚ{xᵢ}." So a reading may answer only where its
feature row is in the row space of the evidence matrix; otherwise it has no
opinion and its posterior mass is reported as undetermined rather than guessed.
`::test_a_query_outside_the_evidence_span_has_no_opinion` **[R]**.

**Open world (Def. 3 + the coverage rule).** A reserved hypothesis BOTTOM, "the
concept is outside my language", holds `2^-bits_bottom` with no evidence term,
and a reading loses its entitlement at an argument value that never varied,
unless it is symmetric in `x, y`. **[E]**; asserted in
`::test_the_reserved_bottom_hypothesis_carries_its_prior_share` and
`::test_the_coverage_rule_refuses_an_argument_that_never_varied` **[R]**.

The answering rule is unchanged from ADR-008 and lives in `Mind._utility`:
`U = max(0, max_y (1 + L) q(y) − L)`, answer only when `U > 0`.

## 4. One mind, two routes

The mathematics is a `Domain` implementation (`hrm_x/domains/concept.py`), not a
second H-level loop. `Mind.solve` drives it: the same
`VOC(a) = E[U after a] − U − λ·cost(a)`, the same
`U = max(0, max_y (1+L) q(y) − L)`, the same modulators, patience, writer and
trace format. The delivery's standalone `concepts.decide()` is the reference the
upstream numbers were measured against and is **not** imported, because two
controllers in one repository is one too many and a head-to-head is only fair if
both arms share one.

`hrm_x/library.py` (program-level DSL mining) and `hrm_x/domains/arith.py`
(program search) remain, as the baseline `experiments/x_v05.py` measures against.

| `Domain` method | v0.4 (`Arith`) | v0.5 (`Concept`) |
|---|---|---|
| `propose` | enumerate programs, group by behaviour | fit every family exactly; return the readings not yet proposed for this task |
| `derive_cost` | rewrite + extraction + execution | one row-space test |
| `verify` | residue phasors over the extracted plan | the **dual** `A^T λ = r`, read off `b` |
| `ask` | prune hypotheses by the teacher's answer | append the datum and **refit** the posterior |
| `remember` | version space of programs | version space of readings; parameters as strings (JSON) |

`propose` returning `None` on its second call is the honest statement of the
project's most dangerous assumption: the hypothesis language is **given**. No
parameter here invents a family that was not written down. **[U]**

## 5. Verification is a different computation

Theorem 3 says the value is determined because the query row is in the row space
of `A` — which is the same statement as "the dual system `A^T λ = r` is
consistent". So `verify` solves the dual and reads the answer back as
`Σ_i λ_i b_i`, and requires exact rational equality with the primal
`Σ_j p_j r_j`. The second number is derived from the right-hand side rather than
from the parameters, so a transposed basis, a free-variable artefact or a
contaminated parameter vector shows up as a disagreement, and an undetermined
query fails for the right reason (`solve` returns `None` on an inconsistent dual)
rather than producing a least-squares answer. `::test_verification_catches_a_reading_whose_parameters_were_corrupted` **[R]**.

## 6. Consolidation

`hrm_x/consolidation.py`: a solved task leaves a fitted reading; a candidate
macro μ is that reading; μ enters the library when

    ΔL(μ) = Σ_{solved} [ L(sol | Λ) − L(sol | Λ ∪ {μ}) ] − L(μ) > 0

(DreamCoder-style, Rissanen). Once admitted, μ becomes a **feature**, so the
hypothesis language grows. Measured: a single solved task has ΔL < 0 and is
rejected; the same reading solved on three tasks is admitted and changes
`families()`. **[R]** (`::test_a_macro_that_cannot_pay_for_itself_is_rejected`,
`::test_repeated_behaviour_classes_pay_for_the_macro`).

What it does **not** do is create a new concept *kind*. That is the project's
most dangerous assumption and it remains untested. **[U]**

## 7. The other components, and what they are for

| module | what it is | why it is here |
|---|---|---|
| `hrm_x/situation.py` | quantities as a `Q^(E×I)` state; verb frames learned from one worked example each; exact arithmetic plus an independent conservation check | the story domain for the v0.5 route; it refuses an untaught frame and an unstated initial quantity rather than guessing |
| `hrm_x/algebra.py` | FHRR phasor binding, exact unbind, characters as homomorphisms; Prop. 8 cleanup bound | binding capacity (E1) and role swap (E2); numpy-only, torch optional |
| `hrm_x/operators.py` | `D4 × Sym(colours)` global actions and predicate-conditioned object actions, solved as MDL over readings rather than searched | the grid route; Theorem 6 makes composition exact after projection, so the error bound is 0 rather than small |
| `hrm_x/reader.py` | Place-Value Reader `v(s) = Σ_k θ_{d_k} b^k` (Thm 5) plus digit baselines | the one learned numeric component. **Requires torch at import and must not be wired into `hrm_x/__init__.py`** |

## 8. What is measured

`python -m experiments/x_v05` → `experiments/results/x_v05.json`. Both routes,
one `Mind`, 26 solves per condition.

| E13a, 21 queries | correct | wrong | abstain | mean ops |
|---|---|---|---|---|
| concept v0.5 | **14** | 7 | 0 | **751.5** |
| arith v0.4 | 9 | 4 | 8 | 4247.2 |

| split | concept v0.5 | arith v0.4 |
|---|---|---|
| in-language (15) | **13 / 2 / 0** at 593 ops | 7 / 4 / 4 at 4265 ops |
| out-of-language (6) | 1 / **5** / 0 at 1147 ops | 2 / **0** / 4 at 4202 ops |

Full reading, including the three negative results, is in
[[ADR-010 Structural Operator Inference]].

## 9. What is refused

- **Coincidental fits.** "The open-world posterior removed the confident errors
  that come from a *missing* hypothesis. It does not remove errors where a
  *simpler wrong* hypothesis fits the evidence" (doc 13, Q1). Measured: 5 wrong
  out of 6 out-of-language queries. Pinned by
  `::test_a_coincidental_symmetric_fit_is_still_confidently_wrong`. **[R]**
- **Asking defeats the coverage guard.** One question brings a second value of
  the argument into the evidence, so "did this argument ever vary?" becomes true
  and the rule stops firing. Pinned by
  `::test_one_question_destroys_the_coverage_guard`. **[R]**
- **L = 4 is not independently choosable.** The delivery calibrated its answer
  threshold at `L = 4`; `Controller` ships `L = 1`. At `L = 4` the route abstains
  on 21 of 21, because before any verification the prior `rho = 0.8` gives
  `U = (1+4)(0.8) − 4 = 0` exactly. **[R]**
- **No new concept kinds.** `consolidate` adds compositions, not kinds. **[U]**
- **No neural baseline.** Abacus (arXiv 2405.17399) has not been run; every
  comparison here is symbolic against symbolic. **[U]**

## 10. Gates

| gate | claim | state |
|---|---|---|
| G1 mathematics | every quantity is exact over ℚ, and Theorem 3 / Prop. 4 / Prop. 8 are tested against the implementation | done |
| G2 representation | a reading is a law-closed family, fitted exactly, never enumerated | done |
| G3 capability | in-language operators are answered exactly and cheaply | done — 13/15 at 7.2× fewer ops than v0.4 |
| G4 generalization | operators outside the language are *refused*, not guessed | **not met** — 5 confident wrong of 6 |
| G5 anti-shortcut | no lookup dressed as reasoning; no result explained by the task distribution | **not met** — §9, Q1 |
| G6 efficiency | metareasoning cost is bounded and small | **not met** — the delivery measured 35k meta ops against 3.3k object ops and nothing bounds the estimator (doc 13, Q3) |

G4 and G5 are the reason this specification is v0.5 and not v1.
