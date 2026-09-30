---
title: HRM-X v0.6 — Constructive Concept Induction (architecture specification)
type: specification
tags: [spec, hrm-x, v0.6, construction, concept-memory]
status: implemented
created: 2026-09-30
supersedes: "[[HRM-X Specification v0.5 - Structural Operator Inference]]"
related: ["[[ADR-013 Constructive Concept Induction]]", "[[research_plan]]", "[[evaluation]]"]
---

# HRM-X v0.6 — Constructive Concept Induction

v0.5 fits hypothesis families exactly but every family is written by hand. v0.6 keeps everything
v0.5 verifies and adds the step it could not take: **building a family the language does not
contain, verifying it, storing it, and reusing it.** No equation below is aspirational; each names
the function that computes it.

## 1. Data flow

```
text ──parse──► evidence E = {((x_i, y_i), z_i)}, query (x*, y*)          hrm_x/language.py
        numerals: exact decimal, or learned place-value reader (θ, b)       hrm_x/reader.py (optional)
                                   │
H level  Mind.solve: options {propose, derive-group, verify, ask, stop}, value of computation
                                   │
L level  propose depth 0: exact fit over F_written (68) ∪ F_memory (admitted concepts)
         propose depth k: construct features of ≤ s_k tokens (s = 3, 5, 7), fit their closures
         derive (lookup) · verify (dual solve) · ask (oracle, h-question horizon)
                                   │
answer | abstain ──► concept memory (candidate → admitted → retired) ──► explain ──► text
```

## 2. State

| state | type / shape | created | updated | read by | persists |
|---|---|---|---|---|---|
| evidence `E` | list of ((ℚ, ℚ), ℚ), n rows | parser / task | `ask` appends | fit, construct | per task |
| hypothesis `h` | `Hyp(key, w, payload=Fitted)` | propose | ask (refit), verify | controller | per task |
| BOTTOM | `Hyp(answer=OUTSIDE, w=1, spread=2^-log₂|Y|)` | propose depth 0 | never (reference) | belief, VOI | per task |
| constructed families | `Family` list | construct | — | belief (refit after ask) | per task |
| concept memory | JSON dict in `mind.knowledge["concept_memory"]` | `remember` | record / admit / fail / name | depth-0 fit, construction alphabet, language | across tasks, saved |
| modulators, statistics | as v0.5 | Mind | observe/feedback | price λ, rates ρ | saved |
| reader (θ ∈ ℤ¹⁰, b ∈ ℤ) | snapped integers | `learn_notation` | never after acceptance | parser | saved |

## 3. Construction (hrm_x/construct.py)

Alphabet A = {x, y, 1, 2} ∪ {+, −, ×, ÷} ∪ {ifgt} ∪ {LIB_k : k admitted}; ifgt(a,b,c,d) = c if a>b else d;
LIB_k(a,b) = concept k applied to (a,b). A feature g is a postfix expression over A.

- **Enumeration** (bottom-up by size, `enumerate_bank`): values on probe set P (|P| = 14 dyadic
  points, no zero coordinate); an expression is kept iff its value vector v_g ∈ ℝ^P is new at 1e-9
  resolution (observational equivalence). Native twin `hx_enumerate`, bit-identical.
- **Canonical class** (`canonical_features`): r_g = (I − Π_B) v_g with Π_B the projector on
  B = span{1, x, y, xy}|_P; drop g if ‖r_g‖∞ < 1e-7·max(1,‖v_g‖∞); key = r_g / r_g[argmax |r_g|].
  One feature per key (shortest). Counts: 6 / 69 / 1,348 / 23,071 features at ≤ 3 / 5 / 7 / 9 tokens.
- **Description length** `bits(g) = |g| · log₂|A|` (a larger library costs every token a little more).
- **Family of g**: m(x,y) = a·g(x,y) + Σ_{s∈S} b_s s(x,y), S ⊆ {x, y, 1} (8 families, bare {g}
  dropped when g is a monomial); prior bits = bits(g) + 3 + |S|.
- **Fit** (`construct`): for each g of the level: float least squares on {g, x, y, 1} as a necessary
  condition (prefilter), then the exact v0.5 solve for families that could come within `margin`
  (10 bits) of BOTTOM: −bits + (n − k)·log₂|Y| + sym > −bits_⊥ − margin.
- **Weight** (v0.5 Prop. 4 with §3b's calibrated |Y| and distinct surplus): log₂w = −bits − paramBits(p) + (m − rank)·log₂|Y| + sym, m = distinct rows of A.

## 3b. Three rules the larger hypothesis space forced (hrm_x/concepts.py)

A hypothesis space that grows by construction makes coincidental fits likelier, so two v0.5
assumptions had to become explicit:

- **Calibrated answer space** (`answer_bits`, default `answer_space="empirical"`): a predicted row
  is worth log₂|Y| bits against BOTTOM, with log₂|Y| = clip(log₂((2⌈max|z|⌉ + 1)·max denom z), 2, 10)
  instead of the fixed 10. With the small integers of these tasks a chance match costs 4–5 bits,
  not 10. Measured (E19): removes every construction-caused wrong answer (5 → 2, the remaining two
  are in the written language in every condition) at the price of 6 correct → abstain.
- **Regime coverage** (`Fitted.order_covered`): an asymmetric reading may not answer at a query whose
  sign(x − y) never occurred in the evidence; symmetric readings are exempt (swapping the arguments
  maps the query into the seen regime), and the diagonal needs both sides seen. Reason: once ifgt
  features are constructible, 2x + 1 and 2·max(x, y) + 1 agree on every row with x ≥ y.

- **Distinct surplus** (`distinct_surplus=True`): surplus = (distinct rows of the family's feature
  matrix A) − rank, not n − rank. A repeated feature vector tests that z depends on these features,
  not which function of them: with max(x, y) ∈ {6, 9} on every row, 15·max − 54 "predicted" three
  rows under the v0.5 count and answered 381 where the truth was 841 (E19 diagnosis).

## 4. Controller changes (hrm_x/mind.py)

- **Unnormalised, BOTTOM-relative weights** w = 2^(log₂w + bits_⊥): readings found at different
  depths and after different asks share one scale; Mind normalises.
- **Propose when OUTSIDE dominates**: option value p_found(level)·share_⊥ − λ·cost(level) when the
  OUTSIDE share of the mass exceeds ½ (v0.5 proposed only when no reading existed).
- **Question horizon**: VOI_h(q) with OUTSIDE's likelihood spread^h in every predicted outcome;
  option value max_h VOI_h − λ·h·(ask_ops + cost), h ≤ 3, only when OUTSIDE is alive (other
  domains unchanged).
- **Propose semantics** (`Concept.propose`): always re-fit first (new evidence can admit readings);
  construct only if the re-fit adds nothing; construction depth tracked per task; `None` = exhausted.

## 5. Concept memory (hrm_x/concept_memory.py)

A concept = its expression (may call earlier concepts), name, status, tasks, successes, failures,
last 32 evidence rows. Identity = canonical behaviour key on P (symbol-free).
- `record`: a verified answer whose winning family came from construction or memory stores / 
  strengthens the concept; **admitted** when verified on a second, different task.
- Admitted concepts: depth-0 families (1 token each) and construction primitives LIB_k.
- `fail` ×(successes + 2) → **retired** (kept, never used, never deleted).

## 6. Language (hrm_x/language.py)

Six statement templates, four query templates, naming ("call ⊕ larger") and named queries; any
other text is refused with the ADR-009 reason. Numerals: exact `DecimalReader`, or `LearnedReader`
with the place-value parameters learned from written sums and accepted only when exact.
`explain` renders the answer and the trace facts (constructed/learned/written, surplus rows, asks).

## 7. Training, checkpoints, CLI (hrm_x/model.py, hrm_x/curriculum.py, hrm_x/__main__.py)

`HRMX.train(Curriculum)` solves pools written → constructible → composite with a teacher;
every task logs a JSONL row; checkpoint dir = config.json, mind.pkl, concepts.json, reader.json,
state.json; resume skips done steps. `python -m hrm_x create|train|chat|notation|info|eval`.

## 8. Complexity (measured)

Bank enumeration (native): 0.064 s at 7 tokens, 0.72 s at 9. A construction level costs
|F_level|·n·|g| float prefilter ops + exact fits of survivors; depth 3 at n = 5 ≈ 46k prefilter ops.

## 9. What was tried and removed (evidence in `[[ablation_results]]`)

- **Anytime stop** with a sound bound (sum of remaining weight ceilings): never fired → removed.
- **Learned ordering** (FHRR signature → MBI): weak ranking, routing costs ~⅓ of exhaustive
  prefiltering, and ordering cannot change an exhaustive level → not wired in (kept as a measured module).
- **Random conjunctive sampling** (MB-style) instead of enumeration: loses at equal budget → not used.
- **Equilibrium core** (EQB): not revived (ADR-012).

## 10. Known limits

- The primitive alphabet is given; only families are no longer given.
- One constructed feature per family (plus its linear closure): concepts needing two independent
  nonlinear features (e.g. max + min²) are out of reach unless one is stored first.
- Construction on 4 rows is refused by design unless memory shortens the description (the Occam
  price is paid in evidence rows).
- Language is template-bound (ADR-009 unchanged).
