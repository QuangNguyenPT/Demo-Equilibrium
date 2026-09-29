---
title: "ADR-012: EQB and HRM-X Audit — a Linear Core, a Recalled Language Score, and a BOTTOM That Now Counts"
type: adr
tags:
  - adr
  - audit
  - hrm-eqb
  - hrm-x
  - negative-result
  - concept-induction
status: accepted
date: 2026-09-29
related:
  - "[[ADR-006 HRM-EQB Training and Benchmark Results]]"
  - "[[ADR-007 Grounded Algebra and Executable Concepts]]"
  - "[[ADR-010 Structural Operator Inference]]"
  - "[[ADR-011 Reintroducing EQB Ideas into HRM-X]]"
  - "[[HRM-EQB Specification v0.2 - Theorems and Proofs]]"
  - "[[HRM-X Specification v0.5 - Structural Operator Inference]]"
---

# ADR-012: EQB and HRM-X Audit — a Linear Core, a Recalled Language Score, and a BOTTOM That Now Counts

## Status

Accepted. One code correction to the live model (`hrm_x/`), two latent crash fixes, one
opt-in data option in the baseline (`hrm_eqb/`), three new experiments, and a set of
measured findings. ADR-011's proposal A is **measured and falsified**; B and C are **not
built**, for reasons recorded below.

## What "EQB" is in this repository

`EQB` is **HRM-EQB** (`hrm_eqb/`, ADR-005/006, spec v0.2): the project's only *learned*
reasoning model. Its input is a few-shot task — demonstration pairs plus a query — encoded as
one token field (content token bound to `(row, col, slot)` by rotation, spec §14). The path is

```
task (demos + query)  →  token field (FHRR rotation binding)  →  3-layer transformer prelude  x
  → per segment s = 1..3:
       routings A_L, A_H = S1(Sinkhorn(attention(x + carry(h*_{s-1}))))     fixed during the solve
       injections c_L = inject_low(x) + memory(u),  c_H = inject_high(x) + chain(h*_1..h*_{s-1})
       (h*, l*) = certified fixed point of  T_L, T_H   (no grad)  →  3 cycles with grad (JFB)
       logits = readout(h* at answer cells) · Embedᵀ ;  halt = σ(w·pool(h*))
  → loss  (1/S) Σ_s CE(logits_s, target) + 0.2 BCE(halt_s, exact_s) + 0.2 CE(coda, description)
```

The learning signal is the exact answer (and a rule description for the coda). What the
objective rewards is *any* map from the task field to the answer — nothing in it
distinguishes computing the rule from recalling it. `hrm_x/` (the live model since ADR-008/010)
learns nothing by gradient; ADR-011 is about carrying EQB's *method* (certify a quantity)
into it.

## Findings

Each finding carries its evidence class: **[D]** demonstrated by an experiment or test in this
commit, **[S]** supported but incomplete, **[H]** hypothesis, **[X]** contradicted by evidence.

### E1 — The EQB equilibrium core is linear in its input at the shipped setting [D]

`σ` in `T_L, T_H` is the radial ReLU with shrink `μ`. The shipped `μ = 0` makes it the identity
(spec v0.2 §3 already remarks this for the sharpness proof). With routings fixed during the
solve, each segment's equilibrium is therefore the resolvent `z* = (I − K(A))⁻¹ c` — a linear
function of the injections, i.e. APPNP-style linear propagation over the routing graph, not a
nonlinear reasoner. All nonlinearity lives in the prelude, the routing softmax, the carry
between three segments and the readout.

Measured on the trained `eqb-small` (`experiments/eqb_probe.py` P1): additivity error
`‖z*(c₁+c₂) − z*(c₁) − z*(c₂)‖ / ‖z*(c₁+c₂)‖ = 6.6e-8` at `μ = 0`, `1.1e-2` at `μ = 0.05`,
`1.2e-1` at `μ = 0.5` (the probe can fail). Pinned by
`tests/test_eqb_model.py::test_core_fixed_point_is_linear_in_its_injections_at_the_shipped_shrink`.

Measured cycle rate from the trained weights `r = 0.304` against a budget of `0.685`; the
JFB gradient through 3 cycles therefore truncates the Neumann series at `r³ ≈ 2.8 %` — the
gradient *is* a good approximation of the implicit gradient (unlike HRM's, see Research).
The mathematics is correct; what it correctly computes is a linear solve.

### E2 — Extra recurrence does not change EQB's accuracy [D]

P2, 690 held-out generated tasks (seed 20260929): accuracy with 1 / 2 / 3 (trained) / 6
segments = 0.320 / 0.316 / 0.315 / 0.317; with the equilibrium capped at 1 / 2 / 5 / 40 cycles =
0.317 / 0.316 / 0.315 / 0.315 (s.e. ≈ 0.018). One cycle already gives the same answer as the
converged solve on 94.6 % of tasks. Recurrence and depth are not load-bearing for the answers
this model gets right. ADR-007's F8 ("every solve converges in ~5 cycles whatever the task")
is the mechanism; this is the consequence.

### E3 — The 0.983 language score is a recalled table, not a rule [D]

The text families draw inputs from a fixed list of 90 words. P4: text accuracy **0.956** on
those words against **0.006** on unseen letter strings of the same lengths (per family:
0.90–1.00 vs 0.00–0.03). Sequences: 0.141 at the trained lengths (4–6), 0.000 at 7–8. Grids:
0.000 at every size. P5: no grid task solved in any view.

### E4 — Demonstrations are used, to *select* what is recalled [D]

P3: swapping the demonstrations for another family's (query and target kept) drops accuracy
0.315 → **0.000**, including every text family (1.00 → 0.00). The model is not ignoring the
demos: caesar with shift 1, 2 or 3 is answered 1.00 in-vocabulary, so the shift is read from
the demos. Combined with E3: the model uses the demos to choose which stored `(word, rule)`
entry to output. That is conditional recall, and it is the most precise statement this repo
can currently make about what EQB learned from its task field.

### E5 — Every EQB module receives gradient [D]

One tiny-preset step: finite, nonzero gradient reaches prelude, both routings, carry, both
injections, memory (keys, values, query, relation, out), chain_in, all four certified maps,
readout, halt and coda. The one exact zero is `chain.query` on a 2-segment model (softmax over a
single chain element) — structural, not a defect. The learning path is intact; the ceiling is
representational (E1, E3), as ADR-007 argued.

### E6 — HRM-X ignored its own BOTTOM hypothesis [D] → fixed

Spec v0.5 Def. 3: a reserved hypothesis "outside my language" keeps prior mass
`2^-bits_bottom`. `Concept.propose` discarded that mass and `Mind._belief` normalised over the
readings alone, so the prior cancelled. Measured before the fix: moving `bits_bottom` from 4 to
∞ to −100 (BOTTOM 2¹⁰⁰× more probable than every reading) changed **0 of 52** E13 outcomes.
This — not only "one question defeats the coverage guard" — is why ADR-010 measured the
closed-world ablation as "close to inert".

**Fix.** `hrm_x/mind.py` gains `OUTSIDE`, a hypothesis answer that keeps its share of the
normalising mass but is never a value, never verified and never the answer; in
`_value_of_question` it joins every predicted outcome with likelihood `spread = 2^-log2|Y|`
(the Bayes factor Prop. 4 already charges per surplus row). `Concept.propose` adds the BOTTOM
hypothesis; `ask` refits its weight with everyone else's; weights are the raw posterior on
both paths (`propose` used `p / max p`, `ask` used `p`, so a re-propose mixed scales). The
closed-world ablation (`bits_bottom = inf`) reproduces the pre-fix numbers exactly.

| shipped config (open world, teacher, L = 1) | before | after |
|---|---|---|
| E13a in-language (15) | 13 / 2 / 0 | 9 / 2 / 4 |
| E13a out-of-language (6) | 1 / **5** / 0 | 0 / **0** / 6 |
| E13b pinned in-language (2) | 2 / 0 / 0 | 2 / 0 / 0 |
| E13b pinned out-of-language (3) | 0 / **3** / 0 | 0 / **0** / 3 |
| held-out, 40 in-language operators | 36 / 3 / 1 | 22 / 2 / 16 |
| held-out, 21 out-of-language operators | 3 / **16** / 2 | 2 / **2** / 17 |

(correct / wrong / abstain; held-out = 61 seeded operators nobody looked at while designing the
fix, `experiments/x_margin.py`.) Wrong answers on operators outside the language: **24 → 2**
across E13 + held-out. The price is real: in-language correct answers drop 51 → 33, because with
two evidence rows and a two-parameter reading (surplus 0) the evidence is uninformative and the
posterior *is* the prior, where BOTTOM (2⁻⁴) outweighs any one reading (≈ 2⁻¹⁰); a question is
then worth 0.017 utility against a price of 0.04, so the mind abstains rather than asks.
Net utility at the controller's own L = 1 is unchanged on the held-out set (20 → 20) and rises on
E13 (6 → 9); at the delivery's L = 4 it goes −37 → +8 held-out.

`bits_bottom` sweep (teacher; E13 and held-out; correct / wrong / abstain): 0 → 10/0/16,
24/2/35 · **4 → 11/2/13, 24/4/33** · 8 → 16/5/5, 25/5/31 · 12 → 16/5/5, 28/8/25 · ∞ → 16/10/0,
39/19/3. Reported, not tuned: the spec's 4 bits stays. The constant was inert before this fix and
is now the one that sets the precision/recall trade, so it inherits doc 13's "unjustified
constant" status as the first open question.

**G4 status**: out-of-language operators are now refused in 26 of 30 (E13 + held-out), with
2 wrong — improved, not closed. G3 regressed from 13/15 to 9/15 on E13a. Both stay open.

### E7 — Two latent crashes in `Concept.propose` [D] → fixed

(a) Evidence no family fits → `max()` of an empty posterior → `ValueError` inside `Mind.solve`.
(b) The "nothing new" path returned `(None, 0)` while `Mind.solve` tests `got is None` →
`TypeError` on any re-propose that found nothing. E13's hand-written tables never reached
either; the first seeded held-out set hit (a) at once. Both return `None`. Regression test
`test_evidence_no_family_fits_is_an_abstention_not_a_crash`; the test that pinned the buggy
tuple now pins `None`.

### E8 — ADR-011 A: the rival margin does not separate right from wrong [X]

`concepts.rival_margin` (log₂ mass behind the answer minus the best disagreeing reading) on
every pre-fix E13 answer: correct 13.3 … ∞ bits, wrong 8.4 … ∞ bits — **five wrong answers have
infinite margin** (no fitting reading disagrees; the truth is simply outside the language). The
held-out set: 11 wrong answers at ∞. ADR-011's falsifier fires: "not a calibration problem — it
needs a different mechanism". The mechanism that worked is E6: the margin compares readings with
each other, BOTTOM compares them with *everything the language cannot write*.

A post-hoc alternative, `min_surplus` (answer only from a reading that predicted ≥ k rows it was
not fitted to), separates better (held-out, k = 2: 19 correct / 1 wrong) but only as a filter:
the controller never asks to *test* a sole surviving reading, so with a teacher it answers 20 of
61. Kept as an opt-in `ConceptConfig.min_surplus` (default 0, no behaviour change) and an E14
condition, not shipped.

### E9 — ADR-011 B and C have nothing live to guard [D]

- **B (leave-one-out on load)**: the concept route's memory is write-only. `Concept.remember`
  stores readings; nothing reads them back except `ambiguity()`, and `propose` always refits from
  the task's evidence. A LOO check on load would protect no computation. Not built.
- **C (held-out keep gate for consolidation)**: `hrm_x/consolidation.consolidate` has no runtime
  caller — only tests. The "most dangerous assumption" is not exercised by the running system at
  all. Not built; it becomes relevant the day `Mind.sleep` calls it.

## Research that changed or confirmed the reading

| claim | source | relevance | confidence |
|---|---|---|---|
| HRM's H/L hierarchy is within ~5 pp of a same-size transformer on ARC-AGI-1; the outer refinement loop (deep supervision) drives the gain, mainly at training time; most performance comes from training on the evaluation tasks | ARC Prize, "The Hidden Drivers of HRM's Performance on ARC-AGI", 2025-08-15; code `arcprize/hierarchical-reasoning-model-analysis` | same pattern as E2: inner recurrence not load-bearing | high (replicated ablation, public code) |
| HRM does not reach the fixed point its 1-step/IFT gradient assumes; back-propagating through the full recursion raised Sudoku-Extreme 56.5 % → 87.4 %; forcing true fixed-point iteration (TorchDEQ) slowed training and generalised *worse* | Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks", arXiv 2510.04871 | EQB is exactly the "make it a certified DEQ" direction; E1/E2 are consistent with TRM's negative result | high for the measurement, medium for transfer to this scale |
| A linear contraction's fixed point is personalised-PageRank-style propagation | Klicpera et al., APPNP, ICLR 2019 | names what E1 measured | established |
| Truncated backprop from a fixed point approximates the implicit gradient with error `O(rᵏ)` | Fung et al., "JFB", AAAI 2022; Geng et al., phantom gradients, NeurIPS 2021 | EQB's gradient is sound (r = 0.30) — the problem is not optimisation | established |

None of this argues for re-importing TRM or HRM into `hrm_x/`: `hrm_x/` has no learned
recurrence to improve. It argues against reading EQB's certificate as evidence of reasoning.

## What was changed

| file | change |
|---|---|
| `hrm_x/mind.py` | `OUTSIDE` answer + `_is_value`; `Hyp.spread`; BOTTOM likelihood in `_value_of_question`; propose trigger counts readings only |
| `hrm_x/domains/concept.py` | BOTTOM hypothesis in `propose`/`ask`/`questions`/`remember`; raw-posterior weights; both `propose` crash paths return `None`; `min_surplus` note |
| `hrm_x/concepts.py` | `rival_margin` (measurement only); `Fitted.min_surplus` / `tested()`; `ConceptConfig.min_surplus = 0` |
| `hrm_eqb/tasks.py`, `hrm_eqb/train.py` | opt-in `text_source="letters"` (default `"words"`, bit-identical to before) |
| `experiments/eqb_probe.py`, `eqb_text.py`, `x_margin.py` | EQB-P, EQB-T, E14 |
| tests | 6 new; 5 adjusted — the ADR-010 regression test now pins the fix plus its closed-world ablation, the exhaustion test pins `None`, three tests skip the BOTTOM hypothesis when picking a reading (`test_x_concept.py`, `test_eqb_model.py`). Suite: 442 passed, 1 skipped |

`experiments/results/x_v05.json` is re-recorded under the fixed route; ADR-010's table is the
pre-fix record and stays as written.

## Consequences

- G4 improved (24 → 2 out-of-language wrong), G3 regressed (in-language 51 → 33 correct). The
  next lever is calibrating `bits_bottom` from experience (the mind's own Beta rates) or letting
  the controller buy a *test* question for a lone untested reading — not tuning 4 bits on E13.
- EQB's language score must not be quoted as language ability. EQB-T (below) measures whether
  the architecture can learn the rule once recall is impossible.
- "Recurrent reasoning" in `hrm_eqb/` is not demonstrated: within a segment the computation is
  linear, and across segments more depth does not help.

## EQB-T — can EQB learn the text rule when recall is impossible?

`experiments/eqb_text.py` → `experiments/results/eqb_text.json`. Fresh `small` models, text
domain only, 20 minutes each on the capped GPU (≈ 1.8–2.2k steps), seed 0, **one seed per arm**;
300 evaluation tasks per suite (50 per family).

| arm | training words | unseen strings, lengths 3–5 | unseen strings, lengths 6–8 |
|---|---|---|---|
| `words` (ADR-006 distribution) | 0.647 | **0.000** | 0.000 |
| `letters` | 0.160 | **0.140** | 0.000 |
| `letters`, shrink 0.1 (nonlinear core) | 0.203 | **0.200** | 0.000 |

Per family on unseen strings, `letters` / `letters+shrink`: upper 0.80 / 1.00, double 0.04 / 0.18,
sort 0.00 / 0.02, reverse, caesar, drop_vowels 0.00 / 0.00.

Reading, with evidence classes:

- **[D]** Trained on the word list, the architecture reproduces the recall result from scratch:
  0.647 on training words and exactly 0 on unseen strings.
- **[D]** Remove recall and it learns a *per-position* rule (upper: each output byte is a function
  of the input byte at the same cell) and a little of `double`. Every rule that needs either a
  parameter read from the demonstrations (caesar's shift) or a position-to-position routing
  (reverse, sort, drop_vowels) stays at 0.00 — the kind of computation the architecture's claims
  are about.
- **[D]** No arm generalises in length: 0.000 at 6–8 characters for every arm, including `upper`.
- **[H]** The nonlinear core helps (0.140 → 0.200, ≈ 1.9 standard errors, one seed, and the gain is
  mostly `upper` saturating). Supported direction, not a demonstrated effect.
- **[S]** "EQB learns reusable structure from its task field" is supported only for per-position
  maps at trained lengths; for everything else it is not demonstrated at this budget. Longer
  training could change the numbers; it would not change that the shipped language score was
  recall (E3), which needs no training to establish.
