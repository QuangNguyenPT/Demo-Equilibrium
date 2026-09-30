---
title: "Model report — how HRM-X v0.6 learns, what it outputs, and how it differs from the original HRM"
type: research-report
tags: [report, hrm-x, v0.6, hrm, comparison]
created: 2026-09-30
related: ["[[architecture_spec]]", "[[evaluation]]", "[[ablation_results]]", "[[ADR-013 Constructive Concept Induction]]", "[[ADR-012 EQB and HRM-X Audit - Linear Core, Recalled Language, Live BOTTOM]]"]
---

# Model report

Evidence tags: **[RESULT]** measured in this repository, **[FACT]** external source,
**[HYPOTHESIS]** untested. Numbers come from `experiments/results/*.json`.

## 1. What the model is, in one paragraph

HRM-X v0.6 is **not a neural network and has no gradient-trained reasoning weights**. It is a
program that (a) writes down candidate explanations of a few worked examples of an unknown
operator, (b) solves for each explanation's parameters *exactly* (rational arithmetic), (c) weighs
explanations by how well the data pays for their complexity, (d) verifies the leading answer a
second, independent way, (e) refuses when "the operator is outside everything I can write" is the
best explanation, and (f) stores explanations that survived as reusable concepts. The original
HRM is the opposite design: a 27 M-parameter recurrent network trained by backpropagation.

## 2. How it actually learns — four different things, and only one is gradient descent

| # | what changes | mechanism | persists as | learned or designed? |
|---|---|---|---|---|
| L1 | the explanation of *this* task | exact linear solve over ℚ; weight = −description bits − parameter bits + (distinct predicted rows − rank) · log₂\|Y\| | nothing (per task) | **inference**, not learning |
| L2 | **concepts** | a verified construction becomes a *candidate*; a second verification on a different task *admits* it; failures retire it | `mind.knowledge["concept_memory"]` (JSON, checkpointed) | **learned from experience** |
| L3 | controller statistics | Beta-rates for "does verification pass", "does proposing find something", modulators, price of computation | Mind statistics (checkpointed) | learned online, small |
| L4 | numeral notation | place-value reader: 10 digit values + base learned by gradient from *written sums only*, snapped to integers, accepted only if every training sum is then exactly right | `reader.json` | **the only gradient-trained part** |
| — | calibration rules (regime coverage, empirical answer space, distinct surplus) | derived by *me/you* from measured wrong answers, then hard-coded | source code | **designed, not learned** — stated plainly |

### What one training run does ([RESULT] `python -m hrm_x train`, 34 curriculum tasks, ~10 s CPU)

```
written (12)      solved at depth 0 by the 68 written families, ~1–1.5 k ops each; memory stays empty
constructible (16) x-range: max, min, |a−b|, a²+b, relu(a−b), a/(a+b), (a+b)², 2max+1, each twice
                   first pass: constructs  → candidate concepts (ifgt(x,y,x,y), (x*x), …)
                   second pass: same operator, new evidence → **admitted**; ops fall
                   (e.g. max 16,685 ops the first time, ~2,000 the second)
composite (6)     max(a,b²), max·b, max(a,b)² — reachable within budget only via the stored `max`
end state         7 concepts stored: 3 admitted (max, x², x/(x+y)), 4 candidates (composites built on them)
```

"Learning" is therefore *accumulation of verified, named building blocks*, and the evidence it
buys is measurable: with `max` stored, rows needed for ≥ 75 % correct fall for related targets and
are unchanged for an unrelated one (E17, [RESULT]):

| target | rows needed, empty memory | rows needed, `max` stored |
|---|---|---|
| \|a−b\| | > 7 | 5 |
| relu(a−b) | > 7 | 5 |
| 2·max+1 | 7 | 4 |
| max(a, b²) | 7 | 5 |
| a²+b (unrelated) | 4 | 4 |

Switching memory off restores the empty-memory numbers exactly, so this is causal.

## 3. How an answer is produced

```
text ─parse→ evidence rows ((x,y)→z)
  depth 0 : exact fit of 68 written families + admitted stored concepts
  depth k : construct features from {x,y,1,2,+,−,×,÷,ifgt} and stored concepts
            (≤3 / 5 / 7 tokens), fit each feature's linear closure exactly
  H level : value-of-computation controller chooses propose / derive / verify / ask / stop
  answer  : only if the best reading beats "outside my language", and the dual solve Aᵀλ = r
            re-derives the *same* rational number
  else    : abstain with the reason, or ask the teacher (planning up to 3 questions ahead)
```

Output is one of **answer** (with the reading that produced it, how many held-out examples it
predicted, and how many questions it asked), **abstain** (with the reason), or a **question**.

## 4. What it outputs — held-out results ([RESULT] E19, 81 tasks, 26 operators, 9 never trained)

| condition | correct | wrong | abstain |
|---|---|---|---|
| **v0.6, trained** | **60** | **0** | 21 |
| v0.6, untrained (= memory off) | 44 | 0 | 37 |
| v0.5 + ADR-012 (no construction, no memory) | 18 | 0 | 63 |
| v0.6, closed world (no "outside my language") | 63 | 12 | 6 |
| nearest-evidence-row matcher | 5 | 76 | – |
| least squares on (1, x, y, xy) | 15 | 66 | – |

By category (correct/wrong/abstain): written 16/0/2; constructible 20/0/4; novel-never-trained
10/0/5; composite 14/0/1; **outside the primitives (gcd, mod, digit-sum) 0/0/9 — all refused**.
Mean operations rise with difficulty: ≈ 2.5 k (written) → 31 k (constructible) → 152 k (novel) →
229 k (unsolvable).

Other measured behaviours: 8/8 identical answers *and* identical stored concept when the same
evidence arrives through six sentence templates with new symbols; 6/6 correct revision when later
evidence contradicts an earlier concept; a learned base-7 numeral reader accepted only after exact
verification.

**Examples** (verbatim output of `python -m hrm_x chat` on a model trained by `python -m hrm_x train`):

```
> 2 ⊕ 3 = 3; 4 ⊕ 1 = 4; 5 ⊕ 5 = 5; 1 ⊕ 6 = 6; 8 ⊕ 2 = 8; 7 ⊕ 2 = ?
7 ⊕ 2 = 7; I used the concept 'c0' I formed earlier (4 examples predicted) and the dual check
re-derived the same number.

> 4 ★ 6 = 2; 9 ★ 6 = 3; 5 ★ 7 = 1; 8 ★ 12 = 4; 10 ★ 15 = ?      (this is gcd)
I do not know 10 ★ 15: not derived: the expected cost of deriving exceeds the value of an answer.
```

Concepts are called `c0, c1, …` until a person names them (`call ⊕ larger`). The refusal reason
for gcd is the controller's own ("not worth deriving"), not a statement that the operator lies
outside the primitives — the writer does not yet phrase that distinction; it is a known
explanation gap, not a correctness gap.

## 5. Comparison with the original HRM

### 5.1 What the original HRM is ([FACT], Wang et al. 2025 / `sapientinc/HRM`)

A ~27 M-parameter recurrent model with a fast low-level and a slow high-level module, trained by
backpropagation on ~1,000 examples with deep supervision, an outer refinement loop with an
adaptive-halting head, and data augmentation; it predicts grid/puzzle answers token by token.

Independent findings ([FACT]):
- ARC Prize's ablation (Aug 2025): the hierarchical H/L structure gave **≲ 5 points** over a
  same-size transformer; the **outer refinement loop** and **training on the evaluation tasks**
  drove most of the score (41 % → 31 % pass@2 when the eval-task data was removed).
- Jolicoeur-Martineau, *Less is More* (arXiv 2510.04871): HRM's fixed-point / one-step-gradient
  justification is not satisfied in practice; a single 7 M-parameter recursive network with full
  backprop through the recursion beat it (45 % ARC-AGI-1).

### 5.2 Side by side

| dimension | original HRM | HRM-X v0.6 |
|---|---|---|
| substance | neural network, 27 M learned weights | exact symbolic program; **no learned reasoning weights** |
| what is "learned" | weights, by gradient descent, from ~1 k examples per task family | concepts (named, verified building blocks), by admission after two verifications |
| how a new task is handled | forward recurrence; ARC needs test-time-like training on the eval tasks | per-task exact solve from 2–7 examples; no per-task gradient training |
| hierarchy (H/L) | two recurrent modules at two timescales | L = exact fit/saturation, H = value-of-computation controller; a *control* hierarchy, not recurrent state |
| more computation | outer refinement loops (ACT halting head) | deeper construction, planned questions; ops grow with difficulty (E19) |
| when it does not know | outputs a grid regardless | **refuses**, states why; 0 wrong of 81 on held-out |
| guarantees | none on answers | every verified answer is re-derived by an independent dual solve over ℚ |
| interpretability | latent states | the answer names the reading, its evidence, and the trace |
| extrapolation | limited to training distribution and augmentations | exact at queries ×5 the evidence range and at fractions, *within its operator class* |
| data need | large synthetic sets per domain | 2–7 examples per task; a 34-task curriculum builds the concept library |

### 5.3 What is *not* comparable — read this before quoting anything

- **There is no head-to-head benchmark.** The original HRM is scored on ARC-AGI grids, Sudoku,
  and mazes. HRM-X v0.6 is scored on binary operators over rationals. Neither was run on the
  other's tasks, so **no claim of "better than HRM" is supported by this repository.**
- HRM-X's own earlier neural attempt (HRM-EQB, now `legacy/`) scored **ARC 0.000 and
  Sudoku-Extreme 0.000** [RESULT, ADR-006]: the project has *not* reproduced HRM's grid results.
- HRM-X's safety comes from a *given* primitive alphabet and exact arithmetic; that is exactly
  what a general puzzle solver does not have. The comparison is between two answers to different
  questions: "learn a mapping from a few thousand grids" (HRM) vs "infer an operator exactly from
  a handful of examples, or say you cannot" (HRM-X).
- One training seed and 81 held-out tasks: differences of 1–2 tasks are not significant.

### 5.4 Where each design is stronger

- **HRM:** open-ended pattern domains where no symbolic language exists; it produces an answer
  for grid tasks HRM-X cannot even parse.
- **HRM-X:** exactness, refusal, interpretability, transfer of an abstraction (E17), and tiny data.

## 6. Limits that remain

1. The primitive alphabet {x, y, 1, 2, +, −, ×, ÷, if-greater} is hand-written; only *families*
   are no longer given. Inducing primitives is the next research question.
2. One nonlinear feature per family (plus linear terms).
3. Language is template-bound (ADR-009); anything else is refused.
4. Recall on 2–3-row tasks fell after calibration (fewer wrong answers, more refusals).
5. The calibration rules were set by hand from measured failures; the system did not discover them.
6. Domain is binary operators over ℚ; nothing here says it transfers to grids, vision or open text.

## 7. Reproduce

```
python -m hrm_x create models/hrmx && python -m hrm_x train models/hrmx
python -m hrm_x info models/hrmx
python -m experiments.x_understanding      # E19 (~20 min); x_transfer = E17
python -m pytest -q                        # 413 passed, 1 skipped (native DLL built)
```
