---
title: Ablation results and negative results (v0.6)
type: research-ablations
tags: [research, ablation, negative-result]
created: 2026-09-30
related: ["[[hypotheses]]", "[[evaluation]]", "[[ADR-013 Constructive Concept Induction]]"]
---

# Ablations (E19 held-out, 81 tasks; correct / wrong / abstain)

| condition | total | what it shows |
|---|---|---|
| **v0.6 trained (all mechanisms)** | **60 / 0 / 21** | reference |
| memory off (= untrained) | 44 / 0 / 37 | stored concepts carry 16 answers; composites 14 → 10, constructible 20 → 10 |
| construction off at test, memory kept | 35 / 0 / 46 | construction on the task carries 25; memory alone answers 17 constructible |
| library tokens off | 41 / 1 / 39 | composites 14 → 0: concepts-as-primitives is what reaches them |
| closed world (no BOTTOM) | 63 / **12** / 6 | BOTTOM is the safety: without it 7 of 9 out-of-primitive tasks get invented answers |
| question horizon 1 | 58 / 0 / 23 | the h-question plan buys 2 answers (novel 10 → 8) |
| no teacher | 43 / 0 / 38 | questions carry 17 answers |
| fixed \|Y\| = 2¹⁰ (v0.5 constant) | 67 / 1 / 13 | the calibrated answer space trades 7 answers for 1 wrong (gcd → 1 via `ifgt(x, y − 2, y, 1)`) |
| v0.5 + ADR-012 | 18 / 0 / 63 | the baseline v0.6 is measured against |

# Calibration fixes forced by measurement (each pinned by a test)

| fix | failure it removed | before → after |
|---|---|---|
| regime coverage (sign(x − y)) | 2x + 1 answering for 2 max(x, y) + 1 across the diagonal | E17 empty-memory wrongs 19 → 8 (fixed \|Y\|) |
| empirical answer space | coincidental 7-token constructions on 5 small-integer rows (gcd → 1, a mod b → 1) | E19 wrongs 5 → 2 |
| distinct surplus | 15·max − 54 "predicting" rows where max took two values (381 for 841) | E19 wrongs 2 → 0 |

# Negative results (hypothesis → test → result → likely cause → next)

| hypothesis | test | result | likely cause | alternative explanation | next |
|---|---|---|---|---|---|
| M2: random conjunctive sampling (MB) beats enumeration beyond the exhaustive budget | E15b, equal budget, 200 targets | enumeration 55.5 % vs random 9 % at 80k; random yields 4.6× fewer distinct classes | uniform tree growth re-draws small, common classes; enumeration never repeats a class | a better sampling distribution (size-stratified) could narrow the gap | not pursued: library compression reaches the targets that matter |
| M4: a learned MB index over FHRR signatures orders features usefully | E16, 40 classes, held-out dreams | top-1 5–7.5 % (chance 2.5 %), kWTA ≤ dense ReLU, routing ≈ ⅓ of exhaustive prefilter cost | the relational signature is not invariant to a·g + b·x + c·y + d | a learned invariant encoder might work, but exact prefiltering already costs little | revisit only if evaluation becomes expensive (much larger features) |
| anytime stop makes ordering answer-preserving and cheap | unit test | never fired | sum of remaining weight ceilings always exceeds the best reading | a tighter (data-dependent) bound | removed |
| fixed \|Y\| stays adequate as the language grows | E19 | 5 wrong with construction on | per-row evidence was overstated ~2× for small-integer answers | — | replaced by the empirical estimate |

# From ADR-012 (still standing)

EQB core linear at shrink 0; recurrence compute-flat; language score = recall (0.956 vs 0.006);
retrained on random strings only per-position rules are learned (EQB-T).
