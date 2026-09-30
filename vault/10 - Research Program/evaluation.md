---
title: Evaluation — the understanding suite (E19) and what it shows
type: research-evaluation
tags: [research, evaluation, understanding, v0.6]
created: 2026-09-30
related: ["[[understanding_definition]]", "[[ablation_results]]", "[[ADR-013 Constructive Concept Induction]]"]
---

# Evaluation (all numbers [RESULT], `experiments/results/`)

## E19 — held-out understanding suite (`x_understanding.json`)

Training: the curriculum (`hrm_x/curriculum.py`, seed 0, 34 tasks: written → constructible →
composite) with a teacher. Test: seed 1, **81 tasks, 26 operators, 9 never seen in training**,
evidence 4–6 rows over 1..9, queries outside that range (to 45, and halves). Memory is restored
before every task, so the test cannot teach itself.

| category (n) | v0.6 trained | v0.6 untrained | v0.5 + ADR-012 | nearest row | least squares |
|---|---|---|---|---|---|
| written (18) | 16 / 0 / 2 | 16 / 0 / 2 | 16 / 0 / 2 | 0 / 18 | 13 / 5 |
| constructible (24) | 20 / 0 / 4 | 10 / 0 / 14 | 0 / 0 / 24 | 0 / 24 | 0 / 24 |
| novel, never trained (15) | 10 / 0 / 5 | 8 / 0 / 7 | 2 / 0 / 13 | 1 / 14 | 2 / 13 |
| composite (15) | 14 / 0 / 1 | 10 / 0 / 5 | 0 / 0 / 15 | 1 / 14 | 0 / 15 |
| outside the primitives (9) | 0 / 0 / 9 | 0 / 0 / 9 | 0 / 0 / 9 | 3 / 6 | 0 / 9 |
| **total (81)** | **60 / 0 / 21** | 44 / 0 / 37 | 18 / 0 / 63 | 5 / 76 | 15 / 66 |

(correct / wrong / abstain; baselines always answer.) Trained model: 35 questions asked,
25 answers from a construction made on the task, 17 from a stored concept.

## The properties of [[understanding_definition]], one by one

| # | property | evidence | verdict |
|---|---|---|---|
| U1 | representation abstraction | 8 tasks × 6 sentence templates × new symbols × shuffled row order: 8/8 identical answers, 8/8 identical concept (E19 surface) | **shown** at template level; free text not tested (ADR-009) |
| U2 | compositionality | composites 14/15 trained vs 0/15 with library tokens off; E15b: `max(a, b²)`, `max(a², b)` reachable only through a stored concept within 80k evaluations | **shown** for one level of composition |
| U3 | role binding | regime coverage refuses asymmetric readings across unseen sign(x − y); symmetric ones answer (test); argument-order shuffles in U1 change nothing | **shown** for two roles |
| U4 | systematic generalisation | every correct answer is exact at queries up to ×5 the evidence range and at halves; baselines that interpolate: 5/81 and 15/81 | **shown** within the operator class |
| U5 | persistent / adaptive computation | mean ops: written 2.5k, constructible 31k, novel 152k, outside 229k; the written category never constructs (test) | **shown**: compute follows difficulty; failure is the most expensive case |
| U6 | error correction | evidence first consistent with max, then contradicting it: revised 6/6 (E19 revision) | **shown** for one revision |
| U7 | abstraction transfer | E17: with `max` stored, rows for ≥ 75 % correct: \|a−b\| >7 → 5, relu >7 → 5, 2max+1 7 → 4, max(a,b²) 7 → 5; unrelated a²+b 4 → 4; memory off = empty exactly | **shown** (causal) |
| U8 | concept formation | 25 held-out answers from concepts constructed on the task; 3 concepts admitted during training (max, x², x/(x+y)) | **shown** over a given primitive alphabet |
| U9 | calibrated uncertainty | 0 wrong of 81 (v0.6); outside-the-primitives 9/9 refused; closed world gives 12 wrong | **shown** on this suite |
| U10 | causal use of internal state | memory off: 60 → 44; library tokens off: composites 14 → 0; construction off: 60 → 35; BOTTOM off: 0 → 12 wrong | **shown** for each mechanism claimed |

## Other records

- **E13** (`x_v05.json`, the thin-evidence v0.5 harness, 2–3 rows): E13a 7/0/14, E13b 1/0/4 — zero
  wrong (v0.5 as ported: 10 wrong; ADR-012: 2 wrong), at the price of recall on 2-row tasks.
- **E14** (`x_margin.json`): rival margin never separates right from wrong; `bits_bottom` = 4 gives
  E13 8/0/18 and 61 held-out operators 23/1/37.
- **E17** (`x_transfer.json`): see U7 and [[ablation_results]].

## What this evaluation does not show

- Nothing beyond binary operators on rationals; no open language; no perception beyond numerals.
- The primitive alphabet is given; concept formation is over it.
- One seed of training, one held-out seed; the suite is small (81 tasks). Differences of 1–2
  tasks between conditions are not significant.
- "Understanding" is not claimed; the properties above are.
