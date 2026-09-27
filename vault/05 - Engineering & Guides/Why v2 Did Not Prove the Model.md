---
title: "Why v2 Did Not Prove the Model"
type: analysis
tags:
  - architecture
  - ablation
  - measurement
  - negative-result
  - fherr
  - equilibrium
agent: "opencode/big-pickle"
model: "big-pickle"
date: 2026-09-27
status: current
related:
  - "[[ADR-004 Discriminating Benchmark and Reasoning Ceiling]]"
  - "[[ADR-003 Unified Cognitive Model]]"
---

# Why v2 Did Not Prove the Model

Benchmark v2 scores 100% of 84 in-budget tasks with zero false accepts, and the
ablation ladder is **flat**. This note explains why, mechanically, with the
measurements behind each claim. The short version:

> The architecture makes no prediction that brute-force enumeration does not also
> satisfy. Every path to an answer terminates in the same exact-verification
> oracle, the candidate set is small enough to enumerate, and the one component
> that could not be replaced by enumeration is provably dominated by a cheaper
> one. A benchmark can only falsify a claim it can make to fail.

## Measured facts

All numbers from `data/benchmark/benchmark-v2.json`, 108 tasks, 84 in-budget.

| configuration | solve rate | notes |
|---|---|---|
| full model | 1.000 | `fast=23 synth=61 abstain=24`, `deep=0` |
| − H-engine | 1.000 | **bit-identical to full** |
| − H-engine − synthesis | 0.274 | fast path alone, real phasor |
| − H-engine − synthesis, random ranking | 0.079 | mean of 5 seeds, max 0.107 |
| synthesis ranking oracle → constant | 0.000 | fast *and* synthesis both die |
| fast_top_k = 8, refresh_top_k = **6** | — | the H-engine's window is narrower |

Retrieval quality of the phasor ranking, over the 60 depth-1 tasks: the exact
program lands in the top-8 for **23**, beyond top-8 for **37**, and at ranks
**7–8 for 0**. Median rank of the exact program: **22** of ~32 candidates.

## Five causes, in order of how much they matter

### 1. The H-engine is dominated by construction, so it never runs

`EquilibriumEngine.h_refresh` scans the top `refresh_top_k = 6` candidates by
phasor score. `_fast` already scanned the top `fast_top_k = 8` **of the same
similarity function**, and both gate acceptance on the identical exactness test.
A 6-wide window of a ranking whose 8-wide prefix was already searched cannot
surface anything new.

Confirmed empirically: `deep` never appears in the status mix, in 108/108 tasks,
and the full-model and no-H-engine runs are identical. This is not a bug and not
a tuning problem. **As configured, `status="deep"` is unreachable.** The
deep engine is dead code.

The resonator perturbs the ranking through `z_h`, so in principle it could
promote a candidate the fast path missed. It never did — and the rank
distribution above says why: the exact program is at median rank 22, so
reshuffling a top-6 window does not reach it.

### 2. Acceptance requires exactness, so there is no gradient to descend

`Reasoner._best_verified` runs the expensive verification **only if
`energy == 0.0`**, and `Verifier.energy` returns `0.0` exactly when the program
reproduces every demonstration. So a program is either bit-exact and accepted, or
ignored. The loop cannot trade approximation for progress.

This is the same property that made the controller livelock (fixed earlier this
session): a landscape that is flat everywhere except a single spike gives an
iterative method nothing to descend. It is the deep reason the H-level is
inert — not that it is too small, but that **exactness makes the objective
non-gradient**.

### 3. The synthesizer carries the work, and it is guided by a hand-written oracle

Removing synthesis drops 1.000 → 0.274. It is the workhorse. But it beams on
`verifier.energy`, which is `1.0 − mean(demo_similarity)` where `similarity` is a
per-domain function the domain author wrote. For `vector3d` it is a cosine term
plus a per-component exponential closeness term — a metric that already encodes
the shape of the answer.

Ablating that oracle to a constant collapses the model to **0.000**, with `fast`
*and* `synth` both at zero (the constant makes `energy == 0.0` never true). So
the reasoner is not searching: it is being told, by a heuristic written by the
same person who generated the task, which programs look right.

### 4. The phasor is informative, but it is asked a trivial question

This is the one genuinely positive result, and it deserves to be stated as such:
with the fast path alone, real phasor ranking solves **0.274** against **0.079**
for random ranking (best of 5 seeds: 0.107). The substrate carries real
retrieval signal — roughly 2.6× the best random seed.

But look at what it is being asked: rank ~32 candidates and return 8. Brute force
over 32 items is free; a dense matmul on 1024–4096-dimensional phasors is not.
FHRR's advantage is supposed to appear when the candidate set is too large to
enumerate, and 32 is not that. The substrate is being used as an expensive
`argsort`.

### 5. The architecture has no division of labour

Three routes to the same oracle: fast path (top-8, phasor), H-engine (top-6,
phasor + perturbation), synthesizer (beam 8 × depth 3, domain oracle). All three
call `verifier.verify`. The most expensive is provably dominated by the cheapest;
the most informative is confined to the cheapest's job; the only one that does
real work is guided by a metric outside the substrate. Remove any one and the
others absorb the loss, which is exactly why every ablation is flat.

## What would make the claim falsifiable

Each of these creates a situation where brute force *cannot* also succeed, so
the benchmark can discriminate. All are ADR-worthy; none taken unilaterally.

1. **Grow the vocabulary past the synthesizer's reach.** The synthesizer sees 32
   candidates with beam 8 and depth 3. At 10³–10⁴ candidates its beam is a
   vanishing fraction, and the phasor's ranking becomes the only route. The
   falsifiable prediction is a *curve*: solve rate versus |V| should stay flat
   for phasor and decay as 8/|V| for random. That single plot would settle
   whether the substrate earns its cost.
2. **Give the H-level a compositional job.** Today it re-ranks a fixed set. Make
   `z_h` *propose* programs (so it can reach depth > 1), and set
   `refresh_top_k > fast_top_k` so iteration can widen a window the fast path
   cannot see. Until it can do something the cheap path cannot, it should be
   gated off rather than paid for.
3. **Replace the domain `similarity()` oracle with a phasor-derived energy.** If
   the reasoner may consult a metric that knows the answer, no ablation of the
   substrate can ever show value, because the substrate is not what is doing the
   discriminating. This is the deepest of the three: it is a design question
   about what the system is *for*.
4. **Report `deep`-status rate as a first-class metric.** `deep=0` was visible
   for the whole of v2 and only became meaningful after the ablation. A
   flagship component that never fires should show up as a failing metric, not
   as a status nobody reads.

## Honest summary

v2 did its job. It did not vindicate the architecture; it falsified a specific
claim ("the deep engine and phasor substrate carry the reasoning") and located
the reason in two measurable facts — `refresh_top_k < fast_top_k` and
`energy == 0.0` requiring exactness. Both are one-line changes to make. That the
flagship machinery currently costs 33s versus 15s for identical output is the
honest summary of its present value.

**Changes in this note:** measurement and analysis only; no code was modified to
produce these numbers. The `energy → constant` and `random ranking` ablations
were run as monkeypatches in throwaway scripts and are not committed.
