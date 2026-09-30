---
title: Hypotheses and competing mechanisms
type: research-hypotheses
tags: [research, hypotheses]
created: 2026-09-30
related: ["[[research_plan]]", "[[experiment_plan]]", "[[ablation_results]]"]
---

# Competing mechanisms for escaping the fixed hypothesis language

Each entry: problem it solves · why it might work · assumption · failure mode · discriminating test.
Status is updated from `[[ablation_results]]`; nothing here is a result until it says so.

## M1 — Exhaustive construction + linear closure + MDL/BOTTOM gate  *(implemented)*
- **Solves:** L1. Features are postfix expressions over {x, y, 1, 2, +, −, ×, ÷, ifgt}; a family
  is `a·g + (subset of x, y, 1)`, fitted exactly. The coefficients are solved, only skeletons are
  searched (v0.5's lesson applied one level up).
- **Why:** the skeleton space modulo observational equivalence and the base span is small
  (69 features ≤ 5 tokens, 1,348 ≤ 7) [RESULT].
- **Assumption:** the missing concept has a short skeleton over the primitives.
- **Failure mode:** coincidental constructions on thin evidence (the v0.4 channel again).
  Mitigation: a constructed reading pays its description length against BOTTOM, so it needs
  surplus rows it did not fit.
- **Test:** E15 — out-of-language operators: correct/wrong/abstain vs v0.5; wrong must stay ≈ 0.

## M2 — Random conjunctive sampling instead of enumeration  *(MB-inspired; to test)*
- **Solves:** reach beyond the exhaustive budget (size 9+ grows ~×20 per 2 tokens).
- **Why:** KCs are random conjunctions of ~5 inputs [RESULT M2]; random features cover a space
  without enumerating it.
- **Assumption:** useful concepts are dense enough among random skeletons of a size.
- **Failure mode:** low hit rate — random sampling finds nothing exhaustive search would not.
- **Test:** E15b — equal-budget comparison at sizes beyond exhaustive reach.

## M3 — Library compression (concepts as primitives)  *(implemented)*
- **Solves:** L2 and reach. A stored concept is one token.
- **Why:** DreamCoder's wake–sleep library learning [FACT, Ellis et al. PLDI 2021]; MB M3: new
  association = new readout of a shared code [RESULT].
- **Prediction:** a concept related to a stored one needs **fewer evidence rows** (its prior cost
  falls from ~21 to ~8 bits) and fewer ops.
- **Failure mode:** stored concepts bloat the alphabet (every token costs log₂ of a larger
  alphabet) and slow unrelated tasks.
- **Test:** E17 — memory on vs off on related and unrelated follow-up tasks.

## M4 — Learned ordering (mushroom-body index over an FHRR signature)  *(to implement)*
- **Solves:** L3; tests L4 safely.
- **Why:** doc 14 N1/X1 (kWTA raises effective dimension; one-shot readouts) [RESULT, delivery];
  Kimi K3's invariant "the routing bias steers selection only" [FACT, kimi-k3-in-c README].
- **Assumption:** evidence of the same concept at different points has similar signatures.
- **Failure mode:** the router memorises points (as EQB memorised words) → no ops reduction on
  held-out evidence. Harmless to correctness by design.
- **Test:** E16 — ops to first verified reading, MDL order vs router vs shuffled wiring vs no kWTA;
  answers must be identical across orders.

## M5 — Neural generator + snapping (ETL-style)  *(not implemented; recorded)*
- A network proposes the feature directly and a symbolic snap makes it exact. Rejected for now:
  with 2–6 evidence rows a generator is underdetermined, and M4 already tests whether learning
  helps where it cannot hurt. Revisit if M1–M4 plateau.

## M6 — More hand-written families  *(rejected on principle)*
- Moves the wall, does not remove it; also the null against which M1–M3 are measured (v0.5).
