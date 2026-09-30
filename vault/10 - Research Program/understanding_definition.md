---
title: Understanding — an operational definition
type: research-definition
tags: [research, evaluation, understanding]
created: 2026-09-30
related: ["[[research_plan]]", "[[evaluation]]"]
---

# "Understanding", operationalised

The word is never a claim in this repository. Each property below is a measurement with a
failure condition; a system "shows" the property only on the named test, at the stated scale.

| # | property | operational test | fails when |
|---|---|---|---|
| U1 | **representation abstraction** | same operator under different symbols, wordings, argument order, irrelevant context → same stored concept (behaviour key, not symbol) | a surface change alters the concept used or the answer |
| U2 | **compositionality** | a concept expressible only by composing two stored concepts is acquired in fewer tokens / less evidence than from scratch | no reduction, or wrong answers |
| U3 | **variable/role binding** | the evidence enters as role-bound codes (x, y, z); swapping argument order changes the answer iff the concept is asymmetric | order-blind answers on asymmetric concepts |
| U4 | **systematic generalisation** | queries far outside the evidence range (magnitude ×10–100, fractions, negatives) answered exactly | any interpolation-style error |
| U5 | **persistent computation** | more computation (deeper construction, more questions) changes the outcome only when the problem needs it; ops grow with difficulty | flat ops, or ops spent on easy tasks |
| U6 | **error correction** | later contradicting evidence revises the concept (refit, retire) instead of being ignored | the old concept is still used |
| U7 | **abstraction transfer** | a concept built on task A lowers ops and required evidence on structurally related task B | memory ablation leaves B unchanged |
| U8 | **concept formation** | a concept outside the written language is constructed, verified, stored and reused | refusal only, or a wrong answer |
| U9 | **calibrated uncertainty** | outcomes split into answer / abstain / ask, with confidently wrong answers on out-of-grammar concepts ≈ 0 | confident wrong answers |
| U10 | **causal use of internal state** | ablate construction / memory / BOTTOM / router → the predicted capability breaks, restore → returns | ablation changes nothing (mechanism decorative) |

Outcomes are always reported as **correct / wrong / abstain**, plus asks, concepts created and
concepts reused. Accuracy alone is never the headline.

What these tests cannot establish: that the system understands anything beyond binary operators
on rationals; that the primitive set is not itself doing the work (it is given — reported as such);
that any of this scales to open language.
