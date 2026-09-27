---
title: "Session: 2026-09-27 — HRM-EQB Theorems and Proofs"
type: session-log
tags:
  - session
  - design
  - proofs
  - hrm-eqb
date: 2026-09-27
status: n0-done
# before git; squashed into baseline commit 2c2675c
code: [hrm_eqb/algebra.py, hrm_eqb/certified.py, hrm_eqb/equilibrium.py, hrm_eqb/attnres.py, hrm_eqb/product_keys.py, hrm_eqb/neuromodulation.py, hrm_eqb/curiosity.py, hrm_eqb/gate.py]
---

# Session: 2026-09-27 — HRM-EQB Theorems and Proofs

## Request

The user asked for:
- `.hrmc` to be the full model, not a regeneration rule;
- provable mathematics, as a proper `.md` with the proofs checked;
- the name **hrm-eqb**, since the model rests on equilibrium;
- ARC solved by the model itself, with hormones and curiosity (it notices a missing
  pattern and asks), influenced by karpathy/autoresearch;
- Sudoku, ARC and language tasks, where the low level does general work and the high
  level thinks in language;
- text stored as rotation vectors (learn arXiv 2603.15031), and knowledge in a 4-D
  space;
- then the vault updated through graphify.

## Research

- **arXiv 2603.15031, Attention Residuals** (Kimi Team, Mar 2026), verified:
  - $h_l=\sum_i\mathrm{softmax}(w_l^\top\mathrm{RMSNorm}(v_i))\,v_i$, with the
    pseudo-query zero-initialized;
  - Block AttnRes costs $O(Nd)$ and matches a baseline trained with 1.25× more compute;
  - it is depth-wise attention, **not** rotation-vector text.
  - An earlier note misattributed the 1.25× figure to RealFormer; this is corrected in
    [[Concept - Attention Residuals (RealFormer & AttnRes)]].
- **4-D knowledge:** quaternions (QuatE lineage) give non-commutative, invertible,
  isometric relations. $SO(4)=\{x\mapsto px\bar q\}$.
- **Hormones:** Doya 2002 (DA, 5-HT, NE, ACh as meta-parameters).
- **Curiosity:** BALD/EIG, which equals vote entropy for a deterministic committee.

## Key decisions

1. **Contraction by construction replaces measured contraction.**
   - Cayley-orthogonal × bounded-sigmoid diagonal weights have an exact spectral norm.
   - Sub-stochastic routing has $\|A\|\le1$.
   - The radial ReLU on quaternions is 1-Lipschitz.
   - Together, four budgets decide convergence:
     $\Gamma=v_h+v_ls_h/(1-s_l)<1$.
2. **Nested equilibrium theorem.** The HRM L/H schedule converges to one equilibrium for
   every number of L-steps, at rate $r_n$ (the Perron root of $M_n$), with $r_n\to\Gamma_A$.
   This fixes the 1/r-Lipschitz gap in PEM's convergence argument.
3. **Halting is a theorem** (the a-posteriori bound). The 1-step gradient is informative
   only when $r<1/2$.
4. **`.hrmc` carries every weight.** The loader recomputes the certificate from SVD norms
   and never trusts the header.

## Built and verified

- `hrm_eqb/` (torch): `algebra`, `certified`, `equilibrium`, `attnres`, `product_keys`,
  `neuromodulation`, `curiosity`, `gate`.
- `tests/test_eqb_theorems.py`: 23 tests, one or more per theorem, in float64. They
  include a sharpness test showing the certified constants are attained.
- Full suite: **243 passed, 1 skipped**.
- Mutation probe: removing the sub-stochastic normalization (ratio 1.52 vs bound 0.48) and
  a gain-1.5 ReLU (1.21 vs 1) are both caught.

## Files

- [[HRM-EQB Specification v0.2 - Theorems and Proofs]] (new; supersedes
  [[HRM-Phi Design Document v0.1]])
- [[ADR-005 Neural HRM-Phi Replaces Program Search]]: resolved decisions and revised plan
- `hrm_eqb/*.py`, `tests/test_eqb_theorems.py`, `pyproject.toml` (package plus the
  `eqb` extra)

## Open / next

- N1: task field and prelude (routing $A_s$ from input, rotation-vector positions,
  augmentation).
- Not proved and flagged in spec §11: learning, generalization, the expressivity cost
  of $\Gamma<1$ (ablation A1′), and convergence across segments.
