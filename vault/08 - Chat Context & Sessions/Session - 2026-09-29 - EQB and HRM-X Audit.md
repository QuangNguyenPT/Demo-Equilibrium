---
title: "Session - 2026-09-29 - EQB and HRM-X Audit"
type: session
tags:
  - session
  - audit
  - hrm-eqb
  - hrm-x
date: 2026-09-29
commits: []
code:
  - hrm_x/mind.py
  - hrm_x/concepts.py
  - hrm_x/domains/concept.py
  - hrm_eqb/tasks.py
  - hrm_eqb/train.py
  - experiments/eqb_probe.py
  - experiments/eqb_text.py
  - experiments/x_margin.py
  - tests/test_x_concept.py
  - tests/test_eqb_model.py
---

# Session — 2026-09-29: EQB and HRM-X audit

**Goal.** Reconstruct what the project actually computes, audit HRM-EQB's learning path
mathematically and experimentally, execute ADR-011 in its stated order (measure A first), and
correct what the evidence says is wrong. Decision record:
[[ADR-012 EQB and HRM-X Audit - Linear Core, Recalled Language, Live BOTTOM]].

Built on branch `feat/hrm-x-v04` at `dec39b3` (ADR-011), merged into the audit branch first so
nothing here contradicts the newest docs.

## Baseline

`python -m pytest` → 431 passed, 6 skipped (5 of them native-parity tests, the DLL is not built
in a fresh worktree; after `.\scripts\build_native.ps1` they pass).

## Information path

| question | research | decision | implementation | test / experiment | result |
|---|---|---|---|---|---|
| Is EQB's recurrence doing nonlinear computation? | spec v0.2 §3 remark; APPNP | record as finding | — | `eqb_probe` P1; `test_core_fixed_point_is_linear…` | linear to 6.6e-8 at shrink 0 **[D]** |
| Does more recurrence help? | ARC Prize HRM analysis; TRM | record | — | `eqb_probe` P2 | flat: 1 cycle ≈ 40 cycles **[D]** |
| Is the language score a rule or recall? | — | add opt-in `text_source` | `hrm_eqb/tasks.py`, `train.py` | `eqb_probe` P4; `eqb_text` | 0.956 words vs 0.006 unseen **[D]**; EQB-T below |
| Does EQB read its demonstrations? | — | record | — | `eqb_probe` P3 | swap demos → 0.000 **[D]** |
| ADR-011 A: does the rival margin separate? | doc 13 P3 | falsified | `concepts.rival_margin` (measurement) | `x_margin` E14 + held-out | wrong answers at ∞ bits **[X]** |
| Why was the open-world ablation inert? | spec v0.5 Def. 3 | fix | `OUTSIDE` in `mind.py`, BOTTOM in `concept.py` | `bits_bottom` 4/∞/−100 → 0/52 changed; new tests | out-of-language wrong 24 → 2, in-language correct 51 → 33 **[D]** |
| ADR-011 B, C | — | not built | — | code search | no consumer / no runtime caller **[D]** |
| Crash on no-fit evidence | — | fix | `Concept.propose` returns `None` | `test_evidence_no_family_fits…` | fixed **[D]** |

## Key decisions

1. **BOTTOM counts inside `Mind`.** Spec v0.5 Def. 3 was documented, tested at the `fit()` level
   and dropped at the `Mind` boundary. It is now a hypothesis with answer `OUTSIDE`: it keeps its
   share of the mass, is never an answer, and in question valuation spreads likelihood `2^-log2|Y|`
   over every outcome. Shipped default stays the spec's `bits_bottom = 4`; the sweep is reported,
   not tuned.
2. **The rival margin is not implemented as a gate.** ADR-011 said to measure it first; the
   measurement falsified it.
3. **`min_surplus` stays opt-in (default 0).** It separates on held-out data but only as a filter,
   because the controller cannot value a question nobody disagrees about.
4. **No change to the EQB architecture.** ADR-011 §5 and ADR-007 stand; the audit adds *why* the
   certified core cannot reason (linearity at shrink 0) and *what* it learned instead (conditional
   recall). The only EQB code change is an opt-in data source for the experiment.

## Research

- ARC Prize, "The Hidden Drivers of HRM's Performance on ARC-AGI" (2025-08-15): hierarchy ≈
  same-size transformer (~5 pp); outer refinement loop drives the gain.
- Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks", arXiv 2510.04871:
  HRM's fixed point is not reached; true fixed-point iteration generalised worse.
- Fung et al. JFB (AAAI 2022) / Geng et al. phantom gradients (NeurIPS 2021): EQB's truncated
  gradient is sound at r = 0.30 — the ceiling is representational, not optimisation.

## EQB-T

Text-only training, 20 min per arm, one seed (`experiments/results/eqb_text.json`): on unseen
strings `words` 0.000, `letters` 0.140, `letters` + shrink 0.1 0.200; all arms 0.000 at longer
lengths. What is learned without recall is a per-position map (upper 0.80–1.00); reverse, sort,
caesar and drop_vowels stay at 0.00. Full table in ADR-012.

## Open questions / next steps

1. Calibrate `bits_bottom` from experience; give the controller a *test question* action for a
   lone untested reading (VOI is 0 today).
2. If EQB is ever revisited: shrink > 0 plus letter-string data was the minimal pair to try, and
   EQB-T says it is not enough — only per-position rules are learned, no length generalisation.
   Multiple seeds and longer budgets are needed before calling the shrink gain real.
3. ADR-011 C becomes relevant the day `Mind.sleep` calls `consolidate`.
