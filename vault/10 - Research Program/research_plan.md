---
title: Research Plan — from selecting families to constructing concepts
type: research-plan
tags: [research, hrm-x, v0.6, concept-formation]
status: active
created: 2026-09-30
related:
  - "[[hypotheses]]"
  - "[[understanding_definition]]"
  - "[[architecture_spec]]"
  - "[[biological_inspiration]]"
  - "[[experiment_plan]]"
  - "[[ADR-012 EQB and HRM-X Audit - Linear Core, Recalled Language, Live BOTTOM]]"
---

# Research plan (living document; evidence changes it)

Evidence tags: **[FACT]** external, **[RESULT]** measured here, **[HYPOTHESIS]** untested,
**[SPECULATION]** exploratory.

## 1. Where the project actually stands (reconstructed 2026-09-30)

| component | state | evidence |
|---|---|---|
| `hrm_x/` concept route (v0.5) | exact law-closed families over ℚ, dual-solve verification, BOTTOM live since ADR-012 | [RESULT] E13, E14 |
| hypothesis language | **68 hand-written families**; anything outside is refused or (before ADR-012) guessed | [RESULT] 0/52 BOTTOM sensitivity pre-fix |
| concept memory | v0.5 stored version spaces nobody read back | [RESULT] ADR-012 E9 |
| `legacy/hrm_eqb/` | the only learned model; linear core at shrink 0, language score = recall | [RESULT] ADR-012 E1–E4, EQB-T |
| FHRR substrate | binding/unbinding + characters (`hrm_x/algebra.py`); used only by capacity experiment E1 | code audit |
| connectome principles (doc 14) | N1 MBI, N2 ETL derived from the local MaleCNS data, **never integrated** | delivery doc 14 |

## 2. The chain this plan follows

| limitation | suspected cause | question | analogy | mechanism | test |
|---|---|---|---|---|---|
| L1 cannot acquire a concept outside the 68 families | the family is the unit of hypothesis, and families are written by hand | can features be *built* from a small primitive set and the family obtained by linear closure? | MB: KCs are random conjunctions of ~5 inputs; learning selects readouts | construction (`hrm_x/construct.py`) | E15 out-of-language operators, confident-wrong rate |
| L2 nothing learned is reused | memory stored readings, never features | does storing a verified construction as a primitive make related concepts cheaper **and need less evidence**? | MB M3: one sparse code, ~9 gated readouts per KC — new association = new readout | concept memory + library tokens | E17 transfer: evidence rows / ops to acquire a related concept, memory on vs off |
| L3 search cost grows with the library | construction enumerates by size | can a learned index order candidates so that ops fall **without changing answers**? | MB index (N1); Kimi K3: routing bias steers selection only, output byte-identical across memory budgets | FHRR signature → sparse kWTA expansion → one-shot readouts (`hrm_x/mbi.py`) | E16 ops to first verified reading; answer identity vs MDL order |
| L4 learned representations memorise (EQB) | the objective rewards any map; recall is cheapest | does learning help when it can only *order* exact hypotheses? | — | router trained on dreams from stored concepts | E16 held-out signatures; shuffled-wiring and no-kWTA ablations |
| L5 language is a front end only | the parser is template-bound (ADR-009) | can surface forms map to one concept by behaviour, and can a word bind to a constructed concept? | — | `hrm_x/language.py`: surface parser, symbol-free concept identity, naming | E18 surface-variation invariance, name → concept retrieval |
| L6 myopic questions | one confirmation cannot overturn BOTTOM | does an h-question horizon make asking rational when it should be? | — | `Controller.ask_horizon` | E15 with/without teacher |

## 3. Priority (highest-risk assumption first)

1. **Is a fixed hypothesis language sufficient?** No by construction (L1). Test whether
   construction escapes it without re-opening the confidently-wrong channel ADR-012 closed.
2. **Does stored structure transfer?** The claim that separates "concept formation" from
   "bigger search". Without it, construction is v0.4's enumeration again.
3. **Does learning order, or only memorise?** Learning is allowed only where it cannot change an
   answer (ordering), so a memorising router costs ops, never correctness.
4. Language and biology come after 1–3 and are kept only where they change a measurement.

## 4. Decisions already taken from evidence

- **Equilibrium core not revived** — linear at shrink 0, compute-flat (ADR-012 E1–E2); TRM found
  forced fixed points generalise worse [FACT arXiv 2510.04871].
- **EQB's rotation binding is kept** as the FHRR evidence signature (role ⊛ character codes);
  its trained weights are not (recall, ADR-012 E3).
- **Construction is exact downstream** — every constructed feature is re-evaluated in `Fraction`
  and fitted by the same exact solve; floats only prune.

## 5. Status log

- 2026-09-30: construction integrated; E13 out-of-language 0/0/6 → 2/0/4 [RESULT]; in-language
  correctness unchanged, ops ×20 when the mind tries to build before abstaining [RESULT].
- 2026-09-30: M2 (random sampling) and M4 (learned router) rejected (E15b, E16) [RESULT].
- 2026-09-30: E17/E19 wrong answers forced three calibration rules (regime coverage, empirical |Y|,
  distinct surplus); final E19 60/0/21 vs 18/0/63 for v0.5 + ADR-012 [RESULT]. L1–L3, L5, L6
  addressed; L4 answered negatively (learning does not help ordering here). Next targets: a given
  primitive alphabet, one nonlinear feature per family, template-bound language.
