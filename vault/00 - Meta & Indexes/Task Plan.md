---
title: "Task Plan: verified state and next gates"
agent: "opencode/stealth/space-bunny-alpha"
model: "space-bunny-alpha"
date: 2026-09-29
status: active
---

# Task Plan

Canonical plan for this project. It **supersedes** the plan tables in
[[ADR-005 Neural HRM-Phi Replaces Program Search]] (phases N3–N8, written for
`hrm_eqb/` before the pivot) and the open questions in
[[ADR-006 HRM-EQB Training and Benchmark Results]]. The normative gate table
remains [[HRM-X Specification v0.5 - Structural Operator Inference]] §10; this note is the
operational version of it, checked against the code. (Section 1b below records what the
v0.5 port changed; sections 1 and 2 are the v0.4 state they were written against.)

## 1. Verified state (2026-09-28, measured not asserted)

| Generation | Package | Size | Role |
|---|---|---|---|
| 1 | `hrm/` | 49 py, 6,055 ln | program-search baseline, ADR-003 |
| 2 | `hrm_eqb/` | 23 py, 2,966 ln | HRM‑Φ equilibrium, ADR-005/006, learned patterns but no arithmetic |
| 3 | `hrm_x/` | 12 py, 3,493 ln + C++ | **current**, ADR-008, operator inference |

(Lines counted per file, UTF‑8, excluding `__pycache__`; `hrm_x/native/__init__.py`
is the ctypes binding and is included. Recounted 2026‑09‑28 after the DSL and
perception work — the previous 11 py / 2,349 ln undercounted.)

- Tests: **436 passed, 1 skipped** (`python -m pytest`, 2026-09-29).
- Native backend built: `hrm_x/native/bin/hrmx_native.dll`, sources
  `egraph.cpp` / `grid.cpp` / `hrmx.h`. Parity `identical_results: True`;
  grid 3.79×, arithmetic 1.47×.
- `experiments/results/`: 12 result files (`x_v05.json` added 2026-09-29).
  Every number claimed in [[Session - 2026-09-28 - HRM-X v0.4 PoC and Native Backend]] was
  re-read from the JSON and matches: E1 1.000 on all four splits (kNN 0.41→0, MLPs 0.00),
  E2 1.000 at depths 1–3, E12 1000×3.1 1,021→192 ops, curiosity 1.0 question
  vs 4.8 random and 0.86 vs 0.71 later accuracy, volatility 2 vs 6 wrong,
  grids 0.984/0.931 (seed 0) and 1.000/0.833 (seed 1).
- Spec v0.4 §24 gates **G1–G6 all done** — for v0.4. Under spec v0.5 §10, **G4 and G5 are
  not met**; see 1b.

## 1b. Where v0.5 leaves the plan (2026-09-29, measured not asserted)

The v0.5 research delivery was ported into `hrm_x/` and measured against the v0.4 route
through the **same `Mind`** and the same value-of-computation controller
(`experiments/x_v05.py` → `experiments/results/x_v05.json`, ADR-010). Nothing below is an
intention.

| | package |
|---|---|
| `hrm_x/` | 20 py, 4,339 non-blank lines + 545 of C++ |
| new v0.5 core | `qlinalg`, `concepts` (68 law-closed families), `consolidation`, `situation`, `algebra`, `operators`, `reader` (torch) |
| new domain | `domains/concept.py` — the same mathematics as a `Domain`, driven by `Mind.solve` |
| kept | `domains/arith.py`, `library.py` — the v0.4 program-search route, now the baseline |

| E13a, 21 queries | correct | wrong | abstain | mean ops |
|---|---|---|---|---|
| **concept v0.5** | **14** | 7 | 0 | **751.5** |
| arith v0.4 | 9 | 4 | 8 | 4247.2 |

- **G3 (capability) improved**: in-language 13/15 at 593 ops against 7/15 at 4265 ops —
  7.2× cheaper, because deriving is no longer a search.
- **G4 (generalization) regressed and is not met**: out-of-language 1/6 correct and
  **5 confidently wrong**, where v0.4 gave 2/6, 0 wrong, 4 abstentions. The mechanism is known
  and pinned — one question brings a second value of an argument into the evidence, so the
  open-world coverage rule stops firing and a *simpler wrong* family that fits exactly gets
  confirmed by the dual check (`::test_one_question_destroys_the_coverage_guard`).
- **G5 (anti-shortcut) is not met** for the same reason: this is the coincidental-fit channel
  (doc 13, Q1). The delivery's own README lists it as an honest limit.
- **G6 (efficiency) is not met**: metareasoning dominates object-level cost (35k meta ops
  against 3.3k object ops per hard query, doc 13, Q3) and nothing bounds the estimator.
- The hypothesis language is still **given**. `Concept.propose` returning `None` on its second
  call is the honest statement of that.

### Next, in priority order

1. **The rival-margin rule** (doc 13, P3) — the identified fix for the coincidental-fit channel.
   It is a design with its own evidence requirement and deserves its own ADR. Until it lands,
   G4 stays open.
2. **The strongest neural baseline** (Abacus, arXiv 2405.17399). Every comparison in E13 is
   symbolic against symbolic, and the v0.4 audit's falsifier (a) has still never been run.
3. **Library growth of new concept *kinds*** — still the project's most dangerous assumption.
   `consolidate` adds compositions, measured to reject a macro it cannot amortise.
4. Everything in section 2 below still stands: the DSL does not pay by default, free text does
   not transfer, event schemas are still given, and learned λ still loses to the fixed constant.

## 2. Next gates (in priority order)

Gates 1, 2 and 4 were measured on 2026‑09‑28 and are recorded as results, not
intentions. Two of the three did not come out the way the gate hoped, and that
is the useful part.

1. **Grow the DSL — MEASURED, and it does not pay by default.** `hrm_x/library.py`
   does real DreamCoder‑style MDL mining over the surviving version space and the
   expansion is exact. On a held‑out symbol it learns `add(a,x)` (6 uses, MDL
   40→38) and the compressed program is reachable one node sooner — but search got
   *more* expensive, 4,218→9,615 propose ops, because an arity‑1 constructor
   multiplies the programs at every size. Swept bounds 4–10, the extended language
   at `B−1` is 2–4× larger than the base one at `B`: the library never pays at
   one use per program. It does pay when the repeated structure is a large share
   — six nested steps earn 11 nodes of credit and beat the base language at every
   bound (8 programs vs 14–18,189). So `use_library_bound` defaults to **off**, and
   a present library no longer touches the search at all (it cost 9.4× when merely
   enumerated). Spec §26 falsifier (b) is **not** triggered, but only because the
   DSL was small; D4 scaling on ARC‑AGI is still unproven.
2. **Perception on free text — MEASURED, and it does not transfer.** Every earlier
   story split reused the training templates' surface forms, so "unseen text" was
   never tested. Scored against an independent hand‑written reader
   (`experiments/x_perception.py`), the top reading is exact 100% on seen forms,
   50% with only the verb changed, **0%** with only the clause order changed, 67%
   with the question reordered. 76–93% of held‑out stories contain a misread, and
   the answer is right ≤25% of the time. The tagger learned template *positions*,
   not a compositional reader. See [[ADR-009 Free Text Is Not Yet Supported]].
3. **Schema induction for events.** Still open. Currently given (TRANSFER,
   ACQUIRE, CONSUME), not learned — `[U]` in spec §17.
4. **A learning rule for patience (λ) — MEASURED, and it lost. Stays off.**
   `Patience` in `hrm_x/mind.py` tracks utility density with an EMA and bounded
   log‑space dual ascent, is anchored to `Controller.lam` as its only prior, and is
   priced into every propose/derive/verify/ask decision. 18 tests, two of which pin
   the measured defect. `experiments/x_patience.py` judged it against a fixed λ swept
   over patience's own reachable range [1e-7, 1e-1] — not just the default, so that
   "learned" could not mean merely "different" — on held-out arithmetic and novel grid
   streams. It reached −0.90 and −0.36 accuracy against the best fixed λ.
   The cause is structural, not a tuning miss: the fixed point is **average** value
   density `U/ops`, and VOC needs **marginal** gain/cost. Where they coincide the rule
   works (its synthetic test sits at 1.6e-4, at break-even). Where an episode is cheap
   they do not: arithmetic costs ~6 ops at U≈1, so the rule climbs toward 0.167 — four
   decades above the λ≤1e-4 band where the mind still acts — and then freezes, because
   `update` cannot take a density sample from a zero-op episode. The published default
   2e-5 sits safely inside the good band, so **nothing about the shipped E9 baseline
   changes.** A replacement rule needs marginal-value telemetry that does not exist yet;
   that is a design decision with its own record, not a patch.

### What the two failures have in common

Neither is a reasoner bug. In both cases the exact core behaved correctly on
what it was given, and the loss is in *representation*: which programs exist to be
searched (gate 1), and what text means (gate 2). Spec §26 falsifiers (b) and (c)
are both still live, and the measurement says the fix belongs in the vocabulary
and the reader, not in the search or the laws.

One measurement went against expectation and is worth keeping: the story verifier
**cannot** see a misreading, because it replays the reader's own reading through a
different representation. Yet only 2 of 91 misread stories produced a confident
wrong answer, because `reading_defect` rejects a reading that is missing a slot
and the mind abstains. The safe failure is a property of that guard, not of
verification — and the guard has now been found incomplete twice (a missing amount
reached `Fraction(None)`, then an owner‑less question reached `owner[0]`), so it is
now pinned slot by slot per schema in `tests/test_x_perception.py`.

## 3. Housekeeping debt found 2026-09-28

Fixed in this pass:
- Ignored the vendored `docs/kimi-k3-in-c-main/` tree (309 files, 115 images).
  It was being LLM-extracted, inflating the corpus to 213 non-code files and
  costing a ~30-minute run over a PDF's figures for zero project signal.
- Deleted three 0-byte `.md` debris files at the repo root left by an
  earlier root-level export.
- Replaced the malformed ignore glob `vault/(*.md` (unbalanced paren, matched
  nothing) that looked like protection and was none.

Open:
- ~~**Stale plan tables.**~~ **NOT OPEN — already done; this entry was stale.**
  Re-verified 2026-09-28: ADR-005 §"Plan (revised for HRM-EQB)" already carries a
  "Superseded 2026-09-28 by ADR-008" blockquote, and ADR-006 §"Open Questions for
  Next Phase" already carries a "Closed 2026-09-28" one naming #3/#4/#5 as
  describing a non-target package. Do **not** re-stamp these; the annotations are
  the fix, and removing them as "redundant" would resurrect the exact confusion
  they prevent.
- ~~**`Dashboard.md` is wrong.**~~ **NOT OPEN — already correct; this entry was
  stale.** Re-verified 2026-09-28: `vault/00 - Meta & Indexes/Dashboard.md` already
  names HRM-X v0.4 in `hrm_x/` as the live model (its note at the top, the subsystem
  table, and the architecture graph's "hrm_x/ — CURRENT (ADR-008, spec v0.4)"
  subgraph) and already marks the ADR-001 C++/Python design as never built and
  historical in `legacy/`. No edit needed; do not "fix" it.
- **`scripts/simplify_graph.py` is orphaned — FIXED 2026‑09‑28.** It is wired back
  into `scripts/update_knowledge_graph.ps1`: update → prune to the module/class
  skeleton (`graphify-out/graph.simple.json`, 2,933→554 nodes, 18.9%) and
  `graph.html` from the *pruned* graph → curated `vault/09 - Code Map/` → 651
  derived notes in the quarantined `vault/_graph-view/`. The full `graph.json` is
  never overwritten, so `query`/`path`/`affected` still get every symbol. Verified
  the loop guard holds: 0 graph nodes are sourced from `vault/_graph-view/`, and
  the 57 real vault notes are untouched. `graphify` prints an
  `.graphify_analysis.json is stale` warning during the export; that is expected
  (the sidecar describes the full graph, the export is handed the pruned one) and
  is documented in the script — do not "fix" it by exporting the full graph, which
  is the 1,922‑note vault‑root explosion.
- **The graph still holds 0 markdown nodes.** The `claude-cli` backend needs no
  key and works, but a semantic pass over the corpus is slow enough to look like
  a hang. Prefer targeted runs; a full default pass is not interactive-safe.
  Until it succeeds, `vault/00 - Meta & Indexes/Code Map.md` and
  `vault/09 - Code Map/` are the code↔doc bridge.
- Run `graphify cluster-only .` after an update: `graph.json` is rebuilt on every
  update but community labels are not, so the graph is unlabelled between runs.
