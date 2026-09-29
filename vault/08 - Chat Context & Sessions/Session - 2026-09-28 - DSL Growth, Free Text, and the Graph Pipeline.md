---
title: "Session - 2026-09-28 - DSL Growth, Free Text, and Rewiring the Graph"
type: session
agent: "opencode/big-pickle"
model: "big-pickle"
date: 2026-09-28
related:
  - "[[Task Plan: verified state and next gates]]"
  - "[[ADR-009 Free Text Is Not Yet Supported]]"
  - "[[ADR-008 Operator Inference over Structured Situations]]"
  - "[[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]"
  - "[[Session - 2026-09-28 - HRM-X v0.4 PoC and Native Backend]]"
---

# Session - 2026-09-28: DSL Growth, Free Text, and Rewiring the Graph

Three gates closed, two of them negatively. Started from **355 passed, 1 skipped**;
ended at **404 passed, 1 skipped**.

## 1. Learned patience λ — implemented, off by default

`Patience` in `hrm_x/mind.py` tracks utility density `utility / ops` with an EMA
and bounded log-space dual ascent, anchored to `Controller.lam` as its single
prior, and prices every propose/derive/verify/ask decision. `ops <= 0` is a
no-op because it yields no density sample. Zero utility lowers λ toward the
exploration floor; high utility raises it. Volatility scales the effective price
without mutating learned λ. 16 tests; removing the episode-end update was
mutation-checked.

`Controller.learn_patience` stays **False** — not because the comparison was never
run, but because it was run later the same day and the learned price lost. See
`experiments/x_patience.py` and §7 below.

## 7. The patience experiment — the rule lost, and the cause is structural

`experiments/x_patience.py` (18 tests in `tests/test_x_patience.py`). The trap
would have been comparing the learned price only against the default 2e-5: "learned"
can easily mean merely "different". So the fixed baseline is a sweep across
patience's own reachable range [1e-7, 1e-1], and the learned arms are judged against
the *best* fixed arm.

Two details of the implementation had to be read before designing anything, and both
are easy to get wrong:

- `Mind.solve` reads the price once per episode and folds the episode in at the end,
  so patience can only adapt **between** episodes. One held-out task tests nothing;
  each arm is a persistent mind walking a stream of held-out tasks.
- `Mind._utility` is `2*weight - 1` clipped at 0 — the mind's own **confidence**, not
  ground truth. So the fixed point is E[confidence]/ops, which is self-consistent
  Bayesian utility but is not accuracy, and the known story miscalibration matters.

| arm | arith acc | grid acc | λ trajectory |
|---|---|---|---|
| best fixed (λ≤1e-4) | **1.000** | **1.000** | 27 / 4163 ops |
| learned from default | 0.100 | 0.636 | 2e-5 → 6.5e-4, then frozen |
| learned from 1e-1 | 0.000 | 0.000 | never moves |

−0.90 and −0.36 against the best fixed price. `pays_off: false`, and λ stayed off.

**The cause, verified arithmetically rather than asserted.** The usable band is
λ ≤ 1e-4; the cliff is between 1e-4 and 1e-3. The rule's fixed point is *average*
value density `U/ops`, but VOC needs *marginal* gain/cost. On arithmetic, one episode
costs ~6 ops at U≈1, so the target is 0.167 — four decades above the band. The first
update alone predicts `2e-5 × (0.167/2e-5)^0.15 = 7.7e-5`, and the recorded trace
*starts* at 7.75e-5. It is not wandering; it is doing exactly what it was told. It then
climbs until nothing clears its own cost, at which point `update` sees `ops == 0` and
returns without sampling — so the price freezes and **being priced out is an absorbing
state, not an oscillation**.

The one place the rule is *not* wrong is instructive. An existing test builds a
synthetic domain costing ~4062 ops per episode, giving density 1.6e-4, which lands at
break-even and behaves. That test is not wrong and was not touched: it holds in the
regime it constructs, and the experiment varies exactly what it holds fixed.

Two characterization tests now pin the failure, both mutation-checked against a
neutralized ascent (a non-climbing rule is detected by each). They are expected to fail
if `update` is ever changed — deliberately, because that change owes the project a
decision record. **No patch was written.** A sound rule needs marginal-value telemetry
that does not exist yet, and a confident-looking patch here would have been worse than
the negative result. The published default 2e-5 sits safely inside the good band, so
nothing about the shipped E9 baseline changes.

## 2. A story-domain crash: incomplete readings are a shape, not a field

`Frame.amount = None` reached `Fraction(None)` and raised `TypeError` inside
`Mind.solve`. Fixed with `reading_defect()` plus guards in `derive`/`simulate`.

The same class of bug reappeared one session later, from held-out text: "Alice
has how many apples ?" is read as a question whose **owner is nobody**, which
reached `q.owner[0]` in `derive` and raised `IndexError`. Left unguarded,
`simulate` would instead have summed over an empty tuple and returned **0** — a
confident wrong answer instead of a crash. Checking one field at a time does not
cover the shape space, so `reading_defect` is now pinned slot by slot per schema.

## 3. Learned DSL — real compression, negative economics

`hrm_x/library.py` does genuine DreamCoder-style MDL mining: subset search over
hole captures, exact declaration cost, positive-gain acceptance only, repeated
mining on the encoded corpus, exact `expand`, and a **measured** per-program
search credit. 15 tests.

Integrated into `hrm_x/domains/arith.py`. On a held-out symbol it learns
`add(a,x)` (6 uses, MDL 40→38) and the compressed program is reachable one node
sooner. Two bugs found by measuring rather than assuming:

- `Arith.bound()` subtracted the credit while ignoring the new
  `use_library_bound` flag, so the opt-in did nothing.
- **A library that was merely present made search 9.4× worse** (39,511 vs 4,218
  ops), because an arity-1 constructor multiplies the programs at every size.
  `propose` now uses the extended enumeration only when the library is actually
  being used for search.

Swept bounds 4–10: the extended language at `B−1` is 2–4× larger than the base
one at `B`, so a mined step **never** pays at one use per program. It does pay
when the repeated structure is a large share — six nested steps earn 11 nodes of
credit and beat the base language at every bound (8 programs vs 14–18,189).
`use_library_bound` defaults to off, and `experiments/x_dsl.json` records
`worth_it: false`. 9 tests, all three guards mutation-checked.

## 4. Free text — measured, and it does not transfer

Every earlier story split reused the templates' own surface forms, so "unseen
text" had never been tested. `experiments/x_perception.py` scores the tagger
against an independent hand-written regex+lexicon reader that shares no code, no
weights and no templates.

| held out | exact reading |
|---|---|
| nothing (control) | 1.00 |
| the verb only | 0.50 |
| the clause order only | **0.00** |
| the question word order | 0.67 |

76–93% of held-out stories contain a misread sentence; the answer is right ≤25%
of the time. The tagger learned template **positions**, not a compositional
reader. Recorded as [[ADR-009 Free Text Is Not Yet Supported]], which also
corrects the spec's perception claim and Limit [R].

**The measurement went against its own expectation, and that is the part worth
keeping.** The story verifier replays the reader's own reading, so it cannot see
a misreading — but only **2 of 91** misread stories produced a confident wrong
answer, because `reading_defect` rejects incomplete readings and the mind
abstains. The safe-failure property belongs to that guard, not to verification.
`tests/test_x_perception.py` pins both the guard and the blind spot
deliberately: a *complete* wrong reading ("eats" read as "acquire" → 13 verified,
text says 7) is confirmed by the verifier.

Two measurement bugs in the experiment itself are worth recording, because both
produced confident-looking nonsense: a spacing mismatch made the reference reader
match nothing (`misread = 0.00`, looking like a clean result), and a wrong
capture-label tuple made it read "how many oranges does Zoltan have ?" as
owner=None, item="zoltan" (`misread = 1.00`, looking like total failure). Both
were caught only by asserting the reference reader against hand-written
expectations, which is now a test.

## 5. The graph, rewired in one command

`scripts/simplify_graph.py` was fixed but orphaned. It is wired back into
`scripts/update_knowledge_graph.ps1`, which now runs update → prune → curated
code map → derived Obsidian view:

- `graph.json` 2,933 nodes / 7,786 links (unchanged, full detail, what
  `query`/`path`/`affected` need)
- `graph.simple.json` 554 nodes (18.9%: 364 modules + 190 classes) and
  `graph.html`, both from the pruned graph
- `vault/09 - Code Map/` 24 curated notes, now including a `hrm_x/library.py`
  subsystem (its absence was caught by `tests/test_knowledge_map.py`)
- `vault/_graph-view/` 651 derived notes, quarantined

Verified the explosion loop stays broken: **0** graph nodes are sourced from
`vault/_graph-view/`, and the 57 real vault notes are untouched. `graphify` warns
that `.graphify_analysis.json` is stale during the export; that is expected — the
sidecar describes the full graph and the export is handed the pruned one. It is
documented in the script so the next agent does not "fix" it by exporting the full
graph, which is the 1,922-note vault-root explosion.

## 6. Two documentation corrections

A mojibake scare turned out to be nothing: `§24` / `0.41→0` / `3.79×` render as
`?` in a PowerShell console but are clean UTF-8 in the file. A scanner over 736
markdown files found zero encoding problems — the single `â€”` in `AGENTS.md` is
the mojibake *documentation*. Recorded because "fixing" the console's rendering
would have corrupted a good file, which is exactly the failure AGENTS.md §3a
warns about.

Separately, the Code Map had no `hrm_x/` section at all and still claimed
markdown "requires an LLM key" — both were false. Fixed, and it now points at
ADR-009 for the perception limits.

## What is still open

- Schema induction for events (`[U]`, spec §17).
- A *sound* learning rule for patience: the current one is pinned as measurably worse
  (§7), so replacing it needs marginal-value telemetry, not a patch.
- ARC-AGI for library growth — gate 3 (b) is unproven, only the small DSL is
  measured, and the trend says the economics need high-redundancy corpora.
- The graph still holds 0 markdown nodes; targeted semantic runs only.
- ADR-005's stale plan tables and `Dashboard.md`, from the 2026-09-28 debt list.

## Files

| file | change |
|---|---|
| `hrm_x/library.py` | new — MDL mining, exact expand, measured credit |
| `hrm_x/domains/arith.py` | library-backed enumeration, `mine`, opt-in bound |
| `hrm_x/domains/story.py` | `reading_defect`; question-owner guard |
| `hrm_x/mind.py` | `Patience`, `Controller.learn_patience`; measured verdict recorded in the docstring |
| `experiments/x_dsl.py` | crossover sweep, dense corpus, honest verdict |
| `experiments/x_perception.py` | new — held-out free text, independent reference |
| `experiments/x_patience.py` | new — learned λ vs a fixed sweep over its reachable range |
| `experiments/results/x_dsl.json`, `x_perception.json`, `x_patience.json` | new results |
| `tests/test_x_dsl.py`, `test_x_perception.py`, `test_x_library.py` | new tests |
| `tests/test_x_patience.py` | +2 characterization tests pinning the measured defect (16 → 18) |
| `scripts/update_knowledge_graph.ps1`, `scripts/knowledge_map.py` | graph pipeline |
| `vault/06 - Devlogs & ADRs/ADR-009 Free Text Is Not Yet Supported.md` | new ADR |
| `vault/00 - Meta & Indexes/Task Plan.md`, `Code Map.md` | gates and map |
| `vault/01 - Architecture Blueprint/HRM-X Specification v0.4 …` | §14, §16, §17, §26 |

agent: opencode/big-pickle (big-pickle), 2026-09-28
