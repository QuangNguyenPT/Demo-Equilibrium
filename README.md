# HRM-X — structural operator inference

The current model is **HRM-X v0.5** (ADR-010), in `hrm_x/`. It is not a neural net and not a
chain-of-thought prompt. A task is a **situation + operator + question**; understanding means
inferring the operator exactly, with a check written in a different representation from the one
that produced the answer.

v0.5 changed what a hypothesis *is*. It is no longer a program the mind has to search for: it is
a **family** — the exact solution set of a law set — whose rational parameters are fitted
exactly, and which may answer a query only where the determinacy theorem says the evidence fixes
it. The v0.4 program-search route is still in the tree, because it is the baseline the new route
is measured against.

- **Normative design:** `vault/01 - Architecture Blueprint/HRM-X Specification v0.5 - Structural Operator Inference.md`
- **Decision record:** `vault/06 - Devlogs & ADRs/ADR-010 Structural Operator Inference.md`
- **Previous:** spec v0.4 and ADR-008, program search over a hand-written DSL.
- `python -m pytest` → **381 passed, 1 skipped**. 128 of those are HRM-X (`tests/test_x_*.py`,
  including differential parity for the C++ backend); the rest cover `hrm/` and the vault tooling.
  The 55 `hrm_eqb` tests moved with the package to `legacy/` and no longer run by default —
  see [Legacy](#legacy).
- **Gates G1–G3 (mathematics, representation, capability) are closed. G4 — refusing operators
  outside the hypothesis language — is not.** On out-of-language operators the v0.5 route gives
  5 confident wrong answers out of 6, where v0.4 gave 0 wrong and 4 abstentions. That regression
  is measured, explained and pinned by a test. Read ADR-010 before quoting any headline here.

The motivating example is the one in the spec: from `2×2=4` and `2×3=6` the model must reach
`10×3.1=31` through a *reusable* operation, not by interpolating. It does, exactly, and the
answer comes with the program and the rule that produced it.

## How it thinks

```
task (demos + query)                     ->  situation + operator + question
  workspace: e-graph with provenance     ->  congruence closure; a conflict is refused and blamed   (I1-I3)
  hypotheses: programs, prior 2^-size    ->  belief is a POSTERIOR over behaviour classes,
                                              not a winner: concepts are stored as version spaces
  H  (a few decisions per task):         ->  value of computation over {propose, derive-group,
                                              verify, ask, stop}; stops when VOC <= 0
  L  (work inside one decision):         ->  saturate rules to a least fixpoint, extract the
                                              cheapest term, execute it exactly (Fractions)
  laws: QuickSpec over concepts         ->  proposed by testing, pruned by congruence, domain-guarded
  verification: independent              ->  arithmetic: the *definition* in residue-phasor space
                                              (repeated addition as exponentiation), ignoring laws
                                              stories: dictionary replay + conservation of items
                                              grids: applicability conditions, not outputs
  answer argmax q(y) if U > 0, else abstain with the reason and the culprits
  writer.py renders the trace as an answer + an explanation
```

- **The controller.** `q(y)` weights each hypothesis by its weight and by `ρ` (its reliability:
  1 if verified, else the running pass rate). The value of the state is
  `U = max(0, (1 + L)·max_y q(y) − L)`, and each cycle takes `argmax_a E[U | a] − U − λ·cost(a)`.
  Two corrections were needed and are tested: hypotheses predicted to agree are derived
  *together* (otherwise two agreeing 50% readings look worthless one at a time), and cost is
  estimated **after** rewriting, or the controller refuses computations that laws make cheap.
- **Neuromodulators, only where they compute.** ACh-like: the reliability `ρ`. DA-like: outcome
  feedback. NE-like: volatility, meaning surprise *in excess of* the mind's own expected surprise
  `H(ρ)`, which discounts stale evidence. 5-HT-like: `λ`, fixed, with no learning rule proposed.
  Nothing else from the original hormone layout is implemented, and none of it may be added
  without an ablation.
- **Curiosity** is expected information gain over a concept's version space, weighted by use,
  during sleep. Sleep also consolidates laws and re-tests them on imagined values.
- **Abstention is a feature.** If no reading answers, the mind says so and names the culprits.
  Similarity never creates an equality, and an answer only ever comes from derivation plus
  verification.

The two things this is *not*: a library lookup dressed up as reasoning, and a model of
perception. The story verifier replays the same reading the story domain produced, so a
mis-parse passes it — that blind spot is measured and reported, not solved (E10 below).

## Native backend (it exists, and it is faster)

`hrm_x/native/` is a real C++ backend behind a pure C ABI, loaded by ctypes:

```
hrm_x/native/include/hrmx.h     C ABI
hrm_x/native/src/egraph.cpp     union-find, hash-cons, rebuild, e-matching, explanations
hrm_x/native/src/grid.cpp       connected components and the object step
hrm_x/native/bin/hrmx_native.dll  (gitignored build output)
```

```powershell
.\scripts\build_native.ps1                 # clang++ -O2 -std=c++17 -shared -> hrm_x\native\bin\hrmx_native.dll
.\scripts\build_native.ps1 -Debug          # -O0 -g
$env:HRMX_BACKEND = "python"               # python | native | auto (default: auto)
python -m experiments.x_native             # parity + timing
```

Measured (`experiments/results/x_native.json`, best of 3, `perf_counter`):

| Workload | Python | Native | Speedup | Results |
|---|---|---|---|---|
| grid: 2-step induction | 0.387 s | 0.102 s | **3.79×** | identical |
| arithmetic: laws + 60 questions | 2.120 s | 1.442 s | **1.47×** | identical |

Python is the blueprint and the reference; C++ runs only the profiled hot paths, and exact
rational arithmetic deliberately stays in Python because arbitrary precision *is* the guarantee
(Amdahl caps that speedup at ~1.5×, which is accepted). Porting the e-graph was not just a
speed change: it surfaced two real reference bugs — union-by-size swapped the roots but not the
constants, dropping a value, and a congruence conflict did not name its argument premises. Both
are fixed and regression-tested.

<!-- HRMX-RESULTS -->

## Measured (`experiments/results/`)

Every number below is read from a JSON file in `experiments/results/`.

| # | Hypothesis | HRM-X | Baseline |
|---|---|---|---|
| E1 | extrapolate an induced operator (4 splits) | **1.000** on all four | kNN 0.41 → 0; MLP 0.00; log-MLP 0.00 |
| E2 | compose learned `×` with `+` | **1.000** at depths 1, 2, 3 | — |
| E12 | abstraction pays for itself | `1000×3.1` **1,021 → 192 ops**; `10⁴×3.1` 10,020 → 411; `10⁵×3.1` refused without laws, computed in 779 with them | — |
| E9 | adaptive compute beats fixed | VOC **1.000** / 248 ops easy, **0.975** / 4,346 hard, 0 wrong | fixed 3,836 / 10,182; greedy 0.750 on hard |
| E10 | verification under injected faults | false law: **0** wrong with the verifier, 9 without | perception role swap: 10 wrong either way (blind spot) |
| E3/E4/E8 | variable binding, role swaps, counterfactual edits | **1.000** | story baseline 0.60 on unseen names at 16k |
| E5 | long chains (3 people, 4–6 events) | **0.993**–1.000, 0 wrong answers | 0.040 |
| E6 | rule composition (grids, never a task) | **0.931** (seed 0) / **0.833** (seed 1) | pixel LUT 0.034; no-objects 0.000 |
| E7 | new grid arrangements | **0.984** (seed 0) / **1.000** (seed 1) | pixel LUT 0.25–0.29 |
| C | curiosity vs random study | `×` identified with **1.0** question vs **4.8**; later accuracy **0.86** vs 0.71 | no study: 0.67, and a third of the answers are wrong |
| V | volatility under a regime change | **2** wrong answers before verification resumes | 6 without the signal |
| N | native parity and speed | identical results; **3.79×** grid, **1.47×** arithmetic | — |

Curiosity detail: with `×` seeded by two examples, curious study asks **1.0** question where
random asks 4.8; unaided accuracy afterwards is 0.86 (curious), 0.71 (random), 0.67 (none).
Volatility detail: with a false law planted after 100 tasks and verification made expensive,
the NE-like signal cuts wrong answers from 6 to 2 and shortens the blind window from 6 tasks
to 2.

The story domain is the clearest single result. On i.i.d. splits, unseen names and items, large
numbers, long chains and contradictory premises, HRM-X is 1.000 with zero wrong answers against
a 3-layer transformer trained on 4,000 stories (0.21 / 0.17 / 0.00 / 0.03 / 0.00); the
transformer improves to 0.65 / 0.60 on the first two when trained on 16,000 stories, and HRM-X
drops to 0.993 only on long chains.

### E13 — the v0.5 concept route against the v0.4 program route (2026-09-29)

Both routes, one `Mind`, one value-of-computation controller, 26 solves per condition
(`experiments/x_v05.py` → `experiments/results/x_v05.json`). Correct / wrong / abstain:

| E13a, 21 queries | correct | wrong | abstain | mean ops |
|---|---|---|---|---|
| **concept v0.5** | **14** | 7 | 0 | **751.5** |
| arith v0.4 | 9 | 4 | 8 | 4247.2 |

| split | concept v0.5 | arith v0.4 |
|---|---|---|
| in-language (15) | **13 / 2 / 0** at 593 ops | 7 / 4 / 4 at 4265 ops |
| out-of-language (6) | 1 / **5** / 0 at 1147 ops | 2 / **0** / 4 at 4202 ops |

Read this table in both directions. **Where the hypothesis language covers the operator, v0.5 is
better on every axis and 7.2× cheaper**, because deriving stopped being a search. **Where it does
not, v0.5 is worse than the route it replaces**: five confident wrong answers out of six, against
none. The mechanism is known and pinned by a test — one question brings a second value of an
argument into the evidence, so the open-world coverage rule stops firing, and a *simpler wrong*
family that fits the evidence exactly gets confirmed by the dual check. That is the coincidental-fit
channel (doc 13, Q1), and closing it is the next design decision, not a tuning problem.

Two further measurements from the same run: the closed-world ablation moves the pinned-argument
task list by 14 ops out of 1618, so on these operators the open-world machinery is close to inert;
and the delivery's answer threshold `L = 4` abstains on **21 of 21** here, because before any
verification `rho` is 0.8 and `(1+4)(0.8) − 4 = 0` exactly.

## The model, and the two older generations

| Package | Size | Role | Status |
|---|---|---|---|
| `hrm_x/` | 20 py, 4,339 lines (non-blank) + 545 lines of C++ | **HRM-X v0.5, the current model.** Structural operator inference: law-closed families fitted exactly over ℚ, determinacy, open-world coverage, VOC control, independent verifiers. The v0.4 program-search route is kept inside it as the comparison baseline. | Current (ADR-010, spec v0.5) |
| `legacy/hrm_eqb/` | 23 py, 2,427 lines | HRM-Phi equilibrium: a two-level solver whose convergence certificate is recomputed from the weights at load. | **Moved to `legacy/` 2026-09-29** — superseded, and an audit found no non-test code imports it |
| `hrm/` | 49 py, 4,991 lines | Program-search baseline: numpy-first, FHRR substrate, SSD knowledge store, learning loops. | Superseded (ADR-003) but **still live**: `scripts/make_benchmark.py` imports `hrm.learning` to generate the frozen benchmark |

Line counts are non-blank source lines, excluding `__pycache__`. `hrm/` stays in the tree
because something still builds on it. `legacy/hrm_eqb/` moved out because nothing did — and
`legacy/` keeps both its contents and the negative results that justify the current design.
See [Legacy](#legacy) for what lives there and how to run its tests.

### Generation 2, in brief: `legacy/hrm_eqb/` (ADR-005, spec v0.2)

Two-level reasoning core, contractive for every weight value, so thinking provably converges
and the halting rule carries a certified error bound recomputed from the weights each time the
model file is loaded. A model directory holds `model.hrmc` (the whole core: weights, budgets,
certificate claims, probe logits, SHA-256s — the loader recomputes `Γ` and `r_n` from SVD
norms, re-runs the probes, and refuses a corrupted or forged file), `knowledge.pt` (quaternion
values, keys, usage, confidence; rows are rewritten in place through a write-ahead journal), a
`state.json` and a `results.tsv` self-study ledger. Theorem checks live in
`legacy/tests/test_eqb_theorems.py` (§2–§8) and are mutation-tested. ADR-006 recorded what it
does *not* do: it learned patterns, and no arithmetic. ADR-011 sets out which of its ideas
are worth re-taking, and why the rest are not.

### Generation 1, in brief: `hrm/` (ADR-003, program search)

| Section | Package | What it does |
|---|---|---|
| Reasoning | `hrm/reasoning` | Two levels, PEM-derived: fast recognition (L) and deep equilibrium reasoning (H), then program synthesis. Abstains when nothing explains the input. |
| Language | `hrm/language` | A native section, not a translator: inner speech proposes hypotheses in words, reads text tasks, explains answers, feeds the intent matrix. |
| Knowledge | `hrm/knowledge` | A dynamic model file on SSD, not RAM: memmapped keys + sqlite, read inside the reasoning pass, written only with verified knowledge. |
| Metacognition | `hrm/metacognition` | Competence map, curiosity, intent regression. |

An input is understood when the model finds a compact executable explanation that reproduces
every example exactly, preferring the simplest; if no program passes the independent verifier
it abstains. Its benchmark is generated from the model's own vocabulary, so it measures
reasoning correctness and honest abstention, not general intelligence:

| benchmark v1 (51 tasks) | |
|---|---|
| solve rate | 93.3% (45 scored; `gravity` deliberately outside the vocabulary) |
| decoy false accepts | 0 |
| benchmark time | 28 s |

| benchmark v2 (108 tasks, adds the vector3d depth axis) | |
|---|---|
| solve rate | 100% of 84 in-budget tasks (depth-1-2 and depth-3 bands both 1.000) |
| decoy false accepts | 0 |
| beyond-search-budget | 4 tasks, 0 solved — the depth-4 ceiling, reported not hidden |
| benchmark time | 30 s |

**What the depth axis found, and what it did not.** Building a benchmark that could actually
fail surfaced four defects v1 could not see, because every v1 task was solvable in one or two
steps: `Vector3DDomain.from_json` truncated vectors to int64 (so `0.861382` became `0`, and
only the JSON round trip failed, never the in-memory probes); `complexity_limit` capped
reasoning at depth 2, so the reasoner looked unable to reason past two steps because a guard had
forbidden it; `redundant()` pruned every rotate pair, true for grid translations and false for
3D rotations, so depth 3 was unreachable; and `Project` made the `gravity` decoy solvable,
producing 2 false accepts against a gate that tolerates none. The first version of the axis
also measured nothing, because the enumerated rotations generate the octahedral group (order
24) and a product of k random rotations collapses to one or two steps. The generator now
*searches* for a product whose shortest explanation is k and raises rather than shipping a
mislabelled task.

Its honest limitations, still the reason ADR-008 was written: **the deep engine and the phasor
ranking are not load-bearing.** Ablating the H-engine, the phasor similarity, and the
priors/SSD knowledge all leave v2 at 100%; the synthesizer ranks by `verifier.energy`, a
per-domain similarity heuristic that sees the whole candidate list, and finds every task within
budget on its own. The flagship machinery costs runtime (30 s vs 13 s) and returns nothing
measurable. Reasoning depth is bounded by `reasoning.synthesis_depth` (3). Verified
understanding exists only where a domain has an exact check. An earlier note here also claimed
C++ "would not pay off yet" — that measurement was about BLAS matmuls, not about this repo's
actual hot paths, and it is now false: see the native backend section above.

## Legacy

`legacy/` is out of the live tree. Nothing in `hrm_x/`, `experiments/` or `scripts/` imports
any of it, and `pyproject.toml` does not package it. It holds two different things and they
are not the same claim:

- **`legacy/cpp_engine/`** — the ADR-001/002 C++ engine and Python blueprint. **Never built.**
  Only the C-ABI boundary survived, re-landed scoped to the e-graph as
  `hrm_x/native/include/hrmx.h`.
- **`legacy/hrm_eqb/`** — HRM-Φ / HRM-EQB. **Built, measured, then superseded.** It scored
  ARC 0.000 and sudoku_extreme 0.000, and its 55 tests all passed. It moved here on
  2026-09-29 because an audit found **zero** files outside its own test suite imported it.
  [[ADR-011 Reintroducing EQB Ideas into HRM-X]] proposes which three of its ideas are worth
  re-taking; the rest stay as the record of why the current design exists.

Its 55 tests are out of `testpaths`, so `python -m pytest` does not collect them:

```powershell
$env:PYTHONPATH = "legacy"
python -m pytest legacy/tests/test_eqb_theorems.py legacy/tests/test_eqb_model.py `
                        legacy/tests/test_eqb_runtime.py -q     # 55 passed
```

Running the whole `legacy/tests` folder instead gives 1 failure — `test_blueprint.py`
wants the compiled ADR-001 engine, which was never built. That failure predates the move.

The knowledge graph excludes `legacy/` on purpose (`.graphifyignore`): these symbols must
never surface in an `affected` query against `hrm_x/`.

---

## Quick start

```powershell
python -m pip install numpy pytest                      # HRM-X itself needs numpy only
python -m pytest                                        # 381 passed, 1 skipped
.\scripts\build_native.ps1                              # optional, enables the C++ backend

python -m experiments.x_arith                           # E1, E2, E12 + kNN / MLP baselines
python -m experiments.x_story                           # stories, variable binding, counterfactuals
python -m experiments.x_grid                            # grids, two seeds via x_grid_seed1
python -m experiments.x_control                         # E9 value-of-computation policies
python -m experiments.x_metacog                         # curiosity, volatility
python -m experiments.x_native                          # differential parity + timing
```

Each runner writes `experiments/results/<name>.json`; nothing is claimed here that is not in one
of those files. `experiments/common.py` holds the shared seeds, splits and baselines.

`python -m hrm …` is still the CLI for the program-search package. The EQB CLI moved with its
package and needs the path: `$env:PYTHONPATH = "legacy"; python -m hrm_eqb …`. See `CLAUDE.md`.

## Repository

```
hrm_x/              HRM-X v0.5, the current model (workspace, rules, induction, mind, writer)
                    v0.5 concept core: qlinalg, concepts (law-closed families), consolidation,
                    situation, algebra, operators, reader (torch; not imported by __init__)
  domains/          concept (v0.5 structural route), arith (v0.4 search + residue verifier),
                    story (parser + situation), grid
  native/           C ABI, egraph.cpp, grid.cpp, ctypes wrapper
legacy/hrm_eqb/       HRM-EQB (moved out 2026-09-29): certified core, field, memory, files,
                    mind, study, bench, CLI — no non-test code imports it
hrm/                ADR-003 program-search baseline (substrate, perception, reasoning, verify,
                    knowledge, language, metacognition, learning, core, model, CLI)
experiments/        x_arith, x_story, x_grid, x_control, x_metacog, x_native, common
                    results/ holds the 12 recorded JSON results
tests/              pytest suite, 20 files: the ADR-003 sections and test_x_*
scripts/            update_knowledge_graph.ps1, knowledge_map.py, simplify_graph.py,
                    build_native.ps1, make_benchmark.py, fetch_benchmarks.py
data/benchmark/     frozen benchmarks v1, v2 (data/external/ holds downloaded ARC + Sudoku)
vault/              Obsidian knowledge vault (specs, ADRs, session logs, code map)
legacy/             out of the live tree: superseded hrm_eqb/ (with its 55 tests) and the
                    never-built ADR-001 C++ engine; see legacy/README.md
graphify-out/       the raw knowledge graph; the full graph is what `query`/`path`/`affected` read
models/, runs/      gitignored trained artifacts and logs
docs/               Executive Summary.pdf and the two reference papers
```

The knowledge graph is regenerated with `.\scripts\update_knowledge_graph.ps1`; the default
`claude-cli` backend needs no API key, and `-Mode code-only` is the no-LLM fallback.
