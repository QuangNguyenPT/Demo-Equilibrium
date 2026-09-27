# HRM-EQB — hierarchical reasoning by certified equilibrium

`hrm_eqb/` is the model (ADR-005). It is a two-level reasoning core, contractive for every
weight value. So its thinking provably converges, its halting rule carries a certified
error bound, and the certificate is recomputed from the weights every time the model file
is loaded.

**Spec and proofs:** `vault/01 - Architecture Blueprint/HRM-EQB Specification v0.2 - Theorems and Proofs.md`.
Every theorem there (§2–§9 and §14) has an executable check, and several of those checks
are mutation-tested.

## How it thinks

```
task (demos + query)  ->  token field: content ⊛ rotation(row, col, slot)        (T1)
                      ->  prelude (small transformer)
for each segment s:                                                            (deep supervision)
    routing A_L, A_H  =  S1(Sinkhorn(attention))  per head, fixed during the solve (R1, S2)
    context c_L       =  input + knowledge read (quaternion product-key memory)    (K2, E5)
    context c_H       =  input + AttnRes over earlier equilibria                   (A1, E2')
    (h*, l*)          =  certified solve of  l <- T_L(l; h),  h <- T_H(h; l)      (C1, E2, E3)
    halt when r/(1-r) * step <= tol                                              (E6)
answer = readout(h* at answer cells) · Embedᵀ      (the solution is read in token space, L2)
words  = coda(detached thinking chain)             (language cannot change the answer, L3)
```

- **Convergence.** Four norm budgets decide it: if Γ = v_h + v_l s_h/(1−s_l) < 1, every
  segment converges for every input, memory state, hormone state and fast/slow schedule.
  The number of fast steps changes the speed, never the answer (E3; tested on the model).
- **Cognition.** A committee of augmented views either agrees, and the model returns an
  **Answer**, or it disagrees or cannot re-derive its own demonstrations, and the model
  asks a **Question**. The question carries the candidate answers and the bits of
  information the reply would give (Q1).
- **Hormones.** Dopamine, serotonin, noradrenaline and acetylcholine set the reward
  signal, patience, exploration and learning rate. They act only outside the certified
  maps, so they cannot break convergence (H1).
- **Learning.** A taught answer becomes a test-time update of the knowledge rows the task
  read. It reaches the SSD only if every demonstration is then re-derived and a canary set
  does not get worse (V1).
- **Self-study.** An autoresearch-style loop practises weak families for a fixed budget,
  then keeps or discards the result with a Hoeffding gate on a frozen, hash-locked harness
  (G1). Every experiment is logged in `results.tsv`.

## The two files

```
models/<name>/
  model.hrmc     the whole core: every weight, budgets, certificate claims, probe logits,
                 SHA-256s. Pickle-free. The loader recomputes Γ and r_n from SVD norms,
                 re-runs the probes, and refuses a corrupted or forged file.
  knowledge.pt   the knowledge: a normal torch archive (quaternion values, keys, usage,
                 confidence). Rows are rewritten in place on SSD through a write-ahead
                 journal; `seal` restores the zip checksums. Without it: an empty brain.
  state.json     hormones, counters, generation, harness lock
  results.tsv    self-study ledger
```

## Quick start

```powershell
python -m pip install numpy pytest torch
python -m pytest                                          # all tests
python scripts/fetch_benchmarks.py                        # ARC-AGI-1 + Sudoku-Extreme (optional)
python -m hrm_eqb create models/eqb-small --minutes 120 --arc-train data/external/arc/training --pause 0.15
python -m hrm_eqb info models/eqb-small                   # certificate, lineage, state
python -m hrm_eqb ask models/eqb-small task.json          # Answer, or Question + candidates + bits
python -m hrm_eqb learn models/eqb-small task.json answer.json
python -m hrm_eqb study models/eqb-small --minutes 10
python -m hrm_eqb bench models/eqb-small --suites generated,language,sudoku,arc,empty_brain,schedule
```

A task file is ARC-style JSON with a `domain` (`grid`, `sequence`, `text` or `sudoku`).
Text may be given as strings:

```json
{"domain": "text", "train": [{"input": "cat", "output": "CAT"}, {"input": "owl", "output": "OWL"}],
 "test": [{"input": "river"}]}
```

**GPU safety.** All CUDA work goes through `select_device()` (allocator capped at 70 % of
VRAM), and batches are budgeted by attention pairs. An unbounded batch once spilled into
system RAM and crashed Windows. `--pause` lowers the duty cycle on a hot laptop GPU.

<!-- EQB-RESULTS -->

---

# Baseline: HRM program-search model (ADR-003)

The numpy model below is kept as teacher, judge and ablation baseline (ADR-005).

One model with four sections that think together:

| Section | Package | What it does |
|---|---|---|
| Reasoning | `hrm/reasoning` | Two levels, PEM-derived: **fast** recognition (L) and **deep** equilibrium reasoning (H), then program synthesis. Abstains when nothing explains the input. |
| Language | `hrm/language` | A native section, not a translator: it proposes hypotheses in words while the model reasons (inner speech), reads tasks written as text, explains answers, and feeds the intent matrix. |
| Knowledge | `hrm/knowledge` | A dynamic model file **on SSD, not RAM**: memmapped keys + sqlite, read inside the reasoning pass, written only with verified knowledge, crash-safe, reversible. |
| Metacognition | `hrm/metacognition` | Knows what it knows (competence map), chooses what to study (curiosity), infers what you want from an input (intent regression). |

The learning section (`hrm/learning`) closes the loop: every interaction can add verified
knowledge; a sleep cycle consolidates proven knowledge into the core ("into the neurons")
behind a frozen, hash-locked benchmark; a self-study loop practises its own gaps; human
feedback outranks the model's own.

Design record: `vault/06 - Devlogs & ADRs/ADR-003 Unified Cognitive Model.md`.

## What "understanding" means here

An input is understood when the model finds a compact executable explanation (a program)
that **reproduces every example exactly**, preferring the simplest. It answers with that
program. If no program passes the independent verifier, it **abstains**; it does not guess.
Confidence is the share of all exact explanations that agree with the answer, so an
ambiguous task never looks certain.

## How it thinks

```
input ─► perception (domain encoder: grid / sequence) ─► task core in phasor space
          │
          ├─ fast (L)      rank by algebra + consolidated priors + SSD knowledge; verify top few
          ├─ language      inner speech proposes rules in the domain's program language; verified
          ├─ deep (H)      equilibrium: resonator ticks inside refresh cycles, restarts, re-parsing
          ├─ synthesis     compose new programs by bounded search
          └─ abstain       nothing reproduces the examples
```

## How it learns

| Loop | Trigger | Judge | Boundary |
|---|---|---|---|
| Online | every answer | exact replay of the examples | never writes unverified knowledge |
| Feedback | you | you | "wrong" quarantines the knowledge used; a correction is learned as a new example |
| Sleep | `sleep()` / every N study tasks | frozen benchmark: no regression, **zero decoy false accepts**, store integrity | benchmark + scorer are hash-locked; a changed judge refuses to run until a human re-signs |
| Self-study | `study()` | the task generator's known answer | budgets; stops when nothing learnable is left or after repeated discarded cycles |

Promotion is `auto` (a passing core goes live) or `staged` (waits for `promote`). Every
generation is kept; `rollback` restores an earlier core and reactivates knowledge graduated
after it. Every cycle is a row in `ledger/results.tsv`.

## Quick start

```powershell
python -m pip install numpy pytest
python -m pytest                                  # full suite (hrm + hrm_eqb)

python -m hrm create models/main --benchmark data/benchmark/benchmark-v1.json
python -m hrm ask models/main "1 2 3 -> 3 2 1 ; 4 5 6 -> ?" --task-id t1
python -m hrm explain models/main t1
python -m hrm feedback models/main t1 wrong --correction "6 5 4"
python -m hrm study models/main --tasks 100 --minutes 480      # overnight self-study
python -m hrm report models/main                                # what it knows / doesn't
python -m hrm rollback models/main 0                            # human controls
```

Python:

```python
from hrm.model import Model
from hrm.metacognition import Message

with Model.create("models/demo", benchmark="data/benchmark/benchmark-v1.json") as m:
    sol = m.solve([([1, 2, 3], [2, 4, 6]), ([5, 1], [10, 2])], [7, 3], task_id="t1")
    print(sol.status, sol.answer, sol.program.describe(), sol.confidence)
    m.perceive(Message(text="4 5 -> 5 4 ; 1 2 3 -> ?"))        # intent inferred, text read
    m.study()                                                    # practise its gaps
```

## Model bundle (the dynamic model file)

```
models/main/
  manifest.json            generations, active generation, harness lock
  config.json              strict config (missing or unknown fields are refused)
  core/gen-NNNN.json       core weights per generation (small, RAM-resident)
  knowledge/               SSD: keys.<n>.f16 (memmap) + knowledge.sqlite
  harness/benchmark.json   frozen judge of the learning loop
  ledger/results.tsv       one row per sleep cycle
  metacognition.json       competence map + intent matrix
```

## Language backends

`LanguageConfig.backend`: `none` (default), `hashing` (dependency-free), `transformers`
(local Hugging Face model; `pip install torch transformers`), `ollama` (local server).
Everything a language model says is untrusted: program proposals and rewritten tasks are
parsed strictly and verified before they can influence an answer.

## Measured (2026-09-27, D=4096)

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

The benchmark is generated from the model's own vocabulary: it measures reasoning
correctness and honest abstention, not general intelligence.

### What the depth axis found

Building a benchmark that could actually *fail* surfaced four defects that v1
could not see, because every v1 task was solvable by a one-or-two step program:

1. **`Vector3DDomain.from_json` truncated vectors to int64** (inherited from the
   base `Domain`, correct for the grid/sequence alphabets). `0.861382` became
   `0`, so every stored demonstration described a different function than the one
   generated and the domain scored 0 on all composed tasks. In-memory probes
   passed throughout; only the JSON round trip failed.
2. **`complexity_limit` capped reasoning at depth 2.** It derived a cap from
   input width, but a 3-vector is size 3 at any depth, so three demonstrations
   implied a limit of 2 and every program of complexity 3+ was rejected as
   "memorization". The reasoner looked unable to reason past two steps because a
   guard had forbidden it.
3. **`redundant()` pruned every rotate pair.** The base test prunes two single
   steps of a family declared closed under composition — true for grid
   translations, false for 3D rotations. `rotate(x,90)` then `rotate(z,90)` is
   not a single rotation, so synthesis could not reach depth 3 at all.
4. **`Project` made the `gravity` decoy solvable** (as `Project(0)` then
   `Project(2)`), producing 2 false accepts against a gate that tolerates none.

The first version of the depth axis also measured nothing: the enumerated
rotations generate the octahedral group (order 24), so a product of k random
rotations collapses to one or two steps and every "deep" family was secretly a
depth-1 family. The generator now *searches* for a product whose shortest
explanation is k and raises rather than shipping a mislabelled task.

## Limitations (honest)

- **The deep engine and the phasor ranking are not load-bearing.** Ablating the
  H-engine, the phasor similarity, and the priors/SSD knowledge all leave v2 at
  100%. The synthesizer ranks candidates by `verifier.energy` — a per-domain
  similarity heuristic that sees the whole candidate list — and finds every task
  within budget on its own. The phasor substrate is consulted only for the fast
  path's top-8, which synthesis then subsumes. Until synthesis is either capped
  below the true program depth or made to rank by phasor signature, the
  flagship machinery costs runtime (30 s vs 13 s) and returns nothing
  measurable. This is the open question, not a solved one.
- Reasoning depth is bounded by `reasoning.synthesis_depth` (3). The fast path
  verifies single ops only, so nothing reaches depth 4; the 4 such v2 tasks are
  reported as an explicit ceiling rather than dropped.
- Verified understanding exists only where a domain has an exact check. New
  domains plug in through `hrm/perception/base.py:Domain`, and must set
  `size_bounds_complexity = False` if their input width does not track program
  depth.
- Knowledge retrieval is strong for programs with phasor algebra (translate, color cycle,
  offset, rotate) and weak for the rest at scale; invariant descriptors in the task key are
  the planned fix.
- The latent bridge between language-model hidden states and phasor space is not trained yet;
  language currently acts through proposals, text perception and intent features.
- No native kernels: profiling showed the hot paths are BLAS matmuls and per-program Python
  overhead, so C++ would not pay off yet (see ADR-003).

## Repository

```
hrm_eqb/        HRM-EQB, the model (certified core, field, memory, files, mind, study, bench, CLI)
hrm/            baseline program-search model (substrate, perception, reasoning, verify,
                knowledge, language, metacognition, learning, core, model, CLI)
tests/          pytest suite (one file per section)
data/benchmark/ frozen benchmarks v1, v2 (data/external/: downloaded ARC + Sudoku, gitignored)
scripts/        make_benchmark.py, update_knowledge_graph.ps1
vault/          Obsidian knowledge vault (ADRs, session logs, concepts)
legacy/         superseded C++ GRU engine + blueprint layer (reference only)
docs/           Executive Summary.pdf (original design study)
```
