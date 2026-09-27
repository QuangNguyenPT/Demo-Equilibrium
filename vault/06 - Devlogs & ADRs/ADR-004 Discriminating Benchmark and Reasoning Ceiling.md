---
title: "ADR-004: Discriminating Benchmark and Reasoning Ceiling"
type: adr
tags:
  - adr
  - benchmark
  - measurement
  - vector3d
  - negative-result
date: 2026-09-27
agent: "opencode/big-pickle"
model: "big-pickle"
status: accepted
supersedes: null
related:
  - "[[ADR-003 Unified Cognitive Model]]"
  - "[[Why v2 Did Not Prove the Model]]"
---

# ADR-004: Discriminating Benchmark and Reasoning Ceiling

## Context

[[ADR-003 Unified Cognitive Model]] shipped a working model and a 51-task
benchmark that reported 93.3%. That number was not trustworthy as evidence of
*architecture*, for two reasons found by measurement:

- **Every v1 task was solvable by a one-or-two step program.** A benchmark whose
  tasks all fit inside a two-step search cannot detect a reasoning system that
  only searches two steps.
- **Ablation showed no delta.** Removing the H-engine, the phasor similarity,
  and the priors/SSD knowledge all left the solve rate at 93.3%. The flagship
  machinery was free but not load-bearing.

The gap was the *instrument*, not yet the model. This ADR records the decision to
build an instrument that can fail, what building it exposed, and the ceiling it
found.

## Decision

### 1. Benchmark v2, two axes, frozen like v1

`data/benchmark/benchmark-v2.json` (108 tasks) keeps v1 intact and adds a
`vector3d` domain: dense 3-vectors in, dense 3-vectors out, with rotations,
reflections and scales. Vectors are the right probe because they are continuous,
so no finite symbolic vocabulary can be complete by construction — unlike grids
and sequences, where the answers are drawn from the same small alphabet the model
already searches.

- **Depth axis** — families `d2_rotate`, `d3_rotate`, `d4_rotate` need 2, 3 and
  4 composed steps. `d4` is past `reasoning.synthesis_depth` (3), so it measures
  the ceiling instead of assuming one.
- **Size axis** — a mixed vocabulary of 32 single ops (22 exact rotations, 3
  reflections, 7 scales) that the top-8 fast path must rank through.
- **Honest abstention** — `gravity` and decoys are verified unreachable, and
  answering one is a false accept the gate tolerates none of.

### 2. Depth labels are verified, not asserted

The first version of the depth axis **measured nothing**, and finding out why is
the most useful outcome in this ADR. The enumerated rotations generate the
octahedral group: 24 distinct matrices, 11 of them single ops, and a random
k-fold product always collapses to one or two steps. So `d5_rotate` was really a
depth-1 task and `synthesis_depth=3` cleared the whole family.

The lesson generalised: a depth axis built by *labelling* tasks is only as real
as the vocabulary that happens to express it. Reflections and scales were then
added — not decoration, but the minimum needed for the generated group to admit
products of 3 and 4 steps — and the generator now **searches** for a product
whose shortest explanation is exactly k, raising rather than shipping a
mislabelled task. `tests/test_vector3d.py` asserts the label for every deep
family.

### 3. Domains declare whether input width bounds complexity

`Domain.size_bounds_complexity` (default `True`) says whether `size()` grows with
the task. It is `False` for fixed-width inputs, where the verifier's relative
complexity cap has no meaning.

## What the instrument found

Four defects, all invisible to v1, all of which produced a plausible trace and a
wrong score rather than a crash:

| # | Defect | Why v1 could not see it | Consequence |
|---|---|---|---|
| 1 | `Vector3DDomain.from_json` inherited the base `int64` cast | v1 had no continuous domain; in-memory probes pass | `0.861382 → 0`; every stored demo described a different function; domain scored 0 on all composed tasks |
| 2 | `complexity_limit` derived the cap from input width | grid/sequence widths *do* track depth, so the heuristic was accidentally right | 3 demos ⇒ limit 2; every depth-3+ program rejected as "memorization" — the model looked unable to reason past 2 because a guard forbade it |
| 3 | `redundant()` pruned any two same-family single steps | only v1's domains are family-closed (grid translations are) | `rotate(x,90)∘rotate(z,90)` is not a single rotation, so all rotate pairs were pruned and depth ≥ 3 was unreachable |
| 4 | `Project` in the vocabulary | gravity was assumed out-of-vocabulary | `Project(0)∘Project(2)` solves it — 2 false accepts against a gate that tolerates none |

Each fix is pinned by a test that fails loudly on regression; the `from_json`
and complexity-cap fixes were mutation-checked.

## The result that matters: no discrimination

With the defects fixed, v2 scores **100% of 84 in-budget tasks, 0 false
accepts**, and the ablation ladder is flat:

| configuration | solve rate | time |
|---|---|---|
| full model | 1.000 | 33 s |
| no H-engine | 1.000 | 15 s |
| + no phasor ranking | 1.000 | 15 s |
| + no priors / SSD | 1.000 | 15 s |

**The deep engine and the phasor substrate are still not load-bearing.** The
cause is structural, and worth stating plainly: `verifier.energy` is a per-domain
similarity heuristic that sees the *whole* candidate list, and the synthesizer
ranks its beam with it (`hrm/reasoning/synthesis.py`). Synthesis therefore finds
any program within budget regardless of the phasor. The phasor is consulted in
exactly one place — the fast path's top-8 (`reasoner.py:153`) — which synthesis
then subsumes.

So v2 did not prove the architecture works. It proved the *benchmark* now can
detect it either way, and it detected that the current design has a redundant
layer. That is a more useful outcome than the 93.3% it replaced.

## Consequences

- **Ceiling is `synthesis_depth`.** The fast path verifies single ops only, so
  nothing can reach depth 4. The 4 such tasks are reported as `beyond_depth` with
  a count rather than dropped, because a silently excluded band reads as "no
  ceiling".
- **Two ways to make the phasor matter**, neither taken unilaterally:
  cap synthesis below the true program depth, or make synthesis rank by phasor
  signature instead of a per-domain heuristic. Both change what the reasoner is;
  both are ADR-worthy.
- New domains must set `size_bounds_complexity = False` if their input width does
  not track program depth, and must verify that any "unreachable" family is
  genuinely unreachable with the shipped vocabulary.

## Rejected

- **Regenerate v1 with a depth field.** v1 is frozen and hash-locked; its harness
  lock is the guarantee that the judge never changes silently. v2 is a new file.
- **Push `synthesis_depth` higher** to make depth-4 tasks pass. That moves the
  number without measuring anything, which is the exact failure this ADR exists
  to correct.
- **Force the depth axis into the octahedral group.** Depth beyond 2 is not
  expressible there at all; enlarging the vocabulary was the honest fix.
