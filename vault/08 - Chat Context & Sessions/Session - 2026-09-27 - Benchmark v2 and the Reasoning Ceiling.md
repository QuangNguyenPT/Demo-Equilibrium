---
title: "Session: 2026-09-27 — Benchmark v2 and the Reasoning Ceiling"
type: session-log
tags:
  - session
  - benchmark
  - vector3d
  - measurement
  - bugfix
  - negative-result
date: 2026-09-27
agent: "opencode/big-pickle"
model: "big-pickle"
status: complete
# before git; squashed into baseline commit 2c2675c
code: [hrm/perception/vector3d.py, hrm/reasoning/controller.py, hrm/learning/harness.py, hrm/verify/verifier.py, scripts/make_benchmark.py]
related:
  - "[[ADR-004 Discriminating Benchmark and Reasoning Ceiling]]"
  - "[[ADR-003 Unified Cognitive Model]]"
  - "[[Why v2 Did Not Prove the Model]]"
---

# Session: 2026-09-27 — Benchmark v2 and the Reasoning Ceiling

Goal: make the benchmark able to *fail*, then find out whether the flagship
architecture survives it. Plan: [[ADR-004 Discriminating Benchmark and Reasoning Ceiling]].

## User decisions (this session)

| Question | Answer |
|---|---|
| Which sections | S1 benchmark + S2 controller fix + S6 docs; S3–S5 deferred |
| Benchmark axes | **Both** vocabulary size and composition depth |
| Vector domain | 3-vectors from an embedding-shaped activation, used for rotation etc. |

## Work done

| Item | Result | Tests |
|---|---|---|
| S2 controller livelock | Exhausted restart budget now returns `ABSTAIN` instead of spinning to `max_ticks`; v1 100.8 s → 29.3 s, max ticks 2000 → 369, solve rate and 0 false accepts unchanged | 199 |
| `vector3d` domain | Rotations (exact, 22), reflections, scales, `Project`; float `from_json`; verified depth search | — |
| Benchmark v2 | 108 tasks, 0–4 depth bands, per-depth scoring, in-budget / beyond-depth / out-of-vocabulary bucketing | — |
| Defect fixes | Four defects below, each pinned by a test | 220 |

## The four defects v2 exposed

v1 could not see any of them, because every v1 task fits in a two-step search.
Each produced a plausible trace and a wrong score, never a crash:

1. **`from_json` truncated vectors to `int64`** (inherited from base `Domain`,
   correct for grid/sequence). Every stored demonstration described a different
   function than the one generated. In-memory probes passed the whole time —
   only the JSON round trip through a written benchmark file exposed it.
2. **`complexity_limit` capped reasoning at depth 2.** It scaled the cap with
   input width, but a 3-vector is size 3 at any depth, so three demos implied a
   limit of 2 and every depth-3+ program was rejected as "memorization". The
   model looked incapable of reasoning past two steps because a guard forbade it.
3. **`redundant()` pruned every rotate pair** on a family-level closure claim
   that is true for grid translations and false for 3D rotations, making depth
   ≥ 3 unreachable by synthesis.
4. **`Project` made the `gravity` decoy solvable** as `Project(0)∘Project(2)` —
   2 false accepts against a gate that tolerates none.

Plus two of my own mistakes, both caught by measurement rather than review: the
first depth axis was built by *labelling* tasks rather than verifying them (the
octahedral group is order 24, so no rotation product needs more than 2 steps —
every "deep" family was secretly depth-1), and a `depth=0` default silently
broke v1's scoring until `Optional[int] = None` separated "unknown" from
"unreachable".

## Result: the instrument works, the architecture does not yet earn its cost

| benchmark | solve | false accepts | note |
|---|---|---|---|
| v1 | 0.933 | 0 | 45 scored, unchanged from the ADR-003 baseline |
| v2 | 1.000 | 0 | 84 in-budget tasks, 4 beyond-budget unsolved |

Ablation ladder on v2: full 1.000 / no H-engine 1.000 / no phasor ranking 1.000 /
no priors+SSD 1.000. **Flat.**

The reason is structural, not a bug: `verifier.energy` is a per-domain similarity
heuristic over the whole candidate list, and the synthesizer beams on it. The
phasor is used in exactly one place — the fast path's top-8 — which synthesis
then subsumes. So the deep engine and the substrate cost runtime (33 s vs 15 s)
and return nothing measurable.

## Open questions

- Cap synthesis below the true program depth, or make it rank by phasor
  signature? Both change what the reasoner is; both need their own ADR.
- Is the 100% on v2 itself a warning sign? A benchmark the model aces may be no
  harder than v1 in the ways that matter. The 4 depth-4 tasks are the only
  unsolved population and they are unsolved *by construction*.
- `language` still has no trained bridge to phasor space; vector3d is
  embedding-*shaped* only, not fed by a real embedding layer.
