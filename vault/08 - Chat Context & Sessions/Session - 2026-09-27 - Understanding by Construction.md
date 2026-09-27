---
title: "Session: 2026-09-27 — Understanding by Construction (model reflection + graph cleanup)"
type: session-log
tags:
  - session
  - design
  - mathematics
  - hrm-x
  - knowledge-graph
agent: "claude-code"
model: "claude-opus-5-5"
date: 2026-09-27
status: design-proposed
code: [scripts/knowledge_map.py, scripts/update_knowledge_graph.ps1]
---

# Session: 2026-09-27 — Understanding by Construction

## Request

- Check the trained model and address its limitations.
- Reflect on the whole model, which "feels like patching over and over".
- Derive a full mathematical formulation for each section. Use a transformer to improve
  pattern input and output.
- The model must *actually understand*: 2 × 3 = 6 because × is repeated addition, not
  because of pattern. It should work well under 1M parameters, with English first.
- Fix the graph view: "two big things", a mess.

## Findings (measured)

| Probe | Result | Meaning |
|---|---|---|
| integer-embedding similarity vs distance | corr +0.44 | a fuzzy number line was learned |
| additive consistency $e(a{+}b)-e(a)$ vs $e(b)-e(0)$ | 1.39 (random 1.41) | **no arithmetic structure** |
| certificate | Γ = 0.214, rate 0.304 | converges in ~5 cycles for every task |
| parameter census | core 131k of 1.87M (7 %) | the "reasoner" is a small fraction |
| `empty_brain` vs `generated` | 0.298 = 0.298 | memory contributes nothing |

## Decisions

1. **Diagnosis.** Every v0.2 theorem is a safety theorem. None states what the model can
   compute, so fixes accumulated as patches.
2. **Proposed design:** [[HRM-X Specification v0.3 - Understanding by Construction]] and
   [[ADR-007 Grounded Algebra and Executable Concepts]] (status *proposed*; no model code
   was written).
   - Residue-phasor algebra: addition is binding (A1) and multiplication is repeated addition
     (A2), by theorem.
   - Executable concept library with an H-level executor; compiled L-level skills.
   - Transformer Reader and Writer only for conventions. About 0.83M parameters.
   - English Understanding Suite v1 with U1–U4 criteria and ablations.
3. **Algebra verified numerically before being written down** (scratch scripts, not in the
   repo):
   - A1: max error 1.7e-8.
   - A2: 20,000 repeated bindings equal the compiled power, exactly.
   - A3: exact decoding over ±3.7M, and the noise bound holds (σ = 1.2: 3.1 % measured,
     3.8 % bound).
4. **Graph view.**
   - Cause: two graphify symbol-level exports inside the vault. The repo root is opened as
     the vault, so both showed up as giant clusters.
   - Removed exactly the files listed in graphify's manifests: 2,084 at the vault root,
     3,211 in `_graph-view`, and 1,924 in a graphify backup directory.
   - Replaced by `scripts/knowledge_map.py`: 17 subsystem notes plus an index in
     `vault/09 - Code Map/`, with no function nodes.
   - Sessions link to code through a `commits:` field (git) or a `code:` field. The `code:`
     field is used for pre-git work that was squashed into the baseline commit.
5. **Repaired** the Foundation & Research note's frontmatter (the `---` fences were missing).
   **Created** the missing [[Session - 2026-09-27 - Graph Rebuild and Benchmarks]] note,
   which the index already linked.

## Files

- New:
  - `scripts/knowledge_map.py`, `tests/test_knowledge_map.py`
  - the spec v0.3 and ADR-007
  - this note
  - `vault/09 - Code Map/*` (generated)
- Changed:
  - `scripts/update_knowledge_graph.ps1`: the code map replaces the graphify Obsidian export
  - `.graphifyignore`
  - session-note frontmatter

## Open questions (for the owner)

- Approve HRM-X (ADR-007)? Package name `hrm_x/`? First English scope?
