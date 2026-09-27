---
title: "Session: 2026-09-27 — Graph Rebuild and Benchmarks"
type: session-log
tags:
  - session
  - benchmark
  - knowledge-graph
agent: "claude-code"
model: "claude-sonnet-4-6"
date: 2026-09-27
status: complete
commits: [6e892b3]
---

# Session: 2026-09-27 — Graph Rebuild and Benchmarks

## Request

- Report training progress.
- Check the mathematical invariants.
- Rebuild the knowledge graph with session links.
- Write a detailed model report.
- Add mandatory session-start rules to CLAUDE.md.

## What happened

- **Training.** It was interrupted at 95 % of budget. It resumed from the checkpoint and
  finished at step 18,233 (8,983 s). `models/eqb-small/` was written (core sha256
  `2bb582c5280bf980`).
- **Invariants.** The equilibrium bound stayed in 1e-3 to 9.9e-3 below `solve_tol`
  throughout, and the theorem suite is green.
- **Benchmarks.**

  | suite | score |
  |---|---|
  | language | 0.983 |
  | schedule invariance | 1.000 |
  | generated | 0.298 (text ≈ 1.0, grid 0.0) |
  | ARC | 0.000 |
  | Sudoku | 0.000 |

  Details are in [[ADR-006 HRM-EQB Training and Benchmark Results]].
- **Graph.** Rebuilt with semantic chunks: 2,948 nodes and 263 communities.
- **CLAUDE.md.** Session-start rules were added.

## Defects this session introduced (fixed in the next session)

- A manual `to_obsidian` export wrote 3,211 symbol-level notes into `vault/_graph-view`. An
  earlier export had already left 2,084 at the vault root. Together they were the "two big
  blobs" in the graph view. Removed in [[Session - 2026-09-27 - Understanding by Construction]].
- An early call passed a directory where graphify's `to_json`/`to_html` expected a *file*
  path, so an HTML file was written over `graphify-out/`. It was recovered from graphify's
  backup the same session.
