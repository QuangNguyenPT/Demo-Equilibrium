---
title: "Session: 2026-09-27 — HRM-EQB Full Build (N1–N8)"
type: session-log
tags:
  - session
  - build
  - hrm-eqb
  - git
agent: "claude-code"
model: "claude-opus-5-5"
date: 2026-09-27
status: in-progress
---

# Session: 2026-09-27 — HRM-EQB Full Build (N1–N8)

## Request

The user asked to:
- continue building until the full model is complete and passes its tests;
- add git version control;
- prove every new mathematical claim (a proof, plus a check that it holds);
- keep the vault and the graph up to date.

The session was interrupted twice by machine crashes (DPC_WATCHDOG_VIOLATION). The user
asked for commands to be checked carefully and for the vault to be used as the lookup
table.

## Decisions

1. **Git.**
   - Initialized with a baseline commit on `main`; work continues on `feat/hrm-eqb-full`.
   - `.gitignore` was rewritten. It had a UTF-8 BOM, so its `.env` pattern never matched.
   - `.gitattributes` pins LF line endings.
2. **The model is built** as N1–N7 in `hrm_eqb/`:
   - task field;
   - prelude with rotation-bound positions;
   - certified segments with Sinkhorn head-wise routing;
   - quaternion product-key memory;
   - AttnRes chain as context;
   - tied answer head, halting head, language coda;
   - `.hrmc` and `knowledge.pt` files;
   - runtime `Mind`, self-study and CLI.
3. **Spec §14** adds 11 construction lemmas with proofs: G1a, S2, R1, P1, E2′, E7, K2,
   L3, F3, V1, plus the F1 update. Rules R2 and R3 are measured rules rather than lemmas.
   Each has a test.

## Defects found by measurement (each fixed and tested)

| Defect | How it showed | Fix |
|---|---|---|
| NaN gradients | `x/‖x‖` backward at zero-state padding tokens | smoothed radius in σ (N1 proof updated) |
| NaN at step 64 | probability-space Sinkhorn divided by underflowed sums | log-domain Sinkhorn (R1) |
| 80 % of step time in GPU float64 SVD | cProfile | certificate on CPU, computed once per forward |
| EMA evaluated as 37 % random init | 0.3 % held-out accuracy | EMA warmup |
| Served segments never trained | loss 0.003, exact match 0 | train every served segment (R2) |
| Answers larger than the canvas | `double` and `rotate` targets dropped | demo-implied shape rule (transpose first, then integral scale) |
| **Machine crashes** | VRAM oversubscription spilled into system RAM | 70 % allocator cap, pair and token budgets, gradient checkpointing, rolling checkpoints |
| CLI test wrote a checkpoint into `runs/` | a later run would resume from it | test uses `tmp_path` |
| `bits: 1.0` reported for a single-view question | Q1 says the information gain is 0 | report the true value |

The mutation checks caught every real defect. Two apparent misses were equivalent mutants
(a redundant detach, and an initial mask that Sinkhorn makes irrelevant). This was checked
by mutating both guards together.

## Proof-of-concept results

- A tiny certified model fits 8 fixed tasks exactly on CPU in about 20 s, with Γ still
  < 1 after training.
- On CPU, training through the core learns fixed rules (`text/upper` 94 % after 400
  steps). In-context families (`sequence/offset`) learn slowly; per step, the full model
  keeps pace with a prelude-only baseline.

## Files

- New: `hrm_eqb/{config,tasks,field,model,memory,coda,knowledge,hrmc,train,mind,study,bench,device,__main__}.py`,
  `scripts/fetch_benchmarks.py`, `tests/test_eqb_model.py`, `tests/test_eqb_runtime.py`,
  `.gitattributes`.
- Vault: spec §14, [[GPU Safety - VRAM Oversubscription and DPC Watchdog]], ADR-005
  status, the AttnRes concept note (use corrected).

## Results

See [[ADR-006 HRM-EQB Training and Benchmark Results]] (filled after the 120-minute run).
