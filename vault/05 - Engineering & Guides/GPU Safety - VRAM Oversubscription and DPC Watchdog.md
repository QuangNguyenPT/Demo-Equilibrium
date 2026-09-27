---
title: "GPU Safety: VRAM Oversubscription Crashed Windows (DPC_WATCHDOG_VIOLATION)"
type: guide
tags:
  - engineering
  - gpu
  - windows
  - hrm-eqb
agent: "claude-code"
model: "claude-opus-5-5"
date: 2026-09-27
status: active
---

# GPU Safety: VRAM Oversubscription Crashed Windows

## What happened

The machine blue-screened twice (DPC_WATCHDOG_VIOLATION) during the same command: HRM-EQB
mixed-domain training with a 16k-token batch budget. Earlier runs on sequence and text
tasks only (short token fields) ran for 20+ minutes without trouble.

## Cause (measured, 2026-09-27)

Routing tensors are (B, heads, n, n). Training keeps about 48 of them alive for backward
(Sinkhorn sweeps, grad cycles, two routings, two segments).
- Measured peak: **≈ 0.9 GiB per million padded attention pairs** (B · n_max²).
- A token-budgeted grid batch (~20 tasks × n = 800) is 12.8 M pairs, so it needs ≈ 11.7 GiB on a
  12 GB RTX 5070 Ti Laptop.

On Windows the NVIDIA driver's **sysmem fallback** lets CUDA allocations spill into system
RAM instead of failing. The resulting paging under sustained load is the likely trigger of
the driver watchdog. This is inferred from the memory arithmetic and the reproduction pattern;
the crash dump was not analysed.

## Rules now in code

| Guard | Where | Effect |
|---|---|---|
| `select_device()` caps PyTorch's allocator at 70 % of VRAM | `hrm_eqb/device.py` | overflow raises an ordinary `OutOfMemoryError` instead of spilling |
| training batches bounded by `batch_pairs` (default 4 M) and `batch_tokens` | `hrm_eqb/train.py` | measured peak 3.5–4.1 GiB |
| inference batches bounded by `EVAL_PAIRS` (8 M) via `field.budget_chunks` | `train.predict`, `mind`, `bench` | committee and ARC evaluation stay bounded |

## Rules for agents

- Never run CUDA work through a path that bypasses `select_device()`.
- Probe the peak memory of a new workload on one worst-case batch (`torch.cuda.max_memory_allocated`)
  before a long run.
- Stop background jobs only by their task ID, never with a broad `taskkill` filter.
