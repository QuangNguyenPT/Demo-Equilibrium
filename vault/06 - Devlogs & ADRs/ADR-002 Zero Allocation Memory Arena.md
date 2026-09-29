---
title: ADR-002 - Zero Allocation Memory Arena
type: adr
tags:
  - adr
  - memory
  - cpp
  - performance
status: superseded-not-implemented
created: 2026-09-26
superseded_by: "[[ADR-003 Unified Cognitive Model]]"
---

# ADR-002: Zero-Allocation Bump Memory Arena for Forward Passes

> [!WARNING] Never implemented — superseded, kept for the record
> This ADR was **accepted on paper but never built**. There is **no arena, bump
> allocator, or custom allocator in any live package** — not in `hrm/`, not in
> `hrm_eqb/`, not in `hrm_x/`. The `ArenaAllocator` paths named below
> (`include/hrm/core/allocator.hpp`, `src/core/allocator.cpp`) exist only under
> `legacy/`, where [[ADR-003 Unified Cognitive Model]] parked the never-working
> engine on 2026-09-27. No measurement in this project was ever made against a
> zero-allocation forward pass.
>
> **Superseded as the design target 2026-09-29.** At the time of writing this record the
> current model was HRM-X v0.4 (spec v0.4, ADR-008). The current target is **HRM-X v0.5**
> (`hrm_x/`), spec v0.5 + ADR-010. This document is kept verbatim as the decision record.
> Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.
>
> Do not implement against this ADR.

## Context & Problem Statement
During inference and iterative recurrent reasoning passes (e.g. up to 10–50 HRM cycles), allocating and freeing temporary activation tensors on the OS heap via `malloc`/`free` or `new`/`delete` causes:
- Heap fragmentation.
- Thread lock contention on OS allocator memory pools.
- Microsecond-level latency jitter across forward passes.

## Decision
We implement a dedicated `ArenaAllocator` (`include/hrm/core/allocator.hpp`, `src/core/allocator.cpp`):
1. Pre-allocate a contiguous memory arena (64MB – 512MB) once at engine initialization.
2. Align all allocations within the arena to 64-byte boundaries (CPU cache line size).
3. Allocate sequentially via an $O(1)$ pointer bump.
4. Reset the arena in $O(1)$ time by setting the offset back to 0 between forward passes.

## Consequences

### Positive
- Zero runtime heap allocation during forward passes.
- Guaranteed cache-line alignment prevents split-cache penalties on AVX-512 / AVX2 instructions.
- Instantaneous reset without traversing destructor chains.

### Negative / Trade-offs
- The maximum single-pass memory footprint must fit within the pre-allocated capacity, requiring graceful `OutOfMemory` detection and capacity sizing.
