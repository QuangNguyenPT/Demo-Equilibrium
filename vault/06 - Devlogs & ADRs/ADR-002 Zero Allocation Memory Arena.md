---
title: ADR-002 - Zero Allocation Memory Arena
type: adr
tags:
  - adr
  - memory
  - cpp
  - performance
status: accepted
created: 2026-09-26
---

# ADR-002: Zero-Allocation Bump Memory Arena for Forward Passes

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
