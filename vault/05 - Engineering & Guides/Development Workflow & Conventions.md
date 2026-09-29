---
title: Development Workflow & Conventions
type: guide
tags:
  - guide
  - conventions
  - workflow
status: partly-superseded
created: 2026-09-26
updated: 2026-09-28
relates_to:
  - "[[ADR-001 Hybrid C++ Engine with Python Blueprint]]"
  - "[[Build & Toolchain Guide]]"
---

# 📐 Development Workflow & Conventions

> [!NOTE] Partly superseded (2026-09-28)
> This note spans both the Python blueprint and the C++ engine of the ADR-001 design, which was
> **never built** (ADR-003 moved it to `legacy/`). The **logging tags, type hints, dataclass
> configs, docstring style and test discipline below still apply and are the parts to follow.**
> Only the C++ standards and the `EngineBridge` rule are retired — there is no `EngineBridge`, no
> `ArenaAllocator`, no `hrm::aligned_alloc()` and no `hrm_c_api.h` in the live tree. For the
> repo-level rules (attribution, no credentials, tests as the attribution floor) see the
> repository's `AGENTS.md`.

---

## 🏷️ Logging Tags Standard

All console outputs must use structured bracketed tags `[TAG]` for clarity and log filtering:
```python
print("[BLUEPRINT] Initializing model topology...")
print("[REASON] ACT halting triggered at step 3...")
print("[ERROR] Out of memory: {e}")
```

The two C++-flavoured example lines from the original note (`[ENGINE] Allocating 128MB
ArenaAllocator...`, `[KERNEL] Testing Fused FlashAttention...`) described an engine that was never
built; the convention itself is unchanged, and the native backend's own script logs `[NATIVE]`.
In C++:
```cpp
std::cout << "[NATIVE] built hrm_x/native/bin/hrmx_native.dll\n";
std::cerr << "[ERROR] Invalid argument.\n";
```

---

## 🐍 Python Coding Standards *(still in force)*

1. **Type Hints**: All functions and methods must use complete type annotations (`typing.Optional`, `typing.List`, `typing.Dict`). *Still the rule across `hrm/`, `hrm_eqb/` and `hrm_x/`.*
2. **Dataclasses for Configs**: Model configurations must inherit from `@dataclass` to ensure strong typing and easy JSON serialization. *True of `hrm/config.py` and `hrm_eqb/config.py`; both also load strictly, refusing unknown or missing keys.*
3. **Short Docstrings**: Keep class and function docstrings concise and focused on *why* rather than trivial *what*. *Still the rule, and it is the dominant style in `hrm_x/`.*
4. ~~**No Direct Numeric Ops in Python**: Do not implement matrix multiplies or loops in Python bytecode. Delegate all tensor math to `EngineBridge`.~~ — **retired.** `EngineBridge` does not exist, and the rule was aimed at tensor math the current model does not have. The accurate version: **the Python modules are the reference semantics.** They are written for clarity, and the C++ exists to make the profiled hot paths faster without changing a single result — enforced by `tests/test_x_native.py`, not by delegation.

---

## ⚙️ C++ Coding Standards *(retired — kept for the record)*

1. **Standard**: C++17 or C++20. *The live build is C++17 (`-std=c++17`).*
2. ~~**Memory Safety**: Use RAII wrappers (`std::unique_ptr`, `Tensor`) or the pre-allocated `ArenaAllocator`. Never leave raw `new` without an explicit owner.~~ — *no `Tensor`, no `ArenaAllocator`; the rule cannot be followed as written. The intent survives: the native code owns its state explicitly and frees it in `hx_eg_free`.*
3. ~~**Cache Line Alignment**: All memory allocated for compute buffers must use 64-byte alignment via `hrm::aligned_alloc()`.~~ — **`hrm::aligned_alloc()` does not exist and the native code performs no alignment work**; buffers are caller-owned NumPy arrays.
4. ~~**Pure C-ABI Exports**: Keep `include/hrm/c_api/hrm_c_api.h` strictly compatible with pure C.~~ — *the file does not exist. The rule is now in force as:* `hrm_x/native/include/hrmx.h` stays pure C — `extern "C"`, `HX_API`, no C++ types, no exceptions, no templates across the boundary, no ownership of caller memory.

There is no threading, no OpenMP, no hand-written SIMD, and no thread pool in the current native
code. Two files, one build script, `clang++ -O2 -std=c++17` — see [[Build & Toolchain Guide]].

---

## 🔗 Related Notes
- [[Build & Toolchain Guide]] — Compiling and running tests.
- [[C-ABI & Interop Specification]] — the real `hrmx.h` surface the C++ rule above now governs.
- [[HRM-X Native Backend]] — the code those standards apply to.
- [[Master MOC]] — Vault index.
