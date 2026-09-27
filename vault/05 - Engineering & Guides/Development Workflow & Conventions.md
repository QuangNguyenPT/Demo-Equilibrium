---
title: Development Workflow & Conventions
type: guide
tags:
  - guide
  - conventions
  - workflow
status: active
created: 2026-09-26
---

# 📐 Development Workflow & Conventions

This guide standardizes code style, logging conventions, and testing protocols across both the Python blueprint and C++ engine layers.

---

## 🏷️ Logging Tags Standard

All console outputs must use structured bracketed tags `[TAG]` for clarity and log filtering:
```python
print("[BLUEPRINT] Initializing model topology...")
print("[ENGINE] Allocating 128MB ArenaAllocator...")
print("[REASON] ACT halting triggered at step 3...")
print("[ERROR] Out of memory in ArenaAllocator: {e}")
```

In C++:
```cpp
std::cout << "[CONFIG] Batch: " << B << ", SeqLen: " << T << "\n";
std::cout << "[KERNEL] Testing Fused FlashAttention...\n";
std::cerr << "[ERROR] Invalid tensor dimensions.\n";
```

---

## 🐍 Python Coding Standards

1. **Type Hints**: All functions and methods must use complete type annotations (`typing.Optional`, `typing.List`, `typing.Dict`).
2. **Dataclasses for Configs**: Model configurations must inherit from `@dataclass` to ensure strong typing and easy JSON serialization.
3. **Short Docstrings**: Keep class and function docstrings concise and focused on *why* rather than trivial *what*.
4. **No Direct Numeric Ops in Python**: Do not implement matrix multiplies or loops in Python bytecode. Delegate all tensor math to `EngineBridge`.

---

## ⚙️ C++ Coding Standards

1. **Standard**: C++17 or C++20.
2. **Memory Safety**: Use RAII wrappers (`std::unique_ptr`, `Tensor`) or the pre-allocated `ArenaAllocator`. Never leave raw `new` without an explicit owner.
3. **Cache Line Alignment**: All memory allocated for compute buffers must use 64-byte alignment via `hrm::aligned_alloc()`.
4. **Pure C-ABI Exports**: Keep `include/hrm/c_api/hrm_c_api.h` strictly compatible with pure C (no C++ templates, classes, or exceptions exported across the DLL boundary).

---

## 🔗 Related Notes
- [[Build & Toolchain Guide]] — Compiling and running tests.
- [[Master MOC]] — Vault index.
