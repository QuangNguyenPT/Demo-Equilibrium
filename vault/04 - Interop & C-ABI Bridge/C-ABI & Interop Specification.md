---
title: C-ABI & Interop Specification
type: interop
tags:
  - interop
  - c-abi
  - ctypes
  - bridge
status: superseded-not-implemented
created: 2026-09-26
updated: 2026-09-28
superseded_by: "[[HRM-X Native Backend]]"
relates_to:
  - "[[ADR-001 Hybrid C++ Engine with Python Blueprint]]"
  - "[[HRM-X Specification v0.4 - Operator Inference over Structured Situations]]"
---

# 🔌 C-ABI & Interop Specification

> [!WARNING] Not implemented as described
> This note describes the ADR-001 C++/Python hybrid engine, which was **never built**; ADR-003 moved it to `legacy/`. The live C++ is only `hrm_x/native/` (`egraph.cpp`, `grid.cpp`) behind the `hrmx.h` C ABI.
>
> **Superseded as the design target 2026-09-29.** At the time of writing this note the current model was HRM-X v0.4 (spec v0.4, ADR-008); the current target is **HRM-X v0.5** (`hrm_x/`), spec v0.5 + ADR-010. Kept as design history.
> Attribution: opencode/stealth/space-bunny-alpha (space-bunny-alpha), 2026-09-29.

---

## 🟢 The only real C ABI: `hrm_x/native/include/hrmx.h`

The reasoning below (pure C-ABI over C++ name mangling) **is** the right convention, and ADR-008
kept it: "C++ backend behind a C ABI (ADR-001 conventions)." What changed is the surface.

- **Header**: `hrm_x/native/include/hrmx.h`, 78 lines, `extern "C"`, `HX_API` (`__declspec(dllexport)`
  on Windows, default visibility elsewhere).
- **Loader**: `hrm_x/native/__init__.py` — `ctypes.CDLL` on `hrm_x/native/bin/hrmx_native.dll`,
  with the signature table in `_load()`. It falls back to the Python reference if the DLL is
  absent, and `HRMX_BACKEND=python` forces the reference. See [[HRM-X Native Backend]].
- **Zero copy**: arrays are passed as raw `const int32_t*` / `int32_t*` pointers into
  caller-owned NumPy buffers. Nothing is copied, and the library owns no data.
- **Opaque handle**: `hx_egraph*` is the one opaque type, freed with `hx_eg_free`.

**E-graph family** — `hx_eg_*` (union-find, hash-cons, rebuild, e-matching, explanations):

| Export | Purpose |
| :--- | :--- |
| `hx_eg_new` / `hx_eg_free` | lifecycle; the constant operator is fixed at construction |
| `hx_eg_add` / `hx_eg_lookup` | hash-consing: add an e-node, or find the existing one for `(op, args, payload)` |
| `hx_eg_find` / `hx_eg_union` | union-find; `union` takes a reason id and its premises, and **refuses** (returns −1) a merge of two classes holding different constants |
| `hx_eg_rebuild` / `hx_eg_take_conflicts` | congruence closure, with refused congruences queued as conflicts to be blamed |
| `hx_eg_classes` / `hx_eg_members` / `hx_eg_node` / `hx_eg_num_nodes` / `hx_eg_const_node` | read-back for verification and the writer |
| `hx_eg_ematch` / `hx_eg_snapshot_index` | e-matching over a pattern preorder code, with a per-saturation-round index snapshot |
| `hx_eg_explain` | reasons on the proof path `a ~ b` (invariant I3) |

**Grid family** — the grid entry points share the `hx_` prefix (`hx_version`, then):

| Export | Purpose |
| :--- | :--- |
| `hx_components` | 4-connected same-colour components (0 = background); returns the object count |
| `hx_object_step` | one object-level step (select: all / by colour / largest / smallest; act: keep, delete, recolour, move, flip). Returns 1 and leaves `out` untouched when the step is invalid: an object would leave the grid, or `strict` is set and the selector is ambiguous / two objects collide / nothing is selected. |

**What stayed in Python on purpose**: exact `Fraction` arithmetic. Arbitrary precision is the
guarantee the whole model rests on, so it was not ported (ADR-008, Rejected).

**How it is tested**: `tests/test_x_native.py` checks the native path against the Python
reference on random inputs. Measured: identical results, grid **3.79×**, arithmetic **1.47×**
(`experiments/results/x_native.json`).

---

## 🔴 What was never implemented

Everything below this line describes `hrm_c_api.h`, `hrm_engine_create`, `hrm_op_gemm`,
`hrm_op_rms_norm`, `hrm_op_flash_attention`, `hrm_op_reasoning_loop`, `hrm_tensor_create`,
`hrm_tensor_wrap`, `hrm_tensor_data_ptr`, `hrm_engine_reset_arena` and `bridge.wrap_numpy`.
**None of these ever existed in the live tree.** `include/hrm/c_api/hrm_c_api.h` is not in the
repo; the header and these entry points exist only under `legacy/cpp_engine/`, dead, and no
Python module in `hrm/`, `hrm_eqb/` or `hrm_x/` calls them. There is no `EngineBridge`, no
opaque tensor-handle scheme, and no zero-copy NumPy `wrap_numpy` — the real interop is the
`int32_t*` pointer passing documented above.

The design record is kept verbatim below because the *rationale* (never export C++ types across
a DLL boundary) is still correct and is still in force in `hrmx.h`.

---

## 🛡️ Why Pure C-ABI Over C++ Name Mangling?

C++ does not define a standard Application Binary Interface (ABI). Binaries compiled with MSVC, Clang++, or GCC cannot reliably link against one another when C++ classes, standard library templates (`std::vector`, `std::string`), or virtual tables are exposed across DLL boundaries.

To solve this, our engine exposes an `extern "C"` interface in `include/hrm/c_api/hrm_c_api.h`:
- Opaque pointer handles (`hrm_engine_handle_t`, `hrm_tensor_handle_t`).
- Plain C primitive types (`int64_t`, `int32_t`, `float`, `size_t`).
- Simple C `struct` definitions for returning execution status (`hrm_reasoning_result_t`).
- Compatible with Python `ctypes`, `cffi`, and `pybind11`.

---

## 📋 Core C-ABI Export Signatures

```c
// 1. Engine Lifecycle
HRM_API hrm_engine_handle_t hrm_engine_create(size_t arena_capacity_mb, int32_t num_threads);
HRM_API void hrm_engine_destroy(hrm_engine_handle_t engine);
HRM_API void hrm_engine_reset_arena(hrm_engine_handle_t engine);

// 2. Tensor Memory
HRM_API hrm_tensor_handle_t hrm_tensor_create(const int64_t* shape, int32_t dims, int32_t dtype, hrm_engine_handle_t engine);
HRM_API hrm_tensor_handle_t hrm_tensor_wrap(void* external_data, const int64_t* shape, int32_t dims, int32_t dtype);
HRM_API void* hrm_tensor_data_ptr(hrm_tensor_handle_t tensor);
HRM_API void hrm_tensor_destroy(hrm_tensor_handle_t tensor);

// 3. High-Speed Kernels
HRM_API int32_t hrm_op_gemm(hrm_tensor_handle_t a, hrm_tensor_handle_t b, hrm_tensor_handle_t c);
HRM_API int32_t hrm_op_rms_norm(hrm_tensor_handle_t x, hrm_tensor_handle_t gamma, hrm_tensor_handle_t out, float eps);
HRM_API int32_t hrm_op_flash_attention(hrm_tensor_handle_t q, hrm_tensor_handle_t k, hrm_tensor_handle_t v, hrm_tensor_handle_t o, int32_t is_causal);

// 4. Hierarchical Reasoning Engine Op
HRM_API int32_t hrm_op_reasoning_loop(
    hrm_tensor_handle_t z_h,
    hrm_tensor_handle_t z_l,
    hrm_tensor_handle_t input_emb,
    hrm_tensor_handle_t w_low,
    hrm_tensor_handle_t u_low,
    hrm_tensor_handle_t b_low,
    hrm_tensor_handle_t w_high,
    hrm_tensor_handle_t u_high,
    hrm_tensor_handle_t b_high,
    hrm_tensor_handle_t w_qhead,
    hrm_tensor_handle_t b_qhead,
    int64_t max_steps,
    int64_t low_steps,
    float act_threshold,
    hrm_reasoning_result_t* out_result
);
```

---

## ⚡ Zero-Copy NumPy Sharing via `wrap_numpy`

```python
# Create NumPy array in Python
data = np.ones((1, 256), dtype=np.float32)

# Pass raw memory pointer directly to C++ without copying!
c_tensor = bridge.wrap_numpy(data)

# C++ engine modifies the buffer in-place
bridge.run_reasoning_loop(z_h=c_tensor, ...)

# Python immediately reads updated values without deserialization
print(data[0, :5])
```

---

## 🔗 Related Notes
- [[HRM-X Native Backend]] — **the C ABI that exists** (`hrmx.h`, `egraph.cpp`, `grid.cpp`).
- [[Engine Architecture & Memory Model]] — Memory allocator behind the C-ABI. *(never implemented)*
- [[Hybrid System Philosophy]] — Architectural motivation. *(historical)*
- [[HRM Recurrent Stepper & ACT]] — C++ implementation of the reasoning op. *(never implemented)*
- [[Build & Toolchain Guide]] — how the real DLL is compiled.
