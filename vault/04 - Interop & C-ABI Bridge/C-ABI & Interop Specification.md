---
title: C-ABI & Interop Specification
type: interop
tags:
  - interop
  - c-abi
  - ctypes
  - bridge
status: active
created: 2026-09-26
---

# 🔌 C-ABI & Interop Specification

The interoperability layer bridges high-level Python code with native C++ execution without introducing runtime compilation requirements or C++ ABI mismatches across compilers.

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
- [[Engine Architecture & Memory Model]] — Memory allocator behind the C-ABI.
- [[Hybrid System Philosophy]] — Architectural motivation.
- [[HRM Recurrent Stepper & ACT]] — C++ implementation of the reasoning op.
