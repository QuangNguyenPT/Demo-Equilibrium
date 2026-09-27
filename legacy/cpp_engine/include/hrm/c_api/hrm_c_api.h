#pragma once

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(HRM_BUILD_SHARED)
    #define HRM_API __declspec(dllexport)
  #else
    #define HRM_API __declspec(dllimport)
  #endif
#else
  #if __GNUC__ >= 4
    #define HRM_API __attribute__((visibility("default")))
  #else
    #define HRM_API
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* hrm_engine_handle_t;
typedef void* hrm_tensor_handle_t;

typedef struct {
    int64_t steps_executed;
    int32_t halted_early;
    float halt_confidence;
} hrm_reasoning_result_t;

// Engine Lifecycle
HRM_API hrm_engine_handle_t hrm_engine_create(size_t arena_capacity_mb, int32_t num_threads);
HRM_API void hrm_engine_destroy(hrm_engine_handle_t engine);
HRM_API void hrm_engine_reset_arena(hrm_engine_handle_t engine);

// Tensor Management
HRM_API hrm_tensor_handle_t hrm_tensor_create(
    const int64_t* shape,
    int32_t dims,
    int32_t dtype,
    hrm_engine_handle_t engine_optional
);
HRM_API hrm_tensor_handle_t hrm_tensor_wrap(
    void* external_data,
    const int64_t* shape,
    int32_t dims,
    int32_t dtype
);
HRM_API void hrm_tensor_destroy(hrm_tensor_handle_t tensor);
HRM_API void* hrm_tensor_data_ptr(hrm_tensor_handle_t tensor);
HRM_API int64_t hrm_tensor_numel(hrm_tensor_handle_t tensor);
HRM_API size_t hrm_tensor_bytes(hrm_tensor_handle_t tensor);

// Computational Kernels
HRM_API int32_t hrm_op_gemm(
    hrm_tensor_handle_t tensor_a,
    hrm_tensor_handle_t tensor_b,
    hrm_tensor_handle_t tensor_c
);

HRM_API int32_t hrm_op_rms_norm(
    hrm_tensor_handle_t tensor_x,
    hrm_tensor_handle_t tensor_gamma,
    hrm_tensor_handle_t tensor_out,
    float eps
);

HRM_API int32_t hrm_op_swiglu(
    hrm_tensor_handle_t tensor_gate,
    hrm_tensor_handle_t tensor_up,
    hrm_tensor_handle_t tensor_out
);

HRM_API int32_t hrm_op_rope(
    hrm_tensor_handle_t tensor_x,
    hrm_tensor_handle_t tensor_out,
    int64_t max_seq_len,
    float theta_base,
    int64_t start_pos
);

HRM_API int32_t hrm_op_flash_attention(
    hrm_tensor_handle_t tensor_q,
    hrm_tensor_handle_t tensor_k,
    hrm_tensor_handle_t tensor_v,
    hrm_tensor_handle_t tensor_o,
    int32_t is_causal
);

// High-level Hierarchical Reasoning Op
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

// Version / Status
HRM_API const char* hrm_version_string(void);

#ifdef __cplusplus
}
#endif
