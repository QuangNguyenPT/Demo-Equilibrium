#define HRM_BUILD_SHARED
#include "hrm/c_api/hrm_c_api.h"
#include "hrm/core/tensor.hpp"
#include "hrm/core/allocator.hpp"
#include "hrm/kernels/gemm.hpp"
#include "hrm/kernels/norm.hpp"
#include "hrm/kernels/activation.hpp"
#include "hrm/kernels/rope.hpp"
#include "hrm/kernels/attention.hpp"
#include "hrm/kernels/hrm_loop.hpp"
#include "hrm/runtime/thread_pool.hpp"

#include <memory>
#include <iostream>

struct HRMEngineInternal {
    std::unique_ptr<hrm::ArenaAllocator> arena;
    std::unique_ptr<hrm::runtime::ThreadPool> thread_pool;
};

extern "C" {

HRM_API hrm_engine_handle_t hrm_engine_create(size_t arena_capacity_mb, int32_t num_threads) {
    auto engine = new HRMEngineInternal();
    size_t bytes = arena_capacity_mb * 1024 * 1024;
    if (bytes == 0) bytes = 64 * 1024 * 1024; // Default 64MB arena
    engine->arena = std::make_unique<hrm::ArenaAllocator>(bytes);
    engine->thread_pool = std::make_unique<hrm::runtime::ThreadPool>(
        num_threads > 0 ? num_threads : std::thread::hardware_concurrency()
    );
    return static_cast<hrm_engine_handle_t>(engine);
}

HRM_API void hrm_engine_destroy(hrm_engine_handle_t engine) {
    if (!engine) return;
    delete static_cast<HRMEngineInternal*>(engine);
}

HRM_API void hrm_engine_reset_arena(hrm_engine_handle_t engine) {
    if (!engine) return;
    auto eng = static_cast<HRMEngineInternal*>(engine);
    if (eng->arena) {
        eng->arena->reset();
    }
}

HRM_API hrm_tensor_handle_t hrm_tensor_create(
    const int64_t* shape,
    int32_t dims,
    int32_t dtype,
    hrm_engine_handle_t engine_optional
) {
    hrm::Shape s(shape, shape + dims);
    hrm::IAllocator* alloc = nullptr;
    if (engine_optional) {
        auto eng = static_cast<HRMEngineInternal*>(engine_optional);
        alloc = eng->arena.get();
    }
    auto t = new hrm::Tensor(s, static_cast<hrm::DataType>(dtype), alloc);
    return static_cast<hrm_tensor_handle_t>(t);
}

HRM_API hrm_tensor_handle_t hrm_tensor_wrap(
    void* external_data,
    const int64_t* shape,
    int32_t dims,
    int32_t dtype
) {
    hrm::Shape s(shape, shape + dims);
    auto t = new hrm::Tensor(s, external_data, static_cast<hrm::DataType>(dtype));
    return static_cast<hrm_tensor_handle_t>(t);
}

HRM_API void hrm_tensor_destroy(hrm_tensor_handle_t tensor) {
    if (!tensor) return;
    delete static_cast<hrm::Tensor*>(tensor);
}

HRM_API void* hrm_tensor_data_ptr(hrm_tensor_handle_t tensor) {
    if (!tensor) return nullptr;
    return static_cast<hrm::Tensor*>(tensor)->raw_data();
}

HRM_API int64_t hrm_tensor_numel(hrm_tensor_handle_t tensor) {
    if (!tensor) return 0;
    return static_cast<hrm::Tensor*>(tensor)->numel();
}

HRM_API size_t hrm_tensor_bytes(hrm_tensor_handle_t tensor) {
    if (!tensor) return 0;
    return static_cast<hrm::Tensor*>(tensor)->byte_size();
}

HRM_API int32_t hrm_op_gemm(
    hrm_tensor_handle_t tensor_a,
    hrm_tensor_handle_t tensor_b,
    hrm_tensor_handle_t tensor_c
) {
    if (!tensor_a || !tensor_b || !tensor_c) return -1;
    try {
        hrm::kernels::matmul(
            *static_cast<hrm::Tensor*>(tensor_a),
            *static_cast<hrm::Tensor*>(tensor_b),
            *static_cast<hrm::Tensor*>(tensor_c)
        );
        return 0;
    } catch (...) {
        return -2;
    }
}

HRM_API int32_t hrm_op_rms_norm(
    hrm_tensor_handle_t tensor_x,
    hrm_tensor_handle_t tensor_gamma,
    hrm_tensor_handle_t tensor_out,
    float eps
) {
    if (!tensor_x || !tensor_gamma || !tensor_out) return -1;
    try {
        hrm::kernels::rms_norm_forward(
            *static_cast<hrm::Tensor*>(tensor_x),
            *static_cast<hrm::Tensor*>(tensor_gamma),
            *static_cast<hrm::Tensor*>(tensor_out),
            eps
        );
        return 0;
    } catch (...) {
        return -2;
    }
}

HRM_API int32_t hrm_op_swiglu(
    hrm_tensor_handle_t tensor_gate,
    hrm_tensor_handle_t tensor_up,
    hrm_tensor_handle_t tensor_out
) {
    if (!tensor_gate || !tensor_up || !tensor_out) return -1;
    try {
        hrm::kernels::swiglu_forward(
            *static_cast<hrm::Tensor*>(tensor_gate),
            *static_cast<hrm::Tensor*>(tensor_up),
            *static_cast<hrm::Tensor*>(tensor_out)
        );
        return 0;
    } catch (...) {
        return -2;
    }
}

HRM_API int32_t hrm_op_rope(
    hrm_tensor_handle_t tensor_x,
    hrm_tensor_handle_t tensor_out,
    int64_t max_seq_len,
    float theta_base,
    int64_t start_pos
) {
    if (!tensor_x || !tensor_out) return -1;
    try {
        auto tx = static_cast<hrm::Tensor*>(tensor_x);
        auto tout = static_cast<hrm::Tensor*>(tensor_out);
        int64_t head_dim = tx->shape().back();
        hrm::kernels::RoPECache cache(max_seq_len, head_dim, theta_base);
        hrm::kernels::rope_forward(*tx, cache, *tout, start_pos);
        return 0;
    } catch (...) {
        return -2;
    }
}

HRM_API int32_t hrm_op_flash_attention(
    hrm_tensor_handle_t tensor_q,
    hrm_tensor_handle_t tensor_k,
    hrm_tensor_handle_t tensor_v,
    hrm_tensor_handle_t tensor_o,
    int32_t is_causal
) {
    if (!tensor_q || !tensor_k || !tensor_v || !tensor_o) return -1;
    try {
        hrm::kernels::flash_attention_forward(
            *static_cast<hrm::Tensor*>(tensor_q),
            *static_cast<hrm::Tensor*>(tensor_k),
            *static_cast<hrm::Tensor*>(tensor_v),
            *static_cast<hrm::Tensor*>(tensor_o),
            is_causal != 0
        );
        return 0;
    } catch (...) {
        return -2;
    }
}

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
) {
    if (!z_h || !z_l || !input_emb || !w_low || !u_low || !w_high || !u_high || !w_qhead) {
        return -1;
    }
    try {
        hrm::Tensor dummy_b;
        auto res = hrm::kernels::run_hrm_reasoning_loop(
            *static_cast<hrm::Tensor*>(z_h),
            *static_cast<hrm::Tensor*>(z_l),
            *static_cast<hrm::Tensor*>(input_emb),
            *static_cast<hrm::Tensor*>(w_low),
            *static_cast<hrm::Tensor*>(u_low),
            b_low ? *static_cast<hrm::Tensor*>(b_low) : dummy_b,
            *static_cast<hrm::Tensor*>(w_high),
            *static_cast<hrm::Tensor*>(u_high),
            b_high ? *static_cast<hrm::Tensor*>(b_high) : dummy_b,
            *static_cast<hrm::Tensor*>(w_qhead),
            b_qhead ? *static_cast<hrm::Tensor*>(b_qhead) : dummy_b,
            max_steps,
            low_steps,
            act_threshold
        );

        if (out_result) {
            out_result->steps_executed = res.steps_executed;
            out_result->halted_early = res.halted_early ? 1 : 0;
            out_result->halt_confidence = res.halt_confidence;
        }
        return 0;
    } catch (...) {
        return -2;
    }
}

HRM_API const char* hrm_version_string(void) {
    return "HRM-LLM Engine v1.0.0 (C++ Core)";
}

} // extern "C"
