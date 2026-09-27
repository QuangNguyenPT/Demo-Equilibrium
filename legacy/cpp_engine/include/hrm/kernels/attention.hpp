#pragma once

#include "hrm/core/tensor.hpp"
#include <cstdint>

namespace hrm {
namespace kernels {

// Fused Multi-Head Attention with online softmax (FlashAttention CPU kernel)
// RealFormer attention-residual extension: pass prev_scores to add cross-layer
// residual attention (pre-softmax scores from previous layer, shape B*H*T_q*T_k).
// Set prev_scores=nullptr and scores_out=nullptr for standard (first-layer) use.
//
// Q,K,V,O: (B, H, T, D)
void flash_attention_f32(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int64_t B,
    int64_t H,
    int64_t T_q,
    int64_t T_k,
    int64_t D,
    bool is_causal = true,
    const float* prev_scores = nullptr,  // (B, H, T_q, T_k) pre-softmax scores from layer l-1
    float* scores_out        = nullptr   // (B, H, T_q, T_k) current scores stored for layer l+1
);

void flash_attention_forward(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    Tensor& O,
    bool is_causal = true,
    const Tensor* prev_scores = nullptr, // optional attention residual input
    Tensor* scores_out        = nullptr  // optional attention residual output
);

} // namespace kernels
} // namespace hrm
