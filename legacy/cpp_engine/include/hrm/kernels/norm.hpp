#pragma once

#include "hrm/core/tensor.hpp"

namespace hrm {
namespace kernels {

// RMSNorm: Y = (X / sqrt(mean(X^2) + eps)) * gamma
void rms_norm(
    const float* input,
    const float* gamma,
    float* output,
    int64_t rows,
    int64_t hidden_dim,
    float eps = 1e-6f
);

// Fused RMSNorm with Residual: output = rms_norm(x + residual), and residual = x + residual
void fused_add_rms_norm(
    float* x,
    const float* residual,
    const float* gamma,
    float* output,
    int64_t rows,
    int64_t hidden_dim,
    float eps = 1e-6f
);

// Tensor wrapper
void rms_norm_forward(const Tensor& X, const Tensor& gamma, Tensor& Y, float eps = 1e-6f);

} // namespace kernels
} // namespace hrm
