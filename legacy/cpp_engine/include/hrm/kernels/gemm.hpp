#pragma once

#include "hrm/core/tensor.hpp"

namespace hrm {
namespace kernels {

// General Matrix Multiply: C = alpha * A * B + beta * C
// A: (M, K), B: (K, N), C: (M, N)
void gemm_f32(
    const float* A,
    const float* B,
    float* C,
    int64_t M,
    int64_t N,
    int64_t K,
    float alpha = 1.0f,
    float beta = 0.0f
);

// High-level tensor wrapper
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

// Fused Linear Layer: Y = X * W^T + bias
// X: (B, in_dim), W: (out_dim, in_dim), bias: (out_dim), Y: (B, out_dim)
void linear(
    const Tensor& X,
    const Tensor& W,
    const Tensor* bias,
    Tensor& Y
);

} // namespace kernels
} // namespace hrm
