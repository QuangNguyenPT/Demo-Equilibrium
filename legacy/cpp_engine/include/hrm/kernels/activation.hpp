#pragma once

#include "hrm/core/tensor.hpp"

namespace hrm {
namespace kernels {

// SiLU: y = x * sigmoid(x) = x / (1 + exp(-x))
void silu(const float* input, float* output, int64_t n);

// GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu(const float* input, float* output, int64_t n);

// SwiGLU: output = (gate * sigmoid(gate)) * up
// gate: (rows, intermediate_dim), up: (rows, intermediate_dim), output: (rows, intermediate_dim)
void swiglu(
    const float* gate,
    const float* up,
    float* output,
    int64_t rows,
    int64_t intermediate_dim
);

void swiglu_forward(const Tensor& Gate, const Tensor& Up, Tensor& Output);

} // namespace kernels
} // namespace hrm
