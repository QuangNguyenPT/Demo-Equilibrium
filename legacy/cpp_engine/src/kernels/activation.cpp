#include "hrm/kernels/activation.hpp"
#include <cmath>
#include <stdexcept>

namespace hrm {
namespace kernels {

void silu(const float* input, float* output, int64_t n) {
    #pragma omp simd
    for (int64_t i = 0; i < n; ++i) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
}

void gelu(const float* input, float* output, int64_t n) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float COEFF = 0.044715f;

    #pragma omp simd
    for (int64_t i = 0; i < n; ++i) {
        float x = input[i];
        float inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void swiglu(
    const float* gate,
    const float* up,
    float* output,
    int64_t rows,
    int64_t intermediate_dim
) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* g_row = gate + r * intermediate_dim;
        const float* u_row = up + r * intermediate_dim;
        float* o_row = output + r * intermediate_dim;

        #pragma omp simd
        for (int64_t d = 0; d < intermediate_dim; ++d) {
            float g = g_row[d];
            float silu_g = g / (1.0f + std::exp(-g));
            o_row[d] = silu_g * u_row[d];
        }
    }
}

void swiglu_forward(const Tensor& Gate, const Tensor& Up, Tensor& Output) {
    if (Gate.numel() != Up.numel() || Gate.numel() != Output.numel()) {
        throw std::invalid_argument("[SWIGLU] Gate, Up, and Output tensor sizes must match.");
    }
    const auto& s = Gate.shape();
    int64_t intermediate_dim = s.back();
    int64_t rows = Gate.numel() / intermediate_dim;

    swiglu(Gate.data<float>(), Up.data<float>(), Output.data<float>(), rows, intermediate_dim);
}

} // namespace kernels
} // namespace hrm
