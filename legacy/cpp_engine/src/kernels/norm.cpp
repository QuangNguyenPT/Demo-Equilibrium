#include "hrm/kernels/norm.hpp"
#include <cmath>
#include <stdexcept>

namespace hrm {
namespace kernels {

void rms_norm(
    const float* input,
    const float* gamma,
    float* output,
    int64_t rows,
    int64_t hidden_dim,
    float eps
) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* in_row = input + r * hidden_dim;
        float* out_row = output + r * hidden_dim;

        float sum_sq = 0.0f;
        #pragma omp simd reduction(+:sum_sq)
        for (int64_t d = 0; d < hidden_dim; ++d) {
            sum_sq += in_row[d] * in_row[d];
        }

        float variance = sum_sq / static_cast<float>(hidden_dim);
        float inv_std = 1.0f / std::sqrt(variance + eps);

        #pragma omp simd
        for (int64_t d = 0; d < hidden_dim; ++d) {
            out_row[d] = in_row[d] * inv_std * gamma[d];
        }
    }
}

void fused_add_rms_norm(
    float* x,
    const float* residual,
    const float* gamma,
    float* output,
    int64_t rows,
    int64_t hidden_dim,
    float eps
) {
    for (int64_t r = 0; r < rows; ++r) {
        float* x_row = x + r * hidden_dim;
        const float* res_row = residual + r * hidden_dim;
        float* out_row = output + r * hidden_dim;

        float sum_sq = 0.0f;
        #pragma omp simd reduction(+:sum_sq)
        for (int64_t d = 0; d < hidden_dim; ++d) {
            x_row[d] += res_row[d]; // In-place residual update
            sum_sq += x_row[d] * x_row[d];
        }

        float variance = sum_sq / static_cast<float>(hidden_dim);
        float inv_std = 1.0f / std::sqrt(variance + eps);

        #pragma omp simd
        for (int64_t d = 0; d < hidden_dim; ++d) {
            out_row[d] = x_row[d] * inv_std * gamma[d];
        }
    }
}

void rms_norm_forward(const Tensor& X, const Tensor& gamma, Tensor& Y, float eps) {
    const auto& sX = X.shape();
    int64_t hidden_dim = sX.back();
    int64_t rows = X.numel() / hidden_dim;

    if (gamma.numel() != hidden_dim) {
        throw std::invalid_argument("[RMSNORM] Gamma size must match hidden_dim.");
    }

    rms_norm(X.data<float>(), gamma.data<float>(), Y.data<float>(), rows, hidden_dim, eps);
}

} // namespace kernels
} // namespace hrm
