#include "hrm/kernels/rope.hpp"
#include <cmath>
#include <stdexcept>

namespace hrm {
namespace kernels {

RoPECache::RoPECache(int64_t max_seq_len, int64_t head_dim, float theta_base)
    : max_seq_len_(max_seq_len), head_dim_(head_dim),
      cos_table_({max_seq_len, head_dim / 2}),
      sin_table_({max_seq_len, head_dim / 2}) {

    float* cos_ptr = cos_table_.data<float>();
    float* sin_ptr = sin_table_.data<float>();
    int64_t half_dim = head_dim / 2;

    for (int64_t pos = 0; pos < max_seq_len; ++pos) {
        for (int64_t i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(theta_base, static_cast<float>(2 * i) / static_cast<float>(head_dim));
            float val = static_cast<float>(pos) * freq;
            cos_ptr[pos * half_dim + i] = std::cos(val);
            sin_ptr[pos * half_dim + i] = std::sin(val);
        }
    }
}

void apply_rope(
    const float* input,
    float* output,
    const float* cos_table,
    const float* sin_table,
    int64_t batch_size,
    int64_t num_heads,
    int64_t seq_len,
    int64_t head_dim,
    int64_t start_pos
) {
    int64_t half_dim = head_dim / 2;

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t t = 0; t < seq_len; ++t) {
                int64_t pos = start_pos + t;
                const float* cos_row = cos_table + pos * half_dim;
                const float* sin_row = sin_table + pos * half_dim;

                int64_t offset = ((b * num_heads + h) * seq_len + t) * head_dim;
                const float* in_vec = input + offset;
                float* out_vec = output + offset;

                #pragma omp simd
                for (int64_t i = 0; i < half_dim; ++i) {
                    float x0 = in_vec[i];
                    float x1 = in_vec[i + half_dim];
                    float c = cos_row[i];
                    float s = sin_row[i];

                    // Complex multiplication (x0 + i*x1) * (cos + i*sin)
                    out_vec[i] = x0 * c - x1 * s;
                    out_vec[i + half_dim] = x0 * s + x1 * c;
                }
            }
        }
    }
}

void rope_forward(
    const Tensor& X,
    const RoPECache& cache,
    Tensor& Y,
    int64_t start_pos
) {
    const auto& s = X.shape();
    // Expected (B, num_heads, seq_len, head_dim)
    int64_t B = s[0];
    int64_t num_heads = s[1];
    int64_t seq_len = s[2];
    int64_t head_dim = s[3];

    apply_rope(
        X.data<float>(),
        Y.data<float>(),
        cache.cos_table().data<float>(),
        cache.sin_table().data<float>(),
        B, num_heads, seq_len, head_dim, start_pos
    );
}

} // namespace kernels
} // namespace hrm
