#pragma once

#include "hrm/core/tensor.hpp"

namespace hrm {
namespace kernels {

// Precomputes cos and sin tables for max_seq_len and head_dim
class RoPECache {
public:
    RoPECache(int64_t max_seq_len, int64_t head_dim, float theta_base = 10000.0f);

    const Tensor& cos_table() const { return cos_table_; }
    const Tensor& sin_table() const { return sin_table_; }
    int64_t head_dim() const { return head_dim_; }
    int64_t max_seq_len() const { return max_seq_len_; }

private:
    int64_t max_seq_len_;
    int64_t head_dim_;
    Tensor cos_table_;
    Tensor sin_table_;
};

// Applies RoPE rotation in-place or out-of-place to Q or K:
// input: (batch_size, num_heads, seq_len, head_dim)
void apply_rope(
    const float* input,
    float* output,
    const float* cos_row,
    const float* sin_row,
    int64_t batch_size,
    int64_t num_heads,
    int64_t seq_len,
    int64_t head_dim,
    int64_t start_pos = 0
);

void rope_forward(
    const Tensor& X,
    const RoPECache& cache,
    Tensor& Y,
    int64_t start_pos = 0
);

} // namespace kernels
} // namespace hrm
