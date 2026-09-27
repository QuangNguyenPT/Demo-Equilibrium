#pragma once

#include "hrm/core/tensor.hpp"
#include <vector>

namespace hrm {
namespace runtime {

// Manages Key-Value states across autoregressive generation steps
class KVCache {
public:
    KVCache(int64_t num_layers, int64_t batch_size, int64_t num_heads, int64_t max_seq_len, int64_t head_dim);

    // Appends new K and V tokens at current sequence step
    void append(int64_t layer_idx, const Tensor& k_new, const Tensor& v_new);

    // Retrieve full past K and V up to current sequence length
    Tensor get_k(int64_t layer_idx) const;
    Tensor get_v(int64_t layer_idx) const;

    int64_t current_len() const { return current_len_; }
    int64_t max_seq_len() const { return max_seq_len_; }
    void reset();

private:
    int64_t num_layers_;
    int64_t batch_size_;
    int64_t num_heads_;
    int64_t max_seq_len_;
    int64_t head_dim_;
    int64_t current_len_{0};

    // Stored as (num_layers, batch_size, num_heads, max_seq_len, head_dim)
    std::vector<Tensor> k_layers_;
    std::vector<Tensor> v_layers_;
};

} // namespace runtime
} // namespace hrm
