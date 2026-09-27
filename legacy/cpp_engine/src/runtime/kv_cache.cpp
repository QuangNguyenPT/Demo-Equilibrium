#include "hrm/runtime/kv_cache.hpp"
#include <cstring>
#include <stdexcept>

namespace hrm {
namespace runtime {

KVCache::KVCache(int64_t num_layers, int64_t batch_size, int64_t num_heads, int64_t max_seq_len, int64_t head_dim)
    : num_layers_(num_layers), batch_size_(batch_size), num_heads_(num_heads),
      max_seq_len_(max_seq_len), head_dim_(head_dim), current_len_(0) {

    k_layers_.reserve(num_layers_);
    v_layers_.reserve(num_layers_);

    for (int64_t l = 0; l < num_layers_; ++l) {
        k_layers_.emplace_back(Shape{batch_size_, num_heads_, max_seq_len_, head_dim_});
        v_layers_.emplace_back(Shape{batch_size_, num_heads_, max_seq_len_, head_dim_});
        k_layers_.back().zero();
        v_layers_.back().zero();
    }
}

void KVCache::append(int64_t layer_idx, const Tensor& k_new, const Tensor& v_new) {
    if (layer_idx < 0 || layer_idx >= num_layers_) {
        throw std::out_of_range("[KVCACHE] Invalid layer index: " + std::to_string(layer_idx));
    }

    const auto& s_new = k_new.shape();
    // k_new expected: (batch_size, num_heads, new_tokens, head_dim)
    int64_t new_tokens = s_new[2];

    if (current_len_ + new_tokens > max_seq_len_) {
        throw std::runtime_error("[KVCACHE] Context limit exceeded: max " + std::to_string(max_seq_len_));
    }

    float* k_buf = k_layers_[layer_idx].data<float>();
    float* v_buf = v_layers_[layer_idx].data<float>();

    const float* k_src = k_new.data<float>();
    const float* v_src = v_new.data<float>();

    for (int64_t b = 0; b < batch_size_; ++b) {
        for (int64_t h = 0; h < num_heads_; ++h) {
            for (int64_t t = 0; t < new_tokens; ++t) {
                int64_t dst_offset = ((b * num_heads_ + h) * max_seq_len_ + (current_len_ + t)) * head_dim_;
                int64_t src_offset = ((b * num_heads_ + h) * new_tokens + t) * head_dim_;

                std::memcpy(k_buf + dst_offset, k_src + src_offset, head_dim_ * sizeof(float));
                std::memcpy(v_buf + dst_offset, v_src + src_offset, head_dim_ * sizeof(float));
            }
        }
    }

    if (layer_idx == num_layers_ - 1) {
        current_len_ += new_tokens;
    }
}

Tensor KVCache::get_k(int64_t layer_idx) const {
    if (layer_idx < 0 || layer_idx >= num_layers_) {
        throw std::out_of_range("[KVCACHE] Invalid layer index");
    }
    // Return view up to current_len_
    Tensor sliced({batch_size_, num_heads_, current_len_, head_dim_});
    float* dst = sliced.data<float>();
    const float* src = k_layers_[layer_idx].data<float>();

    for (int64_t b = 0; b < batch_size_; ++b) {
        for (int64_t h = 0; h < num_heads_; ++h) {
            int64_t dst_offset = (b * num_heads_ + h) * current_len_ * head_dim_;
            int64_t src_offset = (b * num_heads_ + h) * max_seq_len_ * head_dim_;
            std::memcpy(dst + dst_offset, src + src_offset, current_len_ * head_dim_ * sizeof(float));
        }
    }
    return sliced;
}

Tensor KVCache::get_v(int64_t layer_idx) const {
    if (layer_idx < 0 || layer_idx >= num_layers_) {
        throw std::out_of_range("[KVCACHE] Invalid layer index");
    }
    Tensor sliced({batch_size_, num_heads_, current_len_, head_dim_});
    float* dst = sliced.data<float>();
    const float* src = v_layers_[layer_idx].data<float>();

    for (int64_t b = 0; b < batch_size_; ++b) {
        for (int64_t h = 0; h < num_heads_; ++h) {
            int64_t dst_offset = (b * num_heads_ + h) * current_len_ * head_dim_;
            int64_t src_offset = (b * num_heads_ + h) * max_seq_len_ * head_dim_;
            std::memcpy(dst + dst_offset, src + src_offset, current_len_ * head_dim_ * sizeof(float));
        }
    }
    return sliced;
}

void KVCache::reset() {
    current_len_ = 0;
    for (auto& k : k_layers_) k.zero();
    for (auto& v : v_layers_) v.zero();
}

} // namespace runtime
} // namespace hrm
