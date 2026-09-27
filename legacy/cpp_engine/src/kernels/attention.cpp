#include "hrm/kernels/attention.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace hrm {
namespace kernels {

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
    bool is_causal,
    const float* prev_scores,
    float* scores_out
) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t hq_off = (b * H + h) * T_q * D;
            int64_t hk_off = (b * H + h) * T_k * D;
            int64_t hs_off = (b * H + h) * T_q * T_k;

            const float* q_head = Q + hq_off;
            const float* k_head = K + hk_off;
            const float* v_head = V + hk_off;
            float*       o_head = O + hq_off;
            const float* prev_h = prev_scores ? prev_scores + hs_off : nullptr;
            float*       out_s  = scores_out  ? scores_out  + hs_off : nullptr;

            for (int64_t i = 0; i < T_q; ++i) {
                const float* q_vec = q_head + i * D;
                float*       o_vec = o_head + i * D;

                for (int64_t d = 0; d < D; ++d) o_vec[d] = 0.0f;

                float m_prev = -std::numeric_limits<float>::infinity();
                float l_prev = 0.0f;

                int64_t k_limit = is_causal ? std::min(i + 1, T_k) : T_k;

                for (int64_t j = 0; j < k_limit; ++j) {
                    const float* k_vec = k_head + j * D;
                    const float* v_vec = v_head + j * D;

                    float score = 0.0f;
                    #pragma omp simd reduction(+:score)
                    for (int64_t d = 0; d < D; ++d) {
                        score += q_vec[d] * k_vec[d];
                    }
                    score *= scale;

                    // RealFormer: add previous layer's pre-softmax score
                    if (prev_h != nullptr) {
                        score += prev_h[i * T_k + j];
                    }

                    // Store score for next layer (before softmax)
                    if (out_s != nullptr) {
                        out_s[i * T_k + j] = score;
                    }

                    float m_curr = std::max(m_prev, score);
                    float alpha  = std::exp(m_prev - m_curr);
                    float p      = std::exp(score  - m_curr);
                    float l_curr = l_prev * alpha + p;

                    #pragma omp simd
                    for (int64_t d = 0; d < D; ++d) {
                        o_vec[d] = o_vec[d] * alpha + p * v_vec[d];
                    }

                    m_prev = m_curr;
                    l_prev = l_curr;
                }

                if (l_prev > 0.0f) {
                    float inv_l = 1.0f / l_prev;
                    #pragma omp simd
                    for (int64_t d = 0; d < D; ++d) o_vec[d] *= inv_l;
                }
            }
        }
    }
}

void flash_attention_forward(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    Tensor& O,
    bool is_causal,
    const Tensor* prev_scores,
    Tensor* scores_out
) {
    const auto& sQ = Q.shape();
    const auto& sK = K.shape();

    if (sQ.size() != 4 || sK.size() != 4) {
        throw std::invalid_argument("[ATTENTION] Expected 4D tensors: (B, H, T, D).");
    }

    int64_t B   = sQ[0];
    int64_t H   = sQ[1];
    int64_t T_q = sQ[2];
    int64_t D   = sQ[3];
    int64_t T_k = sK[2];

    const float* prev_ptr = (prev_scores && prev_scores->raw_data())
                            ? prev_scores->data<float>() : nullptr;
    float* out_ptr        = (scores_out && scores_out->raw_data())
                            ? scores_out->data<float>() : nullptr;

    flash_attention_f32(
        Q.data<float>(),
        K.data<float>(),
        V.data<float>(),
        O.data<float>(),
        B, H, T_q, T_k, D, is_causal,
        prev_ptr, out_ptr
    );
}

} // namespace kernels
} // namespace hrm
