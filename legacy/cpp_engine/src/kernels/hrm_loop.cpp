#include "hrm/kernels/hrm_loop.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace hrm {
namespace kernels {

// ── helpers ────────────────────────────────────────────────────────────────

inline float sigmoid_scalar(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float tanh_scalar(float x) {
    return std::tanh(x);
}

static float l2_delta_per_dim(const float* a, const float* b, int64_t D) {
    float acc = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
        float diff = a[d] - b[d];
        acc += diff * diff;
    }
    return std::sqrt(acc) / static_cast<float>(D);
}

static float vec_variance(const float* v, int64_t D) {
    float mean = 0.0f;
    for (int64_t d = 0; d < D; ++d) mean += v[d];
    mean /= static_cast<float>(D);
    float var = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
        float diff = v[d] - mean;
        var += diff * diff;
    }
    return var / static_cast<float>(D);
}

// 9-feature Q-head: W_q(2,9) @ features(9) + b_q(2) → [q_halt, q_cont]
// Feature vector mirrors PEM qhead.py featurize()
static void eval_qhead_9feat(
    const ControllerFeatures& f,
    const float* w_q,  // (2, 9)
    const float* b_q,  // (2) or nullptr
    float& q_halt,
    float& q_cont
) {
    float feat[9] = {
        std::min(f.residual, 1.0f),
        std::tanh(100.0f * f.residual_slope),
        std::min(f.energy, 2.0f) / 2.0f,
        f.diversity,
        f.memory_confidence,
        f.verification_margin,
        std::min(static_cast<float>(f.plateau_count), 20.0f) / 20.0f,
        std::min(static_cast<float>(f.restarts_used), 10.0f) / 10.0f,
        1.0f  // bias term
    };

    q_halt = b_q ? b_q[0] : 0.0f;
    q_cont = b_q ? b_q[1] : 0.0f;
    for (int k = 0; k < 9; ++k) {
        q_halt += w_q[0 * 9 + k] * feat[k];
        q_cont += w_q[1 * 9 + k] * feat[k];
    }
}

static void perturb_tensor(float* z, int64_t n, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, sigma);
    for (int64_t i = 0; i < n; ++i) {
        z[i] += dist(rng);
    }
}

// ── GRU cell ──────────────────────────────────────────────────────────────

void gru_cell_forward(
    const float* x,
    const float* h_prev,
    float* h_next,
    const float* W,
    const float* U,
    const float* b,
    int64_t batch_size,
    int64_t hidden_dim
) {
    int64_t D = hidden_dim;

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const float* x_vec  = x     + batch * D;
        const float* h_vec  = h_prev + batch * D;
        float*       out_vec = h_next + batch * D;

        for (int64_t d = 0; d < D; ++d) {
            float w_r_x = 0.0f, u_r_h = 0.0f;
            #pragma omp simd reduction(+:w_r_x, u_r_h)
            for (int64_t k = 0; k < D; ++k) {
                w_r_x += W[d * D + k] * x_vec[k];
                u_r_h += U[d * D + k] * h_vec[k];
            }
            float r_val = sigmoid_scalar(w_r_x + u_r_h + (b ? b[d] : 0.0f));

            float w_z_x = 0.0f, u_z_h = 0.0f;
            #pragma omp simd reduction(+:w_z_x, u_z_h)
            for (int64_t k = 0; k < D; ++k) {
                w_z_x += W[(D + d) * D + k] * x_vec[k];
                u_z_h += U[(D + d) * D + k] * h_vec[k];
            }
            float z_val = sigmoid_scalar(w_z_x + u_z_h + (b ? b[D + d] : 0.0f));

            float w_n_x = 0.0f, u_n_h = 0.0f;
            #pragma omp simd reduction(+:w_n_x, u_n_h)
            for (int64_t k = 0; k < D; ++k) {
                w_n_x += W[(2 * D + d) * D + k] * x_vec[k];
                u_n_h += U[(2 * D + d) * D + k] * h_vec[k];
            }
            float n_val = tanh_scalar(w_n_x + (b ? b[2 * D + d] : 0.0f) + r_val * u_n_h);

            out_vec[d] = (1.0f - z_val) * n_val + z_val * h_vec[d];
        }
    }
}

// ── Main reasoning loop ───────────────────────────────────────────────────

HRMExecutionResult run_hrm_reasoning_loop(
    Tensor& z_h,
    Tensor& z_l,
    const Tensor& input_emb,
    const Tensor& W_low,
    const Tensor& U_low,
    const Tensor& b_low,
    const Tensor& W_high,
    const Tensor& U_high,
    const Tensor& b_high,
    const Tensor& W_qhead,
    const Tensor& b_qhead,
    const HRMLoopConfig& cfg
) {
    HRMExecutionResult result;
    const int64_t B = z_h.shape()[0];
    const int64_t D = z_h.shape()[1];

    std::vector<float> inj(B * D);
    std::vector<float> z_l_tmp(B * D);
    std::vector<float> z_h_tmp(B * D);
    std::vector<float> z_h_prev(B * D, 0.0f);

    const float* w_low_ptr  = W_low.data<float>();
    const float* u_low_ptr  = U_low.data<float>();
    const float* b_low_ptr  = b_low.raw_data()   ? b_low.data<float>()   : nullptr;
    const float* w_high_ptr = W_high.data<float>();
    const float* u_high_ptr = U_high.data<float>();
    const float* b_high_ptr = b_high.raw_data()  ? b_high.data<float>()  : nullptr;
    const float* w_q_ptr    = W_qhead.data<float>();
    const float* b_q_ptr    = b_qhead.raw_data() ? b_qhead.data<float>() : nullptr;

    std::mt19937 rng(42);
    ControllerFeatures feat;
    float best_energy   = 1.0f;
    float prev_residual = 0.0f;
    int   plateau_count = 0;
    int   restarts      = 0;

    for (int64_t step = 0; step < cfg.max_reasoning_steps; ++step) {
        result.steps_executed = step + 1;

        // Save z_h before update for residual computation
        const float* zh_ptr = z_h.data<float>();
        std::copy(zh_ptr, zh_ptr + B * D, z_h_prev.begin());

        // 1. Injection: inj = z_h + input_emb
        const float* in_ptr = input_emb.data<float>();
        #pragma omp simd
        for (int64_t i = 0; i < B * D; ++i) {
            inj[i] = zh_ptr[i] + in_ptr[i];
        }

        // 2. Low-level executor sub-steps
        for (int64_t sub = 0; sub < cfg.low_level_substeps; ++sub) {
            gru_cell_forward(inj.data(), z_l.data<float>(), z_l_tmp.data(),
                             w_low_ptr, u_low_ptr, b_low_ptr, B, D);
            z_l.copy_from(z_l_tmp.data(), z_l.byte_size());
        }

        // 3. High-level planner update
        gru_cell_forward(z_l.data<float>(), z_h.data<float>(), z_h_tmp.data(),
                         w_high_ptr, u_high_ptr, b_high_ptr, B, D);
        z_h.copy_from(z_h_tmp.data(), z_h.byte_size());

        // 4. Compute controller features (batch element 0)
        const float* new_zh = z_h.data<float>();
        feat.residual       = l2_delta_per_dim(new_zh, z_h_prev.data(), D);
        feat.residual_slope = feat.residual - prev_residual;
        prev_residual       = feat.residual;
        feat.diversity      = vec_variance(new_zh, D);

        // 5. Evaluate 9-feature Q-head
        float q_halt_raw, q_cont_raw;
        eval_qhead_9feat(feat, w_q_ptr, b_q_ptr, q_halt_raw, q_cont_raw);
        float max_q  = std::max(q_halt_raw, q_cont_raw);
        float eh     = std::exp(q_halt_raw - max_q);
        float ec     = std::exp(q_cont_raw - max_q);
        float halt_p = eh / (eh + ec);

        feat.energy              = 1.0f - halt_p;
        feat.verification_margin = halt_p - cfg.act_threshold;
        feat.plateau_count       = plateau_count;
        feat.restarts_used       = restarts;
        result.halt_confidence   = halt_p;

        // Plateau tracker
        if (feat.energy < best_energy - 1e-4f) {
            best_energy   = feat.energy;
            plateau_count = 0;
        } else {
            ++plateau_count;
        }
        feat.plateau_count = plateau_count;

        // 6. Hard-rail conjunctive halt (PEM: ACCEPT != high Q alone)
        if (cfg.use_hard_rails) {
            bool kin_ok = feat.residual < cfg.eps_kinetic;
            bool sem_ok = feat.energy   < cfg.eps_energy;
            bool ver_ok = halt_p        >= cfg.tau_verify;

            if (kin_ok && sem_ok && ver_ok) {
                result.halted_early   = true;
                result.restarts_used  = restarts;
                result.final_features = feat;
                return result;
            }

            // False-attractor or plateau → RESTART
            bool false_attractor = kin_ok && !sem_ok;
            bool plateau_stuck   = plateau_count >= cfg.plateau_cycles;

            if ((false_attractor || plateau_stuck) && restarts < cfg.restart_budget) {
                float sigma = cfg.sigma0_restart
                              * std::pow(cfg.gamma_restart, static_cast<float>(restarts));
                perturb_tensor(z_h.data<float>(), B * D, sigma, rng);
                perturb_tensor(z_l.data<float>(), B * D, sigma, rng);
                ++restarts;
                feat.restarts_used = restarts;
                plateau_count      = 0;
                best_energy        = 1.0f;
                continue;
            }

            // Budget exhausted without semantic convergence → ABSTAIN
            if (restarts >= cfg.restart_budget && !sem_ok) {
                result.aborted        = true;
                result.restarts_used  = restarts;
                result.final_features = feat;
                return result;
            }
        } else {
            // Legacy simple threshold halt
            if (halt_p >= cfg.act_threshold) {
                result.halted_early   = true;
                result.restarts_used  = restarts;
                result.final_features = feat;
                return result;
            }
        }
    }

    result.restarts_used  = restarts;
    result.final_features = feat;
    return result;
}

} // namespace kernels
} // namespace hrm
