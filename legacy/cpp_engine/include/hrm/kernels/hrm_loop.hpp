#pragma once

#include "hrm/core/tensor.hpp"
#include <cstdint>

namespace hrm {
namespace kernels {

// Rich controller features computed each reasoning step (mirrors PEM BudgetController)
struct ControllerFeatures {
    float residual{0.0f};            // ||z_h_t - z_h_{t-1}|| / D  — kinetic convergence
    float residual_slope{0.0f};      // residual_t - residual_{t-1} (negative = converging)
    float energy{0.0f};              // 1 - halt_confidence  — semantic "unsolvedness"
    float diversity{0.0f};           // variance of z_h components — exploration proxy
    float memory_confidence{0.5f};   // placeholder; overridden by memory module when present
    float verification_margin{0.0f}; // halt_confidence - act_threshold
    int   plateau_count{0};          // steps without energy improvement
    int   restarts_used{0};          // number of annealed restarts consumed
};

struct HRMExecutionResult {
    int64_t steps_executed{0};
    bool    halted_early{false};
    bool    aborted{false};          // true if restart budget exhausted without convergence
    float   halt_confidence{0.0f};
    int     restarts_used{0};
    ControllerFeatures final_features;
};

// Tunable knobs for the reasoning loop (set once; passed into run_hrm_reasoning_loop)
struct HRMLoopConfig {
    int64_t max_reasoning_steps{8};
    int64_t low_level_substeps{2};
    float   act_threshold{0.5f};
    // PEM-inspired hard-rail conjunctive halt
    bool    use_hard_rails{true};
    float   eps_kinetic{1e-3f};      // residual < eps → kinetic converged
    float   eps_energy{0.15f};       // energy < eps  → semantic converged
    float   tau_verify{0.65f};       // halt_confidence >= tau → verified
    // Annealed restart parameters
    int     restart_budget{3};
    float   sigma0_restart{0.05f};   // initial perturbation std
    float   gamma_restart{0.6f};     // anneal factor per restart
    int     plateau_cycles{4};       // plateau_count threshold before forcing restart
};

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
    const Tensor& W_qhead,  // (2, 9) — operates on ControllerFeatures vector
    const Tensor& b_qhead,  // (2,)
    const HRMLoopConfig& cfg = HRMLoopConfig{}
);

void gru_cell_forward(
    const float* x,
    const float* h_prev,
    float* h_next,
    const float* W,
    const float* U,
    const float* b,
    int64_t batch_size,
    int64_t hidden_dim
);

} // namespace kernels
} // namespace hrm
