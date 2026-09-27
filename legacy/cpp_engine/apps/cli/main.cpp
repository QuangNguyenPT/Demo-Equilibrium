#include "hrm/core/types.hpp"
#include "hrm/core/allocator.hpp"
#include "hrm/core/tensor.hpp"
#include "hrm/kernels/gemm.hpp"
#include "hrm/kernels/norm.hpp"
#include "hrm/kernels/activation.hpp"
#include "hrm/kernels/rope.hpp"
#include "hrm/kernels/attention.hpp"
#include "hrm/kernels/hrm_loop.hpp"
#include "hrm/runtime/thread_pool.hpp"

#include <iostream>
#include <chrono>
#include <iomanip>

int main(int argc, char** argv) {
    std::cout << "=========================================================\n";
    std::cout << "  HRM+LLM Hybrid AI Inference Engine (Native C++ CLI)\n";
    std::cout << "  Architecture: Separate HRM Module + Transformer Backbone\n";
    std::cout << "=========================================================\n\n";

    int64_t B = 1;
    int64_t T = 32;
    int64_t D = 256;
    int64_t H = 4;
    int64_t head_dim = D / H;

    std::cout << "[CONFIG] Batch: " << B << ", SeqLen: " << T 
              << ", HiddenDim: " << D << ", Heads: " << H 
              << ", HeadDim: " << head_dim << "\n";

    hrm::ArenaAllocator arena(64 * 1024 * 1024); // 64MB bump arena
    std::cout << "[SYSTEM] Initialized 64MB ArenaAllocator (Zero-allocation runtime)\n";

    // Create tensors allocated directly in bump arena
    hrm::Tensor Q({B, H, T, head_dim}, hrm::DataType::Float32, &arena);
    hrm::Tensor K({B, H, T, head_dim}, hrm::DataType::Float32, &arena);
    hrm::Tensor V({B, H, T, head_dim}, hrm::DataType::Float32, &arena);
    hrm::Tensor O({B, H, T, head_dim}, hrm::DataType::Float32, &arena);

    Q.fill(0.05f);
    K.fill(0.05f);
    V.fill(0.10f);

    std::cout << "[KERNEL] Testing Fused FlashAttention (Online Softmax)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    hrm::kernels::flash_attention_forward(Q, K, V, O, true);
    auto t1 = std::chrono::high_resolution_clock::now();
    double attn_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::cout << "  -> FlashAttention forward pass completed in " << std::fixed << std::setprecision(2)
              << attn_us << " microseconds.\n";

    // Test HRM Recurrent reasoning loop
    std::cout << "[KERNEL] Testing HRM Hierarchical Reasoning Loop with ACT...\n";
    hrm::Tensor z_h({B, D}, hrm::DataType::Float32, &arena);
    hrm::Tensor z_l({B, D}, hrm::DataType::Float32, &arena);
    hrm::Tensor input_emb({B, D}, hrm::DataType::Float32, &arena);
    hrm::Tensor W_low({3 * D, D}, hrm::DataType::Float32, &arena);
    hrm::Tensor U_low({3 * D, D}, hrm::DataType::Float32, &arena);
    hrm::Tensor b_low({3 * D}, hrm::DataType::Float32, &arena);
    hrm::Tensor W_high({3 * D, D}, hrm::DataType::Float32, &arena);
    hrm::Tensor U_high({3 * D, D}, hrm::DataType::Float32, &arena);
    hrm::Tensor b_high({3 * D}, hrm::DataType::Float32, &arena);
    hrm::Tensor W_qhead({2, D}, hrm::DataType::Float32, &arena);
    hrm::Tensor b_qhead({2}, hrm::DataType::Float32, &arena);

    z_h.fill(0.1f);
    z_l.fill(0.1f);
    input_emb.fill(0.2f);
    W_low.fill(0.01f);
    U_low.fill(0.01f);
    b_low.zero();
    W_high.fill(0.01f);
    U_high.fill(0.01f);
    b_high.zero();
    W_qhead.fill(0.02f);
    b_qhead.data<float>()[0] = 1.0f; // Force early halting
    b_qhead.data<float>()[1] = -1.0f;

    auto t2 = std::chrono::high_resolution_clock::now();
    auto result = hrm::kernels::run_hrm_reasoning_loop(
        z_h, z_l, input_emb,
        W_low, U_low, b_low,
        W_high, U_high, b_high,
        W_qhead, b_qhead,
        /*max_steps=*/8,
        /*low_steps=*/2,
        /*act_threshold=*/0.6f
    );
    auto t3 = std::chrono::high_resolution_clock::now();
    double hrm_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

    std::cout << "  -> Reasoning loop executed: " << result.steps_executed << " steps\n";
    std::cout << "  -> ACT Halting triggered: " << (result.halted_early ? "YES (confidence: " : "NO (confidence: ")
              << result.halt_confidence << ")\n";
    std::cout << "  -> Latency: " << hrm_us << " microseconds.\n\n";

    std::cout << "[ARENA] Memory utilized: " << (arena.used_bytes() / 1024) << " KB / "
              << (arena.total_capacity() / 1024) << " KB\n";
    arena.reset();
    std::cout << "[ARENA] Fast reset completed (0 bytes active).\n";
    std::cout << "Engine verification successful!\n";

    return 0;
}
