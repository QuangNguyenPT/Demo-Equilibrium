#include "hrm/kernels/gemm.hpp"
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace hrm {
namespace kernels {

// Blocked GEMM for L1/L2 cache locality
void gemm_f32(
    const float* A,
    const float* B,
    float* C,
    int64_t M,
    int64_t N,
    int64_t K,
    float alpha,
    float beta
) {
    if (beta == 0.0f) {
        std::memset(C, 0, M * N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int64_t i = 0; i < M * N; ++i) {
            C[i] *= beta;
        }
    }

    constexpr int64_t BLOCK_M = 64;
    constexpr int64_t BLOCK_N = 64;
    constexpr int64_t BLOCK_K = 64;

    for (int64_t bm = 0; bm < M; bm += BLOCK_M) {
        int64_t m_end = std::min(bm + BLOCK_M, M);
        for (int64_t bk = 0; bk < K; bk += BLOCK_K) {
            int64_t k_end = std::min(bk + BLOCK_K, K);
            for (int64_t bn = 0; bn < N; bn += BLOCK_N) {
                int64_t n_end = std::min(bn + BLOCK_N, N);

                for (int64_t i = bm; i < m_end; ++i) {
                    const float* a_row = A + i * K;
                    float* c_row = C + i * N;

                    for (int64_t k = bk; k < k_end; ++k) {
                        float a_val = alpha * a_row[k];
                        const float* b_row = B + k * N;

                        #pragma omp simd
                        for (int64_t j = bn; j < n_end; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    const auto& sA = A.shape();
    const auto& sB = B.shape();

    if (sA.size() < 2 || sB.size() < 2) {
        throw std::invalid_argument("[GEMM] Inputs must have at least 2 dimensions.");
    }

    int64_t M = sA[sA.size() - 2];
    int64_t K_A = sA.back();
    int64_t K_B = sB[sB.size() - 2];
    int64_t N = sB.back();

    if (K_A != K_B) {
        throw std::invalid_argument("[GEMM] Dimension mismatch: K_A (" + std::to_string(K_A) +
                                    ") != K_B (" + std::to_string(K_B) + ")");
    }

    gemm_f32(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K_A);
}

void linear(
    const Tensor& X,
    const Tensor& W,
    const Tensor* bias,
    Tensor& Y
) {
    // X: (B, K), W: (N, K) -> Compute Y = X * W^T
    const auto& sX = X.shape();
    const auto& sW = W.shape();

    int64_t B = 1;
    for (size_t i = 0; i < sX.size() - 1; ++i) {
        B *= sX[i];
    }
    int64_t K = sX.back();
    int64_t N = sW[0];
    int64_t W_K = sW[1];

    if (K != W_K) {
        throw std::invalid_argument("[LINEAR] Input dim " + std::to_string(K) +
                                    " != Weight dim " + std::to_string(W_K));
    }

    const float* x_ptr = X.data<float>();
    const float* w_ptr = W.data<float>();
    float* y_ptr = Y.data<float>();

    // W is (N, K). Row n of W is row-major.
    // Y[b, n] = sum_k (X[b, k] * W[n, k])
    for (int64_t b = 0; b < B; ++b) {
        const float* x_row = x_ptr + b * K;
        float* y_row = y_ptr + b * N;

        for (int64_t n = 0; n < N; ++n) {
            const float* w_row = w_ptr + n * K;
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int64_t k = 0; k < K; ++k) {
                sum += x_row[k] * w_row[k];
            }
            if (bias && bias->raw_data()) {
                sum += bias->data<float>()[n];
            }
            y_row[n] = sum;
        }
    }
}

} // namespace kernels
} // namespace hrm
