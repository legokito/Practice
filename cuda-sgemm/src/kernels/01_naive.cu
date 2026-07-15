#include <cuda_runtime.h>

// Kernel 1: naive.
// One thread per output element C[x][y]. Each thread walks the full K
// dimension, reading a row of A and a column of B straight from global memory.
// Correct, but memory-bound and uncoalesced -> this is the baseline we beat.
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x; // row of C, in [0, M)
  const uint y = blockIdx.y * blockDim.y + threadIdx.y; // col of C, in [0, N)

  if (x < M && y < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

void run_kernel_1(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
  dim3 block(32, 32);
  dim3 grid((M + 31) / 32, (N + 31) / 32);
  sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
