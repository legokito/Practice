#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void sgemm_coalesce(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.y; // row of C, in [0, M)
  const uint y = blockIdx.y * blockDim.y + threadIdx.x; // col of C, in [0, N)

  if (x < M && y < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

void run_kernel_2(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
  dim3 block(32, 32);
  dim3 grid((M + 31) / 32, (N + 31) / 32);
  sgemm_coalesce<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}


