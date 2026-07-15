// SGEMM benchmark harness.
//
// Usage:  ./sgemm <kernel_id> <size> [repeats]
//   kernel_id : 0 = cuBLAS baseline, 1..NUM_KERNELS = your kernels
//   size      : square problem, M = N = K = size
//   repeats   : timed iterations (default 20)
//
// Prints ONE csv row to stdout:  kernel_id,name,size,gflops,max_rel_err,status
// Verification always runs against cuBLAS; status is OK or FAIL.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "kernels.cuh"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

const char *kernel_name(int id) {
  switch (id) {
  case 0: return "cuBLAS";
  case 1: return "01_naive";
  case 2: return "02_gmem_coalesce";
  case 3: return "03_shared_mem_block";
  case 4: return "04_1d_blocktiling";
  case 5: return "05_2d_blocktiling";
  case 6: return "06_vectorized";
  default: return "unknown";
  }
}

static cublasHandle_t g_handle;

// Row-major C = alpha*A@B + beta*C via cuBLAS (which is column-major).
// Trick: row-major (M,N) == column-major (N,M), so we compute the transpose.
static void run_cublas(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
              &beta, C, N);
}

static void dispatch(int id, int M, int N, int K, float alpha, const float *A,
                     const float *B, float beta, float *C) {
  switch (id) {
  case 0: run_cublas(M, N, K, alpha, A, B, beta, C); break;
  case 1: run_kernel_1(M, N, K, alpha, A, B, beta, C); break;
  case 2: run_kernel_2(M, N, K, alpha, A, B, beta, C); break;
  case 3: run_kernel_3(M, N, K, alpha, A, B, beta, C); break;
  case 4: run_kernel_4(M, N, K, alpha, A, B, beta, C); break;
  case 5: run_kernel_5(M, N, K, alpha, A, B, beta, C); break;
  case 6: run_kernel_6(M, N, K, alpha, A, B, beta, C); break;
  default:
    fprintf(stderr, "unknown kernel id %d\n", id);
    exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <kernel_id> <size> [repeats]\n", argv[0]);
    return 1;
  }
  const int id = atoi(argv[1]);
  const int size = atoi(argv[2]);
  const int repeats = argc > 3 ? atoi(argv[3]) : 20;
  const int M = size, N = size, K = size;
  const float alpha = 1.0f, beta = 0.0f;

  const size_t bytesA = (size_t)M * K * sizeof(float);
  const size_t bytesB = (size_t)K * N * sizeof(float);
  const size_t bytesC = (size_t)M * N * sizeof(float);

  float *hA = (float *)malloc(bytesA);
  float *hB = (float *)malloc(bytesB);
  for (size_t i = 0; i < (size_t)M * K; ++i) hA[i] = (float)rand() / RAND_MAX;
  for (size_t i = 0; i < (size_t)K * N; ++i) hB[i] = (float)rand() / RAND_MAX;

  float *dA, *dB, *dC, *dRef;
  CUDA_CHECK(cudaMalloc(&dA, bytesA));
  CUDA_CHECK(cudaMalloc(&dB, bytesB));
  CUDA_CHECK(cudaMalloc(&dC, bytesC));
  CUDA_CHECK(cudaMalloc(&dRef, bytesC));
  CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

  cublasCreate(&g_handle);

  // Reference result from cuBLAS.
  CUDA_CHECK(cudaMemset(dRef, 0, bytesC));
  run_cublas(M, N, K, alpha, dA, dB, beta, dRef);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Correctness: run the kernel once, compare to reference.
  CUDA_CHECK(cudaMemset(dC, 0, bytesC));
  dispatch(id, M, N, K, alpha, dA, dB, beta, dC);
  CUDA_CHECK(cudaDeviceSynchronize());

  float *hC = (float *)malloc(bytesC);
  float *hRef = (float *)malloc(bytesC);
  CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hRef, dRef, bytesC, cudaMemcpyDeviceToHost));
  float max_rel = 0.0f;
  for (size_t i = 0; i < (size_t)M * N; ++i) {
    float ref = fabsf(hRef[i]);
    float diff = fabsf(hC[i] - hRef[i]);
    float rel = diff / (ref > 1e-6f ? ref : 1e-6f);
    if (rel > max_rel) max_rel = rel;
  }
  const char *status = (max_rel < 1e-2f) ? "OK" : "FAIL";

  // Timing: warmup then `repeats` timed iterations.
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  dispatch(id, M, N, K, alpha, dA, dB, beta, dC); // warmup
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int r = 0; r < repeats; ++r)
    dispatch(id, M, N, K, alpha, dA, dB, beta, dC);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  double sec_per = (ms / 1000.0) / repeats;
  double gflops = (2.0 * M * N * K) / sec_per / 1e9;

  printf("%d,%s,%d,%.1f,%.2e,%s\n", id, kernel_name(id), size, gflops, max_rel,
         status);

  free(hA); free(hB); free(hC); free(hRef);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
  cublasDestroy(g_handle);
  return 0;
}
