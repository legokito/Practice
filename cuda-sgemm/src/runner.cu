// SGEMM benchmark harness.
//
// Usage:  ./sgemm <kernel_id> <size> [repeats]
//         ./sgemm list                 # print all registered kernel ids
//   kernel_id : 0 = cuBLAS baseline, others = your kernels (see REGISTRY below)
//   size      : square problem, M = N = K = size
//   repeats   : timed iterations (default 20)
//
// Prints ONE csv row to stdout:  kernel_id,name,size,gflops,max_rel_err,status
// Verification always runs against cuBLAS; status is OK or FAIL.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// ---------------------------------------------------------------------------
// The ONE place to register a kernel. fn == nullptr means "cuBLAS baseline"
// (handled specially below). Add a line here after writing a new kernel file.
// ---------------------------------------------------------------------------
struct KernelEntry {
  int id;
  const char *name;
  KernelFn fn;
};
static const KernelEntry REGISTRY[] = {
    {0, "cuBLAS", nullptr},
    {1, "01_naive", run_kernel_1},
    {2, "02_gmem_coalesce", run_kernel_2},
    {3, "03_shared_mem_block", run_kernel_3},
    {4, "04_1d_blocktiling", run_kernel_4},
    {5, "05_2d_blocktiling", run_kernel_5},
    {6, "06_vectorized", run_kernel_6},
    {9, "09_autotuning", run_kernel_9},
    {10, "10_warptiling", run_kernel_10},
};
static const int NUM_ENTRIES = sizeof(REGISTRY) / sizeof(REGISTRY[0]);

static const KernelEntry *find_kernel(int id) {
  for (int i = 0; i < NUM_ENTRIES; ++i)
    if (REGISTRY[i].id == id) return &REGISTRY[i];
  return nullptr;
}

static cublasHandle_t g_handle;

// Row-major C = alpha*A@B + beta*C via cuBLAS (which is column-major).
// Trick: a row-major (M,N) matrix IS a column-major (N,M) matrix in memory,
// so we compute the transpose and it lands correct in row-major.
static void run_cublas(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
              &beta, C, N);
}

static void dispatch(const KernelEntry *k, int M, int N, int K, float alpha,
                     const float *A, const float *B, float beta, float *C) {
  if (k->fn == nullptr) run_cublas(M, N, K, alpha, A, B, beta, C);
  else k->fn(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "list") == 0) {
    for (int i = 0; i < NUM_ENTRIES; ++i) printf("%d\n", REGISTRY[i].id);
    return 0;
  }
  if (argc < 3) {
    fprintf(stderr, "usage: %s <kernel_id> <size> [repeats]  |  %s list\n",
            argv[0], argv[0]);
    return 1;
  }
  const int id = atoi(argv[1]);
  const int size = atoi(argv[2]);
  const int repeats = argc > 3 ? atoi(argv[3]) : 20;
  const KernelEntry *k = find_kernel(id);
  if (!k) {
    fprintf(stderr, "unknown kernel id %d\n", id);
    return 1;
  }
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
  dispatch(k, M, N, K, alpha, dA, dB, beta, dC);
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
  dispatch(k, M, N, K, alpha, dA, dB, beta, dC); // warmup
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int r = 0; r < repeats; ++r)
    dispatch(k, M, N, K, alpha, dA, dB, beta, dC);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  double sec_per = (ms / 1000.0) / repeats;
  double gflops = (2.0 * M * N * K) / sec_per / 1e9;

  printf("%d,%s,%d,%.1f,%.2e,%s\n", id, k->name, size, gflops, max_rel, status);

  free(hA); free(hB); free(hC); free(hRef);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
  cublasDestroy(g_handle);
  return 0;
}
