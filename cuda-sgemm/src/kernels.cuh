#pragma once

// SGEMM: C = alpha * A @ B + beta * C
//   A is (M x K), B is (K x N), C is (M x N), all row-major.
// Each kernel version from the blog post gets its own launch function here.
// runner.cu dispatches to these by id.

void run_kernel_1(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C); // naive
void run_kernel_2(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C); // gmem coalescing
void run_kernel_3(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C); // shared-mem blocking
void run_kernel_4(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C); // 1D block tiling
void run_kernel_5(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C); // 2D block tiling
void run_kernel_6(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C); // vectorized loads

// Highest kernel id currently defined (bump as you add files).
#define NUM_KERNELS 6

// Human-readable name for a kernel id (0 == cuBLAS baseline).
const char *kernel_name(int id);
