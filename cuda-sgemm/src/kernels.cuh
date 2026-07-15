#pragma once

// SGEMM: C = alpha * A @ B + beta * C
//   A is (M x K), B is (K x N), C is (M x N), all row-major.
//
// Every kernel version from the blog post lives in its own .cu file and exposes
// ONE launch function with this exact signature. The wrapper hides each
// kernel's own <<<grid, block>>> config (and any template params) so the
// harness in runner.cu can call them all the same way.
//
// To add a new kernel: (1) create src/kernels/NN_name.cu with its run_kernel_N,
// (2) declare it below, (3) add one line to REGISTRY in runner.cu. That's it.

typedef void (*KernelFn)(int M, int N, int K, float alpha, const float *A,
                         const float *B, float beta, float *C);

// --- The blog post's narrative kernels (7 & 8 are skipped in the post too) ---
void run_kernel_1(int, int, int, float, const float *, const float *, float, float *);  // naive
void run_kernel_2(int, int, int, float, const float *, const float *, float, float *);  // gmem coalescing
void run_kernel_3(int, int, int, float, const float *, const float *, float, float *);  // shared-mem blocking
void run_kernel_4(int, int, int, float, const float *, const float *, float, float *);  // 1D blocktiling
void run_kernel_5(int, int, int, float, const float *, const float *, float, float *);  // 2D blocktiling
void run_kernel_6(int, int, int, float, const float *, const float *, float, float *);  // vectorized loads
void run_kernel_9(int, int, int, float, const float *, const float *, float, float *);  // autotuning
void run_kernel_10(int, int, int, float, const float *, const float *, float, float *); // warptiling
