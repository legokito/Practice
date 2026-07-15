#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Kernel 6: vectorized  --  TODO: implement while working through the post.
// Until then this is a stub so the project still compiles and links.
void run_kernel_6(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
  (void)M; (void)N; (void)K; (void)alpha; (void)A; (void)B; (void)beta; (void)C;
  fprintf(stderr, "kernel 6 (vectorized) not implemented yet\n");
  exit(3);
}
