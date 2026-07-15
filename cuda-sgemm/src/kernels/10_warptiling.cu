#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Kernel 10: warptiling  --  TODO: implement while working through the post.
// Until then this is a stub so the project still compiles and links.
void run_kernel_10(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
  (void)M; (void)N; (void)K; (void)alpha; (void)A; (void)B; (void)beta; (void)C;
  fprintf(stderr, "kernel 10 (warptiling) not implemented yet\n");
  exit(3);
}
