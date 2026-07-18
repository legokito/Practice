#include <cuda_runtime.h>

#define BLOCKSIZE 32

// Kernel 3: shared-memory cache-blocking (tiling).
// The block stages a 32x32 tile of A and of B in shared memory, marches along
// K, and reuses each staged value ~32x instead of re-reading from global.
__global__ void sgemm_shared(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
  // which 32x32 output tile of C this block owns
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // this thread's slot within the tile
  const uint threadRow = threadIdx.y; // 0..31
  const uint threadCol = threadIdx.x; // 0..31

  // shared-memory scratchpads (one tile of A, one of B)
  __shared__ float As[BLOCKSIZE][BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

  // this thread's global output coordinates
  const uint globalRow = cRow * BLOCKSIZE + threadRow;
  const uint globalCol = cCol * BLOCKSIZE + threadCol;

  float tmp = 0.0f; // accumulates the FULL dot product across all K-tiles

// iterate over width of A and height of B in terms of blocks. each iteration computes over a block within A and B.  
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // each thread loads one A element and one B element into shared memory. loads the corresponding thread within a block.
       As[threadRow][threadCol] = A[globalRow * K + (bkIdx + threadCol)];
       Bs[threadRow][threadCol] = B[(bkIdx + threadRow) * N + globalCol];

    __syncthreads(); // let all necessary values load in SMEM.

    // partial dot product which we'll store and incrementally add up (to eventually store in C)
       for (int k = 0; k < BLOCKSIZE; ++k)
           tmp += As[threadRow][k] * Bs[k][threadCol];

    __syncthreads(); // let all computations needing this memory finish
  }

     C[globalRow * N + globalCol] = alpha * tmp + beta * C[globalRow * N + globalCol];
}

void run_kernel_3(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
  dim3 block(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);
  sgemm_shared<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
