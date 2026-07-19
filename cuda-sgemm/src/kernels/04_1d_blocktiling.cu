#include <cuda_runtime.h>

// Kernel 4: 1D blocktiling.

// aiming for larger computation within a block - output tile grew, K size shrank, and total thread per block dropped (since you're computing more per thread). 

// desc:
//   Output tile per block: BM x BN = 64 x 64
//   K-strip width:         BK = 8
//   Results per thread:    TM = 8   (a vertical run of 8 cells)
//   Threads per block:     (BM*BN)/TM = 64*64/8 = 512
#define BM 64
#define BN 64
#define BK 8
#define TM 8

__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha,
                                     const float *A, const float *B, float beta,
                                     float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // this thread's spot in the 64x64 output tile.
  const uint threadCol = threadIdx.x % BN; 
  const uint threadRow = threadIdx.x / BN; 

  __shared__ float As[BM * BK]; 
  __shared__ float Bs[BK * BN]; 

  // move the A/B/C base pointers to this block's tile (then we index locally)
  A += cRow * BM * K;            
  B += cCol * BN;                
  C += cRow * BM * N + cCol * BN;

  const uint innerRowA = threadIdx.x / BK; // 0..63
  const uint innerColA = threadIdx.x % BK; // 0..7
  const uint innerRowB = threadIdx.x / BN; // 0..7
  const uint innerColB = threadIdx.x % BN; // 0..63

  float threadResults[TM] = {0.0f};

  // march along K in BK-wide strips
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

    // load into memory
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance the pointers to the next K-strip
    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; dotIdx++){
	// moving through entirety of B, down the column 
	float bValue = Bs[dotIdx * BN + threadCol]; 

	// moving through TM steps of A, down the column. 
	for (uint aIdx = 0; aIdx < TM; aIdx++){
	    threadResults[aIdx] += As[(threadRow * TM + aIdx) * BK + dotIdx] * bValue;
	}
    }	

    __syncthreads();
  }

  for (uint placement = 0; placement < TM; placement++){
    C[(threadRow * TM + placement) * N + threadCol] = alpha * threadResults[placement] + beta * C[(threadRow * TM + placement) * N + threadCol];
  }
} 

void run_kernel_4(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block((BM * BN) / TM); // 512 threads, 1D
  sgemm_1d_blocktiling<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
