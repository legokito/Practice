#include <cuda_runtime.h>

// sizes; B is input, T is output
#define BM 128 
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha,
                                     const float *A, const float *B, float beta,
                                     float *C) {
  // block identifier
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // thread identifier
  const uint threadCol = threadIdx.x % (BN/TN);
  const uint threadRow = threadIdx.x / (BN/TN);

  // memory holders
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // set pointers to start at the section of each matrix we're interested in.
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN; 

  const uint numThreads = (BM * BN) / (TM * TN);  // 256
  const uint strideA    = numThreads / BK;        // 32
  const uint strideB    = numThreads / BN;        // 2

  const uint innerRowA = threadIdx.x / BK; // 0..31
  const uint innerColA = threadIdx.x % BK; // 0..7
  const uint innerRowB = threadIdx.x / BN; // 0..1
  const uint innerColB = threadIdx.x % BN; // 0..128

  float threadResults[TM][TN] = {0.0f};

  float cacheA[8];
  float cacheB[8]; 

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK){
    for (uint i = innerRowA; i < BM; i += strideA){
      As[i * BK + innerColA] = A[i * K + innerColA]; 
    }
    for (uint i = innerRowB; i < BK; i += strideB){
      Bs[i * BN + innerColB] = B[i * N + innerColB]; 
    }

    __syncthreads();

    A += BK;
    B += BK * N;

   
    for (uint dotIdx = 0; dotIdx < BK; dotIdx++){ // for all 8 outer products
      for (uint cacher = 0; cacher < TM; cacher++){ //ez since TM = TN
    	cacheA[cacher] = As[(threadRow * TM + cacher) * BK + dotIdx];
        cacheB[cacher] = Bs[dotIdx * BN + (threadCol * TN) + cacher];
      }

      for (uint resM = 0; resM < TM; resM++){
	for (uint resN = 0; resN < TN; resN++){
	  threadResults[resM][resN] += cacheA[resM] * cacheB[resN]; 
	}	
      }
    }
    __syncthreads();
  }	 

  for (uint cM = 0; cM < TM; cM++){
    for (uint cN = 0; cN < TN; cN++){
      C[(threadRow * TM + cM) * N + (threadCol * TN + cN)] = 
         alpha * threadResults[cM][cN] + 
         beta * C[(threadRow * TM + cM) * N + (threadCol * TN + cN)];
    }
  }
}

void run_kernel_5(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block((BM * BN) / (TM * TN)); // 256 threads
  sgemm_2d_blocktiling<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
