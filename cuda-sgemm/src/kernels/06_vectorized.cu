#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

/*
logic of 5 copied 

edited to vectorize memory loading and updated inner indexing accordingly
*/
__global__ void sgemm_vectorized(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint threadCol = threadIdx.x % (BN/TN);
  const uint threadRow = threadIdx.x / (BN/TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN; 

  const uint numThreads = (BM * BN) / (TM * TN);  // 256

  const uint innerRowA = threadIdx.x / (BK/4); 
  const uint innerColA = threadIdx.x % (BK/4); 
  const uint innerRowB = threadIdx.x / (BN/4);  
  const uint innerColB = threadIdx.x % (BN/4); 

  float threadResults[TM][TN] = {0.0f};

  float cacheA[8];
  float cacheB[8]; 

  float4 tmp;
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK){
    // vectorize and transpose A
    tmp = reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0]; 

    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    //vectorize B
    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
	  reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4])[0];

    __syncthreads();

    A += BK;
    B += BK * N;

   
    for (uint dotIdx = 0; dotIdx < BK; dotIdx++){ 
      for (uint cacher = 0; cacher < TM; cacher++){ 
      	cacheA[cacher] = As[dotIdx * BM + (threadRow * TM) + cacher];
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

void run_kernel_6(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block((BM * BN) / (TM * TN)); // 256 threads
  sgemm_vectorized<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
