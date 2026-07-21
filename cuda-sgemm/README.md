# CUDA SGEMM

Working through Simon Boehm's [CUDA matmul optimization post](https://siboehm.com/articles/22/CUDA-MMM): from a naive SGEMM kernel toward cuBLAS throughput, one kernel at a time.  
Goals: understand gpu architecutre, roofline analysis, common patterns behind kernel (and generally computation) speedups.

![benchmark](assets/benchmark.png)

`size = 4096`, FP32, single T4. GFLOPs = throughput (higher is better).

| # | kernel | GFLOPs | % cuBLAS |
|---|--------|-------:|---------:|
| 0 | cuBLAS (nvidia) | 4109 | 100% |
| 1 | naive | 62 | 2% |
| 2 | gmem coalesce | 497 | 12% |
| 3 | shared-mem blocking | 868 | 21% |
| 4 | 1D blocktiling | 1764 | 43% |
| 5 | 2D blocktiling | 2506 | 61% |
| 6 | vectorized | 3420 | 83% |

## Writeups

### Setup (roofline reference)
GPU used: T4 (free through Colab). Calculated with FP32 of precision, and size: M = N = K = 4096.
- FP32 peak: ~8100 GFLOPs
- Bandwidth: ~320 GB/s
- Ridge point: 8100 / 320 ≈ 25 FLOP/byte (below this means kernel is memory-bound, above means compute-bound)
- Roofline: attainable = min(peak, arithmetic intensity × bandwidth)
- cuBLAS ≈ 4100 GFLOPs is the practical ceiling (~50% of peak); "% cuBLAS" is measured against it.

### High level overview
Each kernel kills one bottleneck, and every fix is one of two kinds:
- do more math per byte of memory read (reuse)
- read memory more efficiently (waste fewer bytes). 

Each writeup: what was slow, what did I change, why the number moved.

| # | kernel | lever | metric that moved | GFLOPs |
|---|--------|-------|-------------------|-------:|
| 1 | naive | — (baseline) | — | 62 |
| 2 | coalesce | read efficiently | scattered reads to contiguous reads | 497 |
| 3 | shared blocking | reuse | global reads decrease by ~32× (stage tile once) | 868 |
| 4 | 1D tiling | reuse | math per shared read is increased | 1764 |
| 5 | 2D tiling | reuse | math per shared read is increased again! | 2506 |
| 6 | vectorized | read efficiently | load instructions in larger chunks using float4 | 3420 |

### Kernel 1 — Naive
Implements vanilla matmul - each thread goes pulls a row from A and a col from B and performs A @ B for their respective output cell in C. 

This has extremely slow performance due to 0 memory reuse, poor thread indexing. Overall it makes no active effort to use hardware/software exploitations to help compute the result faster. 

We use this as a baseline to iteratively work toward increasing performance!

### Kernel 2 — Global memory coalescing
Kernel 2 does identical math and reads identical bytes as kernel 1 - why is it still faster???

Answer: arranging reads to where more values are used per "memory serving". 

This is due to warps. Warps are structures that enable a fixed number of threads to perform same operations in lockstep with one another. Whenever a warp (for nvidia gpus: holds 32 threads) wants to access memory, it is served in 128-byte chunks (i.e. 32 floats for FP32). 

This means, everytime a warp accesses a certain chunk of memory, it can simultaneously read all the values within that piece of memory and perform actions on it at the exact same time. 

Threads are grouped into warps of 32 by consecutive thread index (threadIdx.x is the fast(est/er) one). So mapping threadIdx.x to col (instead of row in k1) means it will generate threads that require floats that are contiguous in memory, therefore letting a warp group them together and allowing them to exploit the 'parallel-processing' ability that warps possess!

In practice, this regrouping allows each warp to make use of all 32 floats of a memory serving (rather than just 1 in k1), causing a significant increase in GFLOPs! 

### Kernel 3 — Shared memory blocking (tiling)
K3 chooses to exploit a memory layer we hadn't enabled before: SMEM (shared memory)!

Each block has ability to load in a certain chunk of memory and read it at extremely fast rates. You pay the cost of reading the values from GMEM and loading it into SMEM again, but if you have enough numbers in SMEM that can be reused, then the cost is negligible and the upgrade is definitely worth having (exactly this case)!

The algorithm remains the same in principle, except instead of being able read the entire row or col at once, we stage a chunk of it (specifically 32x32), calculate from a row and col belonging within that chunk, and then load in a new chunk and compute over it until we've gone across the same row and col and computed the full A @ B.

This simple reusing strategy gave us a nearly 2x speedup from k2, showing just how powerful and fast SMEM is!

Notice that each thread does 2 SMEM reads for every 1 multiply. We're spending more time fetching from SMEM than computing with it. Surely we can squeeze more math out of each read..... (I wonder why Manit's flagging this here hmmmmmmm 👀).

### Kernel 4 — 1D blocktiling
In K4, we aim to do more math out of each SMEM read to spend less time in data transfer.

K3 does 2 SMEM reads for every 1 multiply (1 from A, 1 from B), so we're bottlenecked on reading, not computing.

The fix? Read a B value once, cache it, and reuse it for every A value that needs it. Instead of each thread owning a single output cell, it now owns a whole column of 8 (TM=8) — so one cached B value feeds 8 multiplies instead of 1. While there is a tiny sequential penalty we pay with this, it is negligible compared to the speedup since the bandwidth thresholds for reading values from memory are far lower than the time it takes to make an additional computation. 

Counting reads for those 8 results: K3 needs 2 each = 16 reads. K4 needs 8 reads from A + 1 cached B read = 9 reads. Nearly half the SMEM traffic for the exact same compute! When it comes to the speedup, the GFLOPs speak for themselves :))

### Kernel 5 - 2D blocktiling
Exact same philosophy as K4, except now each thread computes an entire 8x8 tile using the same caching strategy across both dimensions rather than just one. 

Like K4, the penalty for computing more per thread is marginal compared to the memory bandwidth bottlenecks - making this speedup extremely worth it from an intuition standpoint as well. The numbers again show an increase, showing just how effective caching is!

Notice the speedup (~1.4x, 1764 to 2506) is much smaller than the drop in reads (~4.5x). That gap is useful info: SMEM reads are becoming less of the bottleneck — we're starting to bump into other limits, so cutting reads buys less than it used to, and we're forced to look for other avenues to trigger speedups.

### Kernel 6 - Vectorized memory accessing
This one lets us load values into SMEM in bunches of 4 (16 byte loads) using float4. This cuts the number of load instructions by ~4x since larger chunks are getting transported at once - leading to a simple but effective speedup!

For matrix B, this is as straightforward as it sounds. Matrix A is trickier - loading it from global is already sequential, but the catch comes later: each thread reads a column of A's tile into registers to compute, and a column sits scattered in memory (float4 can only grab 4 sequential floats, not scattered ones). So we store A transposed, which turns that column into a sequentially accessible segment and lets those compute-side reads vectorize as well!

## Run

Open `run_colab.ipynb` in Colab (T4 runtime), Run All → compiles, benchmarks, regenerates the chart.

