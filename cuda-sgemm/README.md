# CUDA SGEMM

Working through Simon Boehm's [CUDA matmul optimization post](https://siboehm.com/articles/22/CUDA-MMM): from a naive SGEMM kernel toward cuBLAS throughput, one kernel at a time.

![benchmark](assets/benchmark.png)

`size = 4096`, FP32, single T4. GFLOPs = throughput (higher is better).

| # | kernel | GFLOPs | % cuBLAS |
|---|--------|-------:|---------:|
| 0 | cuBLAS | 4211 | 100% |
| 1 | naive | 62 | 1.5% |
| 2 | gmem coalesce | 474 | 11% |
| 3 | shared-mem blocking | — | — |
| 4 | 1D blocktiling | — | — |
| 5 | 2D blocktiling | — | — |
| 6 | vectorized | — | — |
| 9 | autotuning | — | — |
| 10 | warptiling | — | — |

## Writeups

### Preliminary Info
I'm using T4 GPU (google Colab). Specs: 8100 GLOPs, 320 GB/s memory bandwidth. 

### Kernel 1 — Naive
Our inner loop is fetching 2 numbers (8 bytes), multiplying them and eventually adding it to tmp (2 math ops). This means I'm getting 0.25 math ops per byte.

Even with max bandwidth, this would mean I get 320 * 0.25 = 80 GFLOPs at best (ceiling). Since I'm getting ~62 GFLOPs on this kernel (when max possible is 8100), I can confidently say that a major part to optimize is either how much math I'm doing per byte of retrieval (justifiable by these numbers) and/or how fast I'm retreiving the bytes themselves (not in terms of literal bandwidth speed [cause at 320 max GFLOPs are still low], but in terms of the pattern in which I'm retreiving them which could be slowing down GFLOPs).

But the bigger realization is that matmul doesn't *have* to be memory-bound at all. The least memory I could possibly move is reading A, B, C once + writing C back — ~270MB for 4096², which this card moves in ~0.85ms (270MB / 320 GB/s). The actual math is ~137 billion ops, which takes ~17ms at 8100 GFLOPs. So the math is ~20x slower than the minimum memory movement, meaning if I retrieved memory efficiently this should actually be compute-bound. The naive kernel breaks that in the two exact ways above: no reuse means each thread re-reads its whole row/col from scratch (so I move way more than that 270MB minimum), and the scattered pattern means I only use a fraction of the 320 GB/s. So the memory-bound-ness here is self-inflicted — both levers (reuse + pattern) are what the later kernels fix.

### Kernel 2 — Global memory coalescing
By making each warp's 32 threads read contiguous memory, one fetch now feeds many threads at once instead of mostly wasting bytes — so each warp does far more useful work per byte fetched. More useful computations per byte retrieved in a given interval bumped GFLOPs from ~62 to ~474 (~8x which logically makes sense, ~11% of cuBLAS).

## Run

Open `run_colab.ipynb` in Colab (T4 runtime), Run All → compiles, benchmarks, regenerates the chart.
