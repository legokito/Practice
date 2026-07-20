# CUDA SGEMM

Working through Simon Boehm's [CUDA matmul optimization post](https://siboehm.com/articles/22/CUDA-MMM): from a naive SGEMM kernel toward cuBLAS throughput, one kernel at a time.

![benchmark](assets/benchmark.png)

`size = 4096`, FP32, single T4. GFLOPs = throughput (higher is better).

| # | kernel | GFLOPs | % cuBLAS |
|---|--------|-------:|---------:|
| 0 | cuBLAS | 4211 | 100% |
| 1 | naive | 62 | 1.5% |
| 2 | gmem coalesce | 474 | 11% |
| 3 | shared-mem blocking | 860 | 21% |
| 4 | 1D blocktiling | 1749 | 42% |
| 5 | 2D blocktiling | 2412 | 55% |
| 6 | vectorized | — | — |
| 9 | autotuning | — | — |
| 10 | warptiling | — | — |

## Writeups

### Preliminary Info
I'm using T4 GPU (google Colab). Specs: 8100 GLOPs, 320 GB/s memory bandwidth. 

I will fill out explanations for review + proper documentation after having implemented all the kernels.

### Kernel 1 — Naive

### Kernel 2 — Global memory coalescing

### Kernel 3 — Shared memory blocking (tiling)

### Kernel 4 — 1D blocktiling

## Run

Open `run_colab.ipynb` in Colab (T4 runtime), Run All → compiles, benchmarks, regenerates the chart.
