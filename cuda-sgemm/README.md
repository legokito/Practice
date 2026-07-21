# CUDA SGEMM

Working through Simon Boehm's [CUDA matmul optimization post](https://siboehm.com/articles/22/CUDA-MMM): from a naive SGEMM kernel toward cuBLAS throughput, one kernel at a time.  
Goals: understand gpu architecutre, roofline analysis, common patterns behind kernel (and generally computation) speedups.

![benchmark](assets/benchmark.png)

`size = 4096`, FP32, single T4. GFLOPs = throughput (higher is better).

| # | kernel | GFLOPs | % cuBLAS |
|---|--------|-------:|---------:|
| 0 | cuBLAS | 4109 | 100% |
| 1 | naive | 62 | 2% |
| 2 | gmem coalesce | 497 | 12% |
| 3 | shared-mem blocking | 868 | 21% |
| 4 | 1D blocktiling | 1764 | 43% |
| 5 | 2D blocktiling | 2506 | 61% |
| 6 | vectorized | 3420 | 83% |

## Writeups

### Preliminary Info
I'm using T4 GPU (google Colab). Specs: 8100 GLOPs, 320 GB/s memory bandwidth. 

I will fill out explanations for review + proper documentation after having implemented all the kernels.

### Kernel 1 — Naive

### Kernel 2 — Global memory coalescing

### Kernel 3 — Shared memory blocking (tiling)

### Kernel 4 — 1D blocktiling

### Kernel 5 - 2D blocktiling

### Kernel 6 - Vectorized memory accessing

## Run

Open `run_colab.ipynb` in Colab (T4 runtime), Run All → compiles, benchmarks, regenerates the chart.
