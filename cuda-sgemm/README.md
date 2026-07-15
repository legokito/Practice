# CUDA SGEMM

Working through Simon Boehm's [CUDA matmul optimization post](https://siboehm.com/articles/22/CUDA-MMM): from a naive SGEMM kernel toward cuBLAS throughput, one kernel at a time.

![benchmark](assets/benchmark.png)

`size = 4096`, FP32, single T4. GFLOPs = throughput (higher is better).

| # | kernel | GFLOPs | % cuBLAS |
|---|--------|-------:|---------:|
| 0 | cuBLAS | 3969 | 100% |
| 1 | naive | 62 | 1.6% |
| 2 | gmem coalesce | — | — |
| 3 | shared-mem blocking | — | — |
| 4 | 1D blocktiling | — | — |
| 5 | 2D blocktiling | — | — |
| 6 | vectorized | — | — |
| 9 | autotuning | — | — |
| 10 | warptiling | — | — |

## Writeups

### Kernel 1 — Naive

### Kernel 2 — Global memory coalescing

<!-- Add one short section per kernel as you implement it. -->

## Run

Open `run_colab.ipynb` in Colab (T4 runtime), Run All → compiles, benchmarks, regenerates the chart.
