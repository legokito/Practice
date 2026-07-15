# Practice

A learning log. Each folder is me working through a blog post, paper, or concept from scratch — implementing it, benchmarking it, and writing up what I understood. Mostly AI systems: GPUs, kernels, inference, training infra.

The point isn't polished libraries; it's *doing the work* and showing it — real code, real measurements, honest writeups.

## Projects

| Project | Topic | Source | Status |
|---------|-------|--------|:------:|
| [`cuda-sgemm`](cuda-sgemm/) | Optimizing a CUDA matmul kernel toward cuBLAS, step by step | [siboehm.com](https://siboehm.com/articles/22/CUDA-MMM) | 🚧 in progress |

<!-- Add a row per new project. Status: 🚧 in progress · ✅ done · 💤 paused -->

## How this is organized

- One folder per topic, self-contained (own README, code, results).
- Where there's a GPU/perf angle, results are benchmarked and charted, and the raw numbers are committed as CSV so the charts are reproducible.
- I'm on Apple Silicon, so GPU code is built and run on a free Colab T4 via a `run_colab.ipynb` in each project.
