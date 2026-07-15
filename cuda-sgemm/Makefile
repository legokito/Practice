# Build the SGEMM benchmark. Requires an NVIDIA GPU + CUDA toolkit (nvcc).
# On a Mac (no NVIDIA GPU) you can't build here -- use run_colab.ipynb instead.
#
#   make            # build ./sgemm
#   make ARCH=sm_75 # target a specific GPU (T4=sm_75, A100=sm_80, etc.)
#   make clean

ARCH ?= sm_75
NVCC ?= nvcc
NVCCFLAGS := -O3 -arch=$(ARCH) -std=c++14 -Isrc
LDFLAGS := -lcublas

SRC := src/runner.cu $(wildcard src/kernels/*.cu)

sgemm: $(SRC) src/kernels.cuh
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f sgemm

.PHONY: clean
