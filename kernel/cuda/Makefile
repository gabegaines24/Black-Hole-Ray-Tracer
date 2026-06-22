# CUDA batch kernel — standalone nvcc build
#
# Prerequisites: CUDA toolkit with nvcc on PATH.
#
# Usage:
#   cd kernel/cuda
#   make cuda_batch              # build with default architecture
#   make cuda_batch ARCH=sm_89   # Ada Lovelace (RTX 4090)
#   make clean
#
# Output: ../../build/libbh_rt_phase2_cuda.so
#   Loaded at runtime by src/blackhole_ray_tracer/native_phase2_cuda.py
#   via the BLACKHOLE_CUDA_LIB env var or automatic build/ search.

ARCH     ?= sm_75   # GTX 1650 Super / Turing — safe default
NVCC     ?= nvcc
CFLAGS   := -O2 -arch=$(ARCH) -I../include --compiler-options -fPIC
OUTDIR   := ../../build
LIB_NAME := libbh_rt_phase2_cuda.so
SRC      := bh_rt_schwarzschild_phase2_batch.cu

.PHONY: all cuda_batch clean

all: cuda_batch

cuda_batch: $(OUTDIR)/$(LIB_NAME)

$(OUTDIR)/$(LIB_NAME): $(SRC)
	@mkdir -p $(OUTDIR)
	$(NVCC) $(CFLAGS) -shared -o $@ $< -lm
	@echo "Built $@ (arch=$(ARCH))"

clean:
	rm -f $(OUTDIR)/$(LIB_NAME)

# ── Windows (nmake) alternative ───────────────────────────────────────────────
# On Windows use:
#   nvcc -O2 -arch=sm_75 -I..\include --compiler-options /LD ^
#        -o ..\..\build\bh_rt_phase2_cuda.dll ^
#        bh_rt_schwarzschild_phase2_batch.cu
