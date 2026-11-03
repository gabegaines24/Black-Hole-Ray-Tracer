.PHONY: all harmonic_demo schwarzschild_demo phase2_batch_demo cuda clean

CC ?= cc
CFLAGS ?= -std=c99 -Wall -Wextra -O2 -Iinclude
NVCC ?= nvcc
CUDA_ARCH ?= sm_75
BUILD_DIR ?= ../build

PHASE2_SRCS = src/bh_rt_rk4.c src/bh_rt_schwarzschild_phase2.c
PHASE2_BATCH_SRCS = $(PHASE2_SRCS) src/bh_rt_schwarzschild_phase2_batch.c

all: harmonic_demo schwarzschild_demo phase2_batch_demo

harmonic_demo: src/bh_rt_rk4.c src/demo_harmonic.c include/bh_rt_rk4.h
	$(CC) $(CFLAGS) -o harmonic_demo src/bh_rt_rk4.c src/demo_harmonic.c -lm

schwarzschild_demo: src/bh_rt_rk4.c src/bh_rt_schwarzschild_u.c src/demo_schwarzschild_u.c \
		include/bh_rt_rk4.h include/bh_rt_schwarzschild_u.h
	$(CC) $(CFLAGS) -o schwarzschild_demo src/bh_rt_rk4.c src/bh_rt_schwarzschild_u.c \
		src/demo_schwarzschild_u.c -lm

phase2_batch_demo: $(PHASE2_BATCH_SRCS) src/demo_schwarzschild_phase2_batch.c \
		include/bh_rt_rk4.h include/bh_rt_schwarzschild_phase2.h \
		include/bh_rt_schwarzschild_phase2_batch.h \
		include/bh_rt_schwarzschild_phase2_internal.h \
		include/bh_rt_status.h
	$(CC) $(CFLAGS) -o phase2_batch_demo \
		$(PHASE2_BATCH_SRCS) src/demo_schwarzschild_phase2_batch.c -lm

cuda:
	@command -v $(NVCC) >/dev/null 2>&1 || { \
		echo "CUDA build unavailable: nvcc not found ($(NVCC))." >&2; \
		exit 2; \
	}
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -O2 -arch=$(CUDA_ARCH) -Iinclude --compiler-options -fPIC -shared \
		-o $(BUILD_DIR)/libbh_rt_phase2_cuda.so \
		cuda/bh_rt_schwarzschild_phase2_batch.cu

clean:
	rm -f harmonic_demo harmonic_demo.exe \
	      schwarzschild_demo schwarzschild_demo.exe \
	      phase2_batch_demo phase2_batch_demo.exe
