# CUDA batch kernel

GPU-accelerated null-geodesic ray tracing for the Schwarzschild Phase 2 batch path.

## Files

| File | Description |
|------|-------------|
| `bh_rt_schwarzschild_phase2_batch.cu` | CUDA kernel: each thread traces one ray |
| `Makefile` | Standalone `nvcc` build target |

## Build

Prerequisites: CUDA toolkit with `nvcc` on `PATH`.

```bash
# From this directory
cd kernel/cuda

# Default (sm_75 = Turing, e.g. GTX 1650 Super / RTX 20xx)
make cuda_batch

# Override architecture for your GPU
make cuda_batch ARCH=sm_86   # Ampere (RTX 30xx)
make cuda_batch ARCH=sm_89   # Ada Lovelace (RTX 40xx)

# Output: ../../build/libbh_rt_phase2_cuda.so
```

### Windows (`nvcc` command-line)

```bat
nvcc -O2 -arch=sm_75 -I..\include --compiler-options /LD ^
     -o ..\..\build\bh_rt_phase2_cuda.dll ^
     bh_rt_schwarzschild_phase2_batch.cu
```

## Python usage

The library is loaded at runtime by
[`native_phase2_cuda.py`](../../src/blackhole_ray_tracer/native_phase2_cuda.py):

```python
from blackhole_ray_tracer.native_phase2_cuda import cuda_batch_available, schwarzschild_phase2_batch_cuda

if cuda_batch_available():
    result = schwarzschild_phase2_batch_cuda(y0_batch, m=1.0, dlambda=0.06, ...)
```

Set `BLACKHOLE_CUDA_LIB=/path/to/libbh_rt_phase2_cuda.so` to override the automatic
`build/` search path.

## Architecture notes

The kernel mirrors the CPU batch kernel (`bh_rt_schwarzschild_phase2_batch.c`) with identical
physics (Christoffel + RK4). Each CUDA thread traces one ray independently (embarrassingly
parallel). No inter-thread communication is needed.

Grid/block sizing: 1 thread per ray, 256 threads per block (tunable).

## GPU architecture reference

| GPU family | `sm_XX` value |
|------------|---------------|
| Turing (GTX 1650, RTX 20xx) | `sm_75` |
| Ampere (RTX 30xx, A100) | `sm_86` / `sm_80` |
| Ada Lovelace (RTX 40xx) | `sm_89` |
| Hopper (H100) | `sm_90` |
