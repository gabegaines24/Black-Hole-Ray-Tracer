# C kernel (`kernel/`)

Pure C primitives for high-throughput tracing. **No Python headers** here.

## File map

| Path | Purpose |
|------|---------|
| `include/bh_rt_rk4.h` | Generic **RK4** step (function-pointer RHS) |
| `src/bh_rt_rk4.c` | RK4 implementation (`workspace` = `5 × n` doubles) |
| `include/bh_rt_schwarzschild_u.h` | Equatorial 2D Schwarzschild trace in u(φ) |
| `src/bh_rt_schwarzschild_u.c` | Mirrors `phase1.trace_single_schwarzschild_ray` |
| `include/bh_rt_status.h` | Canonical `BH_RT_STATUS_*` int constants (0–3) |
| `include/bh_rt_schwarzschild_phase2.h` | Phase 2 single-ray API + result struct |
| `src/bh_rt_schwarzschild_phase2.c` | 3D Schwarzschild trace; exposes shared helpers |
| `include/bh_rt_schwarzschild_phase2_internal.h` | Shared Christoffel / renormalize helpers |
| `include/bh_rt_schwarzschild_phase2_batch.h` | N-ray SoA batch API contract |
| `src/bh_rt_schwarzschild_phase2_batch.c` | CPU batch implementation |
| `cuda/bh_rt_schwarzschild_phase2_batch.cu` | CUDA port of the batch kernel |
| `src/demo_harmonic.c` | Harmonic oscillator RK4 smoke demo |
| `src/demo_schwarzschild_u.c` | 2D Schwarzschild one-result demo |
| `src/demo_schwarzschild_phase2_batch.c` | Phase 2 batch smoke demo (4 rays) |

## Build (CPU)

```bash
make -C kernel          # builds all: harmonic_demo, schwarzschild_demo, phase2_batch_demo
make -C kernel clean
```

### Batch demo smoke test

```bash
./kernel/phase2_batch_demo
# Expected prefix: BH_PHASE2_BATCH_RESULT ...
```

### Shared library for ctypes parity tests (POSIX)

```bash
cc -std=c99 -Wall -O2 -fPIC -shared -I kernel/include \
  kernel/src/bh_rt_rk4.c \
  kernel/src/bh_rt_schwarzschild_phase2.c \
  kernel/src/bh_rt_schwarzschild_phase2_batch.c \
  -o /tmp/libbh_rt_phase2_batch.so -lm
```

## Build (CUDA — Phase 4)

Requires CUDA 11+ and a GPU with compute capability ≥ 7.5 (GTX 1650 Super = sm_75).

```bash
nvcc -O2 -arch=sm_75 -I kernel/include \
     --compiler-options -fPIC -shared \
     -o build/libbh_rt_phase2_cuda.so \
     kernel/cuda/bh_rt_schwarzschild_phase2_batch.cu
```

Then set the env variable so the Python wrapper can find it:

```bash
export BLACKHOLE_CUDA_LIB=build/libbh_rt_phase2_cuda.so
uv run python -c "from blackhole_ray_tracer.native_phase2_cuda import cuda_batch_available; print(cuda_batch_available())"
```

The CUDA kernel uses the same `(N×8)` input layout and the same four output arrays as the CPU batch
kernel — see `docs/STATE_API.md`.

## Tests

| Test file | What it checks |
|-----------|---------------|
| `tests/test_kernel_harmonic_parity.py` | RK4 vs analytic harmonic oscillator |
| `tests/test_kernel_schwarzschild_u_parity.py` | C 2D tracer vs Python |
| `tests/test_kernel_phase2_parity.py` | C single-ray 3D vs Python |
| `tests/test_kernel_phase2_batch_parity.py` | C N-ray batch vs N Python calls |

All kernel tests skip automatically if no C compiler is present (`cc`/`gcc`/`clang`).
