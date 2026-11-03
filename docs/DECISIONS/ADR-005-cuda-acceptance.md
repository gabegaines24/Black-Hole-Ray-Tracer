# ADR-005: Optional CUDA acceptance

## Context

CUDA source exists, but local qualification depends on `nvcc`, a compatible GPU library, and hardware.
Unchecked host errors could expose stale or uninitialized output arrays.

## Decision

Treat CUDA as optional. The host launcher must check every allocation, memory copy, kernel launch,
synchronization, and cleanup operation. Return `int32_t` CUDA error codes to ctypes.
Python raises `RuntimeError` for nonzero codes and never silently labels a failed call successful.
CI and documentation may skip CUDA only with explicit unavailable evidence.

## Consequences

- CPU and Python paths remain usable without CUDA.
- CUDA parity claims require a machine that builds and runs the kernel.
- Benchmark numbers must include backend, platform, image size, step budget, and availability status.

## Evidence

- `kernel/cuda/bh_rt_schwarzschild_phase2_batch.cu`
- `src/blackhole_ray_tracer/native_phase2_cuda.py`
- `tests/test_cuda_batch_parity.py`
- `docs/STATUS.md`
