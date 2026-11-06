# `bridge/` — Python ↔ C bindings

This directory holds **minimal** embedding code (PyBind11) that exposes selected `kernel/` entry
points to Python.

Design rules:

- **No Schwarzschild/Christoffel math** belongs here beyond marshalling dtypes and arrays.
- All physics stays in **`kernel/`** (`bh_rt_*` sources and headers).

## Current surface

| Translation unit | Wrapped API | Python wrapper |
|-----------------|-------------|----------------|
| [`module_phase2.cpp`](module_phase2.cpp) | `bh_rt_schwarzschild_phase2_trace` (single-ray) | `native_phase2.schwarzschild_phase2_trace_native` |
| [`module_phase2.cpp`](module_phase2.cpp) | `bh_rt_schwarzschild_phase2_batch_trace` (N-ray) | `native_phase2.schwarzschild_phase2_batch_native` |
| [`module_phase3.cpp`](module_phase3.cpp) | `bh_rt_kerr_phase3_trace` (single-ray) | `native_phase3.kerr_phase3_trace_native` |
| [`module_phase3.cpp`](module_phase3.cpp) | `bh_rt_kerr_phase3_batch_trace` (N-ray) | `native_phase3.kerr_phase3_batch_native` |

## Batch output contract

`schwarzschild_phase2_batch_trace` returns a Python `dict` with these NumPy arrays:

| Key | dtype | Shape | Description |
|-----|-------|-------|-------------|
| `status` | `int32` | `(N,)` | `BH_RT_STATUS_*` code |
| `steps_taken` | `int32` | `(N,)` | Steps consumed |
| `termination_r` | `float64` | `(N,)` | r coordinate at termination |
| `r_min` | `float64` | `(N,)` | Minimum r reached (NaN if not tracked) |
| `disk_hit` | `int32` | `(N,)` | 1 when first strict equatorial crossing exists |
| `disk_r_cross` | `float64` | `(N,)` | Interpolated r at first crossing |
| `disk_vt` | `float64` | `(N,)` | Interpolated v^t at first crossing |
| `disk_vph` | `float64` | `(N,)` | Interpolated v^phi at first crossing |
| `eq_r_cross` | `float64` | `(N,)` | Compatibility alias for `disk_r_cross` |

Disk values are NaN when no crossing occurs. `disk_hit` is zero in that case.

## Build

```bash
BLACKHOLE_BUILD_NATIVE=1 uv sync   # Linux/macOS
# Windows: requires MSVC and same env var
```

After building, `from blackhole_ray_tracer._native_phase2 import ...` works.
Python helpers live in [`native_phase2.py`](../src/blackhole_ray_tracer/native_phase2.py).
