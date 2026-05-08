# `bridge/` — Python ↔ C bindings

This directory holds **minimal** embedding code (`PyBind11`) that exposes selected `kernel/` entry points to Python.

Design rules:

- **No Schwarzschild/Christoffel math** belongs here beyond marshalling dtypes and arrays.
- All physics stays in **`kernel/`** (`bh_rt_*` sources and headers).

Current surface:

| Translation unit | Wrapped API |
|------------------|-------------|
| [`module_phase2.cpp`](module_phase2.cpp) | [`bh_rt_schwarzschild_phase2_trace`](../kernel/include/bh_rt_schwarzschild_phase2.h) |

Build hooks live in repo-root [`setup.py`](../setup.py); Python helpers in [`native_phase2.py`](../src/blackhole_ray_tracer/native_phase2.py). Import `blackhole_ray_tracer._native_phase2` after an editable/`uv` install with compilers enabled.
