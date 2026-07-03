# Project overview — blackhole_ray_tracer

## Purpose

Simulate **null geodesics** (light rays) in strong gravity, starting from **Schwarzschild** and evolving toward **Kerr** / production performance. Architecture goal: move heavy integration into **C** (SIMD-friendly) or **GPU** while **Python** handles presets, orchestration, tests, ML tooling, and fast prototypes.

See also: [ROADMAP.md](./ROADMAP.md) for phases and acceptance criteria, [AGENT_PROMPT.md](./AGENT_PROMPT.md) for per-session starters, [STATE_API.md](./STATE_API.md) for draft FFI / SoA rays, **`.cursorrules`** (repo root) for Cursor persistent rules.

## What exists today vs planned

### Implemented (Python)

Under `src/blackhole_ray_tracer/`:

| Area | Modules | Role |
|------|-----------|------|
| **Phase 1** | `phase1.py`, `phase1_driver.py`, `phase1_image.py`, `phase1_tuning.py` | RK4 sanity (harmonic oscillator), equatorial Schwarzschild tracing in \(u(\phi)=1/r\), single ray logging, batch impact-parameter sweeps, simple **Einstein ring**–style PPM, tuning presets/report |
| **Phase 2** | `phase2_*.py` | Spherical Schwarzschild **Christoffel + RK4** in affine parameter \((x^\mu,v^\mu)\); static observer **pinhole** camera; **3D shadow** PPM via per-pixel rays; presets/benchmark (`phase2_report.py`), driver CLI |
| **Phase 3 (Kerr)** | `phase3_*.py` | Boyer–Lindquist Kerr null geodesics + pinhole render; optional native C batch when `_native_phase3` is installed |
| **Entry** | `main.py` (`blackhole-ray-tracer` console script), `phase1_driver`, `phase2_driver`, `phase3_driver` | User-facing CLI |
| **Bridge (optional)** | `native_phase2.py`, `native_phase3.py`, extensions `_native_phase2` / `_native_phase3` from [`bridge/`](../bridge/README.md) | PyBind11 single-ray + SoA batch trace; skip if wheel built without compilers |
| **Tests** | `tests/test_phase1_extensions.py`, `tests/test_phase2.py`, `tests/test_phase3_*.py`, kernel + bridge skips | Regression / smoke |

Tooling: `pyproject.toml` — `uv` for env/lockfile; dependency group `dev` has `pytest`, `ruff`, `mypy`.

### Implemented (C kernel, bridge, ML)

| Area | Location | Role |
|------|-----------|------|
| **C kernel** | `kernel/` | RK4 core; Schwarzschild 2D/3D + **Kerr Phase 3** Christoffel tracers; SoA batch APIs |
| **Bridge** | `bridge/module_phase2.cpp`, `bridge/module_phase3.cpp` | PyBind11 `_native_phase2` (Schwarzschild batch) and `_native_phase3` (Kerr batch) |
| **ML scaffold** | `ml/` | NumPy MLP surrogate, dataset generator, `RuntimeGate`; wired into `phase2_render` via `use_ml_surrogate` |
| **CUDA (optional)** | `kernel/cuda/`, `native_phase2_cuda.py` | GPU batch kernel + ctypes loader; wired into `phase2_render` via `use_cuda_batch` |

High-level layering:

```mermaid
flowchart TB
  cli["CLI — main, phase1_driver, phase2_driver, phase3_driver"]
  pyproto["Python physics — phase1, phase2, phase3"]
  bridge["bridge — pybind Phase2 + Phase3"]
  kernel["kernel — C RK4 + Schwarzschild + Kerr"]
  ml["ml — surrogate scaffold"]
  cuda["cuda — optional GPU batch"]

  cli --> pyproto
  pyproto --> bridge
  bridge --> kernel
  pyproto -.-> ml
  cuda -.->|"ctypes, optional"| pyproto
```

## Conventions and strategy

- **Units:** Geometric units \(G = c = 1\); Schwarzschild radius \(r_s = 2M\).
- **Future Kerr:** **Boyer–Lindquist** coordinates in Python (`phase3_*`) and C (`bh_rt_kerr_phase3_*`); Schwarzschild is the \(a \to 0\) limit.
- **Performance direction:** **SoA** (arrays of `r[]`, `phi[]`, …) for SIMD; **early exit** when \(r < r_\text{horizon}+\epsilon\) or \(r > r_\text{escape}\).
- **Phase discipline:** New experiment layers should stay in clear modules (`phaseN_*` or future `kernel/`) rather than growing one monolith.

## Development commands

```bash
uv sync
uv sync --group dev   # pytest, ruff, mypy

uv run blackhole-ray-tracer --help
uv run pytest
uv run ruff check src tests
uv run mypy src       # after dev group
```

## Primary file map (quick reference)

| Path | Notes |
|------|--------|
| `src/blackhole_ray_tracer/main.py` | `blackhole-ray-tracer` script; Phase 1 + Phase 2 flags |
| `src/blackhole_ray_tracer/phase1.py` | Shared `rk4_step`, equatorial ray trace, batch helpers |
| `src/blackhole_ray_tracer/phase1_driver.py` | Phase 1 steps A–F with argparse |
| `src/blackhole_ray_tracer/phase1_image.py` | PPM I/O; Einstein-ring render |
| `src/blackhole_ray_tracer/phase1_tuning.py` | Phase 1 presets / benchmark |
| `src/blackhole_ray_tracer/phase2_christoffel.py` | Schwarzschild Christoffel / ODE RHS |
| `src/blackhole_ray_tracer/phase2_geodesic.py` | 3D null RK4 tracer |
| `src/blackhole_ray_tracer/phase2_camera.py` | Static observer pinhole → initial null 4-velocity |
| `src/blackhole_ray_tracer/phase2_types.py` | `Phase2RenderConfig`, camera, trace result types |
| `src/blackhole_ray_tracer/phase2_render.py` | Per-pixel pinhole image loop; optional **`use_native_phase2`** → `_native_phase2` |
| `src/blackhole_ray_tracer/phase2_report.py` | Phase 2 presets + benchmark text |
| `src/blackhole_ray_tracer/phase2_driver.py` | Phase 2 `--render` / `--report` CLI |
| `src/blackhole_ray_tracer/phase3_christoffel.py` | Kerr Christoffel / ODE RHS (BL coords) |
| `src/blackhole_ray_tracer/phase3_geodesic.py` | Kerr null RK4 tracer |
| `src/blackhole_ray_tracer/phase3_render.py` | Kerr pinhole image; native batch when `use_native_phase3` |
| `src/blackhole_ray_tracer/phase3_driver.py` | Phase 3 Kerr render CLI (`--native`) |
| `src/blackhole_ray_tracer/native_phase3.py` | Optional Kerr C extension façade |
| `kernel/include/bh_rt_kerr_phase3.h`, `kernel/src/bh_rt_kerr_phase3.c` | Kerr single-ray C trace (parity with `phase3_geodesic`) |
| `kernel/include/bh_rt_kerr_phase3_batch.h`, `kernel/src/bh_rt_kerr_phase3_batch.c` | Kerr SoA batch trace + `eq_r_cross` |
| `bridge/module_phase3.cpp` | PyBind11 `_native_phase3` extension |
| `kernel/include/bh_rt_rk4.h`, `kernel/src/bh_rt_rk4.c` | Shared C **RK4** step (no Python) |
| `kernel/include/bh_rt_schwarzschild_u.h`, `kernel/src/bh_rt_schwarzschild_u.c` | Schwarzschild 2D \(u(\phi)=1/r\) ray trace (parity with `phase1.py`) |
| `kernel/include/bh_rt_schwarzschild_phase2.h`, `kernel/src/bh_rt_schwarzschild_phase2.c` | Schwarzschild **3D** null tracer (Christoffel + RK4; parity with `phase2_geodesic.py`) |
| `kernel/include/bh_rt_schwarzschild_3d.h`, `kernel/src/bh_rt_schwarzschild_3d.c` | Scalar + SoA-batch Schwarzschild **3D** trace API for the next bridge/batch path |
| `kernel/src/demo_harmonic.c`, `kernel/src/demo_schwarzschild_u.c`, `kernel/Makefile` | RK4 + Schwarzschild CLI demos (`make -C kernel`) |
| `bridge/module_phase2.cpp`, `MANIFEST.in`, `setup.py` | setuptools + PyBind11 extension `_native_phase2` embedding `bh_rt_*` sources |
| `src/blackhole_ray_tracer/native_phase2.py` | Optional import façade + `native_phase2_available()` |
| `kernel/README.md` | Kernel build and next steps |
| `bridge/README.md` | Bridge directory rules and naming |
