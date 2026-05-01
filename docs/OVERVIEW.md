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
| **Entry** | `main.py` (`blackhole-ray-tracer` console script), `phase1_driver`, `phase2_driver` module entrypoints | User-facing CLI |
| **Tests** | `tests/test_phase1_extensions.py`, `tests/test_phase2.py` | Regression / smoke |

Tooling: `pyproject.toml` — `uv` for env/lockfile; dependency group `dev` has `pytest`, `ruff`, `mypy`.

### Planned / scaffold only (verify before assuming implementation)

Directories **`kernel/`**, **`bridge/`**, and **`ml/`** are described in the README as the long-term split (pure C integration, pybind11 bridge, ML surrogate pipeline). **`kernel/`** now hosts a generic **RK4** step and a **harmonic oscillator** demo (`make -C kernel`). **`bridge/`** and **`ml/`** may still be **empty** until those milestones start. Do not assume a pybind extension exists until you see it under `bridge/`.

High-level layering:

```mermaid
flowchart TB
  cli["CLI — main, phase1_driver, phase2_driver"]
  pyproto["Python physics — phase1, phase2"]
  bridge["bridge — pybind11 planned"]
  kernel["kernel — C RK4 + demos started"]
  ml["ml — surrogate planned"]

  cli --> pyproto
  pyproto -.->|"future offload"| bridge
  bridge -.-> kernel
  pyproto -.-> ml
```

## Conventions and strategy

- **Units:** Geometric units \(G = c = 1\); Schwarzschild radius \(r_s = 2M\).
- **Future Kerr:** Target **Boyer–Lindquist** coordinates so Schwarzschild is the \(a \to 0\) limit of one code path.
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
| `src/blackhole_ray_tracer/phase2_render.py` | Per-pixel pinhole image loop |
| `src/blackhole_ray_tracer/phase2_report.py` | Phase 2 presets + benchmark text |
| `src/blackhole_ray_tracer/phase2_driver.py` | Phase 2 `--render` / `--report` CLI |
| `kernel/include/bh_rt_rk4.h`, `kernel/src/bh_rt_rk4.c` | Shared C **RK4** step (no Python) |
| `kernel/src/demo_harmonic.c`, `kernel/Makefile` | Phase A harmonic parity demo; `make -C kernel` |
| `kernel/README.md` | Kernel build and next steps |
