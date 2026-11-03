# Session starter — blackhole_ray_tracer

Copy the **Session block** below into Claude, Cursor Composer, or any agent thread at the start of a work session.
Persistent rules live in `.cursorrules` (Cursor) — this file is for per-session context only.

---

## Session block

You are assisting on **blackhole_ray_tracer**, a Python-first numerical null-geodesic ray tracer.
Phase 1 and Phase 2 are qualified reference/native layers. Phase 3 is a validated Python-only Kerr prototype.
CUDA and ML remain optional, explicitly bounded experiments.

### Read first

1. `docs/OVERVIEW.md` — current architecture and module map.
2. `docs/ROADMAP.md` — phases, acceptance criteria, CLI reference, and gaps.
3. `docs/STATUS.md` — observed evidence and unavailable-environment boundaries.
4. `docs/PROJECT_CONTRACT.md` and `docs/IMPLEMENTATION/SPEC.md` — normative behavior.
5. `.cursorrules` — hard rules that always apply.

### Ground truth (current)

**Python** (`src/blackhole_ray_tracer/`):
- Phase 1: equatorial Schwarzschild and shared `phase1.rk4_step`.
- Phase 2: Schwarzschild Christoffel/RK4, static camera, disk, batch/render/report/driver.
- Phase 3: Kerr BL metric/geodesic, ZAMO camera, invariants/Carter checks, render/report/driver.
- Native façades: `native_phase2.py`, optional `native_phase2_cuda.py`, and `preview.py`.

**C kernel** (`kernel/src/`, `kernel/include/`):
- Canonical Phase 2 scalar and SoA batch Schwarzschild APIs.
- `bh_rt_status.h` canonical status constants `0`–`3`.
- Disk crossing outputs and documented ABI in `docs/STATE_API.md`.
- `bh_rt_kerr_phase3.h/.c` provides tested C BL metric/RHS, bounded single-ray termination, ZAMO seed, first equatorial crossing, and geometry-independent disk observables; no color renderer, bridge, batch, or CUDA.

**Bridge** (`bridge/`):
- `module_phase2.cpp` PyBind11 module with scalar and batch CPU entrypoints.

**CUDA** (`kernel/cuda/`):
- Optional Phase 2 batch launcher with checked host allocation, copies, launch, sync, and cleanup.
- Local `nvcc`/GPU qualification is environment-dependent.

**ML** (`ml/`):
- Versioned dataset/model metadata, strict float32 artifacts, holdout check, runtime validation, explicit RK4 fallback.
### Batch render path

`phase2_render.py` uses `build_camera_y0()` (`phase2_batch.py`) to construct an `(N, 8)` initial-state
array. It chooses CUDA only when `cfg.use_native_phase2`, `cfg.use_cuda_batch`, and CUDA availability hold.
Otherwise it chooses native CPU batch when available, then Python or ML fallback paths.
Batch outputs include disk-crossing columns (`disk_hit`, `disk_r_cross`, `disk_vt`, `disk_vph`) for coloring.

### FFI contract

See `docs/STATE_API.md`. Status codes 0–3 match `phase1.RayStatus`.
Batch outputs also include `r_min`, disk crossing values, and `int32` status/step arrays.
CUDA launcher errors return nonzero CUDA codes; Python raises `RuntimeError`.

### Toolchain reminder

```
uv sync --group dev               # install pytest, ruff, mypy
BLACKHOLE_BUILD_NATIVE=1 uv sync  # enable PyBind11 build when compiler exists
uv run pytest
uv run ruff check src tests ml
uv run mypy src ml
```

**macOS native build:** requires Xcode Command Line Tools with a working C++ standard library
(`<cstddef>` available). If the PyBind build fails, use `uv sync` without `BLACKHOLE_BUILD_NATIVE=1`;
pytest kernel/bridge tests skip when the extension is absent. CPU kernel demos still build via
`make -C kernel`.

### Workflow

1. State which **ROADMAP phase** and **acceptance rows** your change targets.
2. Cite **concrete file paths** in all explanations.
3. If uncertain about a file's contents or a function's existence — **check first, don't assume**.
4. If adding C / GPU / ML logic, draft the API or update `docs/STATE_API.md` before writing implementation.
5. Author commits as `gabegaines24@gmail.com`.

---

## Today's goal

> *(Paste one sentence here — e.g. "Add accretion disk intersection" or "Implement Kerr design doc".)*
