# Session starter — blackhole_ray_tracer

Copy the **Session block** below into Claude, Cursor Composer, or any agent thread at the start of a work session.
Persistent rules live in `.cursorrules` (Cursor) — this file is for per-session context only.

---

## Session block

You are assisting on **blackhole_ray_tracer**, a Python-first numerical null-geodesic ray tracer.

### Read first

1. `docs/OVERVIEW.md` — architecture, what is implemented vs scaffold.
2. `docs/ROADMAP.md` — phases, acceptance criteria, CLI reference, gaps checklist.
3. `.cursorrules` — hard rules that always apply (toolchain, diff discipline, naming).

### Ground truth (as of Next Phase Plan A–D)

**Python** (`src/blackhole_ray_tracer/`):
- Phase 1: `phase1.py`, `phase1_driver.py`, `phase1_image.py`, `phase1_tuning.py`
- Phase 2: `phase2_christoffel.py`, `phase2_geodesic.py`, `phase2_camera.py`, `phase2_types.py`,
  `phase2_render.py`, `phase2_report.py`, `phase2_driver.py`, `phase2_batch.py`, `phase2_disk.py`
- **Phase 3 (Kerr)**: `phase3_christoffel.py`, `phase3_geodesic.py`, `phase3_types.py`, `phase3_render.py`
- Native bridge wrappers: `native_phase2.py`, `native_phase2_cuda.py`

**C kernel** (`kernel/src/`, `kernel/include/`):
- `bh_rt_rk4.c/.h` — generic RK4 step
- `bh_rt_schwarzschild_phase2.c/.h` — single-ray 3D Schwarzschild trace
- `bh_rt_schwarzschild_phase2_batch.c/.h` — N-ray SoA batch trace (outputs: status, steps_taken, termination_r, r_min, **eq_r_cross**)
- `bh_rt_schwarzschild_phase2_internal.h` — shared step helpers (Christoffel, renormalize, etc.)
- `bh_rt_status.h` — canonical `BH_RT_STATUS_*` int constants (0–3)
- `kernel/cuda/bh_rt_schwarzschild_phase2_batch.cu` — CUDA batch kernel (buildable via `kernel/cuda/Makefile`)

**Bridge** (`bridge/`):
- `module_phase2.cpp` — PyBind11 `_native_phase2` module; exposes both
  `schwarzschild_phase2_trace` (single-ray) and `schwarzschild_phase2_batch_trace` (batch with `eq_r_cross`).

**Build**:
- `setup.py` — builds `blackhole_ray_tracer._native_phase2`; optional on Windows
  (set `BLACKHOLE_BUILD_NATIVE=1` + MSVC to enable).
- `kernel/cuda/Makefile` — `make -C kernel/cuda cuda_batch ARCH=sm_75` → `build/libbh_rt_phase2_cuda.so`
- CI: `.github/workflows/ci.yml` — pytest + native build on ubuntu-latest; optional CUDA step (`CUDA_AVAILABLE=1`).

**`ml/`** — functional scaffold:
- `ml/schema.py` — normalisation constants + I/O contract (6 inputs, 4 outputs)
- `ml/dataset.py` — generate `(X, Y)` training pairs from Python integrator; CLI: `python -m ml.dataset --n-rays 5000`
- `ml/surrogate.py` — pure NumPy 3-layer MLP, mini-batch SGD; CLI: `python -m ml.surrogate train`
- `ml/runtime_gate.py` — `RuntimeGate`: routes r > 10M to surrogate, r ≤ 10M to RK4

### Batch render path

`phase2_render.py` uses `build_camera_y0()` (`phase2_batch.py`) to construct an `(N, 8)` initial-state
array, then calls `schwarzschild_phase2_batch_native()` (one C call for the whole image) when
`cfg.use_native_phase2` is True and the extension is available. Falls back to the Python loop otherwise.

The batch result now includes `eq_r_cross(N,float64)` — first equatorial crossing per ray — used for
disk coloring in the native batch path when `cfg.disk` is set.

Anti-aliasing: set `Phase2RenderConfig.supersample = 2` (or `--aa 2` CLI) to render at 2× resolution
then box-average. Works for both Python and batch paths.

### Kerr geodesic path

`phase3_render.render_kerr_3d_image(cfg: KerrRenderConfig)` dispatches to Phase 2 when `cfg.a == 0`,
otherwise uses the Python Kerr integrator (`phase3_geodesic.trace_kerr_null_geodesic`).

### FFI contract

See `docs/STATE_API.md`. Status codes 0–3 match `phase1.RayStatus`.
Batch output arrays: `status(N,int32)`, `steps_taken(N,int32)`, `termination_r(N,float64)`,
`r_min(N,float64)`, `eq_r_cross(N,float64)`.

### Toolchain reminder

```
uv sync --group dev               # Windows (no native build by default)
BLACKHOLE_BUILD_NATIVE=1 uv sync  # Linux/macOS or Windows with MSVC
uv run pytest
uv run ruff check src tests
```

### Workflow

1. State which **ROADMAP phase** and **acceptance rows** your change targets.
2. Cite **concrete file paths** in all explanations.
3. If uncertain about a file's contents or a function's existence — **check first, don't assume**.
4. If adding C / GPU / ML logic, draft the API or update `docs/STATE_API.md` before writing implementation.
5. Author commits as `gabegaines24@gmail.com`.

---

## Today's goal

> *(Paste one sentence here — e.g. "Add accretion disk intersection" or "Implement Kerr design doc".)*
