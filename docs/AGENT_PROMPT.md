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

### Ground truth (render backends + kernel)

**Phase 2 render backend priority** (`phase2_render.py`):

1. `cuda_batch` — `use_cuda_batch` + `libbh_rt_phase2_cuda.so`
2. `native_batch` — `use_native_phase2` + `_native_phase2`
3. `ml_surrogate` — `use_ml_surrogate` + trained model (`phase2_ml.py` / `RuntimeGate`)
4. `python` — per-pixel Python RK4 fallback

**Python** (`src/blackhole_ray_tracer/`):
- Phase 1: `phase1.py`, `phase1_driver.py`, `phase1_image.py`, `phase1_tuning.py`
- Phase 2: `phase2_christoffel.py`, `phase2_geodesic.py`, `phase2_camera.py`, `phase2_types.py`,
  `phase2_render.py`, `phase2_report.py`, `phase2_driver.py`, `phase2_batch.py`, `phase2_disk.py`,
  `phase2_ml.py`
- **Phase 3 (Kerr)**: `phase3_christoffel.py`, `phase3_geodesic.py`, `phase3_types.py`, `phase3_render.py`
- Native bridge wrappers: `native_phase2.py`, `native_phase3.py`, `native_phase2_cuda.py`

**C kernel** (`kernel/src/`, `kernel/include/`):
- `bh_rt_rk4.c/.h` — generic RK4 step
- `bh_rt_schwarzschild_phase2.c/.h` — single-ray 3D Schwarzschild trace
- `bh_rt_schwarzschild_phase2_batch.c/.h` — N-ray SoA batch trace (outputs: status, steps_taken, termination_r, r_min, **eq_r_cross**)
- `bh_rt_kerr_phase3.c/.h`, `bh_rt_kerr_phase3_batch.c/.h` — Kerr BL single + batch trace (parity with `phase3_geodesic.py`)
- `bh_rt_schwarzschild_phase2_internal.h` — shared step helpers (Christoffel, renormalize, etc.)
- `bh_rt_status.h` — canonical `BH_RT_STATUS_*` int constants (0–3)
- `kernel/cuda/bh_rt_schwarzschild_phase2_batch.cu` — CUDA batch kernel (buildable via `kernel/cuda/Makefile`)

**Bridge** (`bridge/`):
- `module_phase2.cpp` — PyBind11 `_native_phase2` module; exposes
  `schwarzschild_phase2_trace` (single-ray) and `schwarzschild_phase2_batch_trace` (batch with `eq_r_cross`).
- `module_phase3.cpp` — PyBind11 `_native_phase3` module; Kerr single + batch trace.

**Build**:
- `setup.py` — builds `_native_phase2` and `_native_phase3`; optional on Windows
  (set `BLACKHOLE_BUILD_NATIVE=1` + MSVC to enable).
- `kernel/cuda/Makefile` — `make -C kernel/cuda cuda_batch ARCH=sm_75` → `build/libbh_rt_phase2_cuda.so`
- CI: `.github/workflows/ci.yml` — pytest + native build on ubuntu-latest; optional CUDA step (`CUDA_AVAILABLE=1`).

**`ml/`** — functional scaffold:
- `ml/schema.py` — normalisation constants + I/O contract (6 inputs, 4 outputs)
- `ml/dataset.py` — generate `(X, Y)` training pairs from Python integrator; CLI: `python -m ml.dataset --n-rays 5000`
- `ml/surrogate.py` — pure NumPy 3-layer MLP, mini-batch SGD; CLI: `python -m ml.surrogate train`
- `ml/runtime_gate.py` — `RuntimeGate.infer_or_integrate` + batch; wired via `phase2_ml.py`

### Batch render path

`phase2_render.py` uses `build_camera_y0()` (`phase2_batch.py`) for an `(N, 8)` grid, then dispatches
to CUDA, C CPU batch, ML surrogate batch, or a Python per-pixel loop. Batch outputs include
`eq_r_cross` for disk coloring when using C or CUDA backends.

Anti-aliasing: `Phase2RenderConfig.supersample` or `--aa 2` (box-average downsample).

### Kerr geodesic path

`phase3_render.render_kerr_3d_image` uses Phase 2 native batch when `a=0`, Kerr C batch when
`use_native_phase3` and `a≠0`, otherwise Python Kerr integrator.

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
