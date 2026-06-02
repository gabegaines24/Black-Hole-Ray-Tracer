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

### Ground truth (as of Milestones 1–4)

**Python** (`src/blackhole_ray_tracer/`):
- Phase 1: `phase1.py`, `phase1_driver.py`, `phase1_image.py`, `phase1_tuning.py`
- Phase 2: `phase2_christoffel.py`, `phase2_geodesic.py`, `phase2_camera.py`, `phase2_types.py`,
  `phase2_render.py`, `phase2_report.py`, `phase2_driver.py`, `phase2_batch.py`
- Native bridge wrappers: `native_phase2.py`

**C kernel** (`kernel/src/`, `kernel/include/`):
- `bh_rt_rk4.c/.h` — generic RK4 step
- `bh_rt_schwarzschild_phase2.c/.h` — single-ray 3D Schwarzschild trace
- `bh_rt_schwarzschild_phase2_batch.c/.h` — N-ray SoA batch trace
- `bh_rt_schwarzschild_phase2_internal.h` — shared step helpers (Christoffel, renormalize, etc.)
- `bh_rt_status.h` — canonical `BH_RT_STATUS_*` int constants (0–3)
- `bh_rt_schwarzschild_phase2_batch.h` — batch API contract (matches `docs/STATE_API.md`)

**Bridge** (`bridge/`):
- `module_phase2.cpp` — PyBind11 `_native_phase2` module; exposes both
  `schwarzschild_phase2_trace` (single-ray) and `schwarzschild_phase2_batch_trace` (batch).

**Build**:
- `setup.py` — builds `blackhole_ray_tracer._native_phase2`; optional on Windows
  (set `BLACKHOLE_BUILD_NATIVE=1` + MSVC to enable).
- CI: `.github/workflows/ci.yml` — runs pytest on ubuntu-latest with native build.

**`ml/`** is still empty scaffold — do not reference as implemented.

### Batch render path

`phase2_render.py` uses `build_camera_y0()` (`phase2_batch.py`) to construct an `(N, 8)` initial-state
array, then calls `schwarzschild_phase2_batch_native()` (one C call for the whole image) when
`cfg.use_native_phase2` is True and the extension is available. Falls back to the Python loop otherwise.

### FFI contract

See `docs/STATE_API.md`. Status codes 0–3 match `phase1.RayStatus`.
Batch output arrays: `status(N,int32)`, `steps_taken(N,int32)`, `termination_r(N,float64)`, `r_min(N,float64)`.

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
