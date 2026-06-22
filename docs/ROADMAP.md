# Roadmap — blackhole_ray_tracer

Canonical phased plan aligned with **what is in git** versus **planned**. Update this file when a milestone completes.

Companion: [OVERVIEW.md](./OVERVIEW.md), [STATE_API.md](./STATE_API.md) (draft batch / FFI contract).

## Vision and scope

**In scope**

- Reliable null-geodesic integration for Schwarzschild (done in Python prototypes; later in C/GPU).
- Clear APIs for ray batches, presets, regression tests.
- Educational, inspectable phases (`phase1`, `phase2`, …).

**Explicit non-goals (for early phases)**

- Real-time cinematic quality at 1080p in pure Python.
- Full radiative transfer or accretion physics before a stable geodesic core exists.

## Phase map

### Phase 1 — Equatorial Schwarzschild (Python) — substantially complete

| Step | Capability | Evidence in repo |
|------|-------------|----------------|
| A | RK4 on toy ODE (harmonic oscillator) | `phase1.run_rk4_sanity`, `phase1_driver --step a` |
| B | Single equatorial Schwarzschild ray \(u(\phi)\) | `trace_single_schwarzschild_ray`, `--phase1-step-b` |
| C | Ray **termination** / status (captured, escaped, max steps, numerical error) | Folded into `RayStatus` + `RayTraceResult` in `phase1.py`; surfaced in `summarize_phase1_a_b()`, `phase1_driver` Step B pass/fail, and batch rows — there is no separate `--step c` flag |
| D | Batch \(b\) sweep, table/CSV optional | `batch_schwarzschild_rays`, `--phase1-step-d` / driver `--step d` |
| E | Simple ring PPM (impact-parameter camera) | `phase1_image`, `--phase1-step-e` |
| F | Presets + single-ray \(d\phi\) benchmark | `phase1_tuning`, `--phase1-step-f` |

**Acceptance (maintain)**

- `uv run pytest` passes Phase 1 tests.
- RK4 sanity and at least one batch table render without NaNs for defaults.

Teaching notes remain in **`plan.txt`** (Phase 1 build sheet).

---

### Phase 2 — Full 3D Schwarzschild prototype (Python) — substantially complete

| Item | Status | Modules |
|------|--------|---------|
| Christoffel + RK4 affine null geodesic | Done | `phase2_christoffel.py`, `phase2_geodesic.py` |
| Static observer pinhole \((t,r,\theta,\phi)\) initializer | Done | `phase2_camera.py`, `phase2_types.py` |
| PPM shadow + synthetic sky | Done | `phase2_render.py`; `phase1_image.write_ppm_rgb` |
| Presets + benchmark report | Done | `phase2_report.py` |

**Known gaps**

- Phase 2 render loop is still **Python-orchestrated** (per-pixel); optional **C per-ray** path when `use_native_phase2` + `_native_phase2` is installed (see `phase2_render`, `--phase2-native`).
- No anti-aliasing, no accretion disk texture, no Kerr.

**Acceptance (maintain)**

- `uv run pytest` passes `tests/test_phase2.py`.
- `--phase2-report` and a small `--phase2-render` (e.g. 32×32) complete without crash.

---

### Phase 2+ / kernel bridge — in progress

| Item | Goal | Status |
|------|------|--------|
| **State API draft** | SoA + `RayStatus` codes for FFI | [STATE_API.md](./STATE_API.md) |
| **`kernel/` RK4 core** | Explicit RK4 step for future batched RHS | Done: [`kernel/src/bh_rt_rk4.c`](../kernel/src/bh_rt_rk4.c), [`kernel/include/bh_rt_rk4.h`](../kernel/include/bh_rt_rk4.h) |
| **Phase A harmonic parity** | C vs `phase1.run_rk4_sanity` | Done: [`kernel/src/demo_harmonic.c`](../kernel/src/demo_harmonic.c), `make -C kernel`, [`tests/test_kernel_harmonic_parity.py`](../tests/test_kernel_harmonic_parity.py) (skips if no C toolchain) |
| **Schwarzschild \(2D equatorial\) kernel** | `u(\phi)=1/r` loop vs `phase1.trace_single_schwarzschild_ray` | Done: [`kernel/include/bh_rt_schwarzschild_u.h`](../kernel/include/bh_rt_schwarzschild_u.h), [`kernel/src/bh_rt_schwarzschild_u.c`](../kernel/src/bh_rt_schwarzschild_u.c), [`kernel/src/demo_schwarzschild_u.c`](../kernel/src/demo_schwarzschild_u.c), [`tests/test_kernel_schwarzschild_u_parity.py`](../tests/test_kernel_schwarzschild_u_parity.py) (skipped without a toolchain or when `SKIP_KERNEL_TESTS=1`) |
| **Schwarzschild / Phase 2 \(3D Christoffel\) kernel** | Match `phase2_geodesic` Python RHS + termination | Done: [`kernel/include/bh_rt_schwarzschild_phase2.h`](../kernel/include/bh_rt_schwarzschild_phase2.h), [`kernel/src/bh_rt_schwarzschild_phase2.c`](../kernel/src/bh_rt_schwarzschild_phase2.c), [`tests/test_kernel_phase2_parity.py`](../tests/test_kernel_phase2_parity.py) (skipped without a toolchain or when `SKIP_KERNEL_TESTS=1`) |
| **`bridge/`** | pybind11 — single-ray Phase 3D trace (+ future batch API) | Started: [`bridge/module_phase2.cpp`](../bridge/module_phase2.cpp) → **`_native_phase2`**; optional **per-pixel** use from [`phase2_render.py`](../src/blackhole_ray_tracer/phase2_render.py) (`use_native_phase2`); pytest bridge parity skips if extension missing |

**Acceptance**

- `make -C kernel` builds **`harmonic_demo`** and **`schwarzschild_demo`** where `cc`/`gcc`/`clang` exists.
- **`uv pip install -e .` / `uv sync`** rebuilds **`_native_phase2`** where a suitable C++/C toolchain is configured ( MSVC / gcc / clang ).
- Full `pytest` run passes (kernel parity tests skip without a toolchain or when `SKIP_KERNEL_TESTS=1`; native bridge parity skips without the compiled extension).
- SoA batched bridge API remains future work ([`STATE_API.md`](./STATE_API.md)).

---

### Phase 3 — Kerr physics + visual fidelity + ML — substantially complete

| Item | Status | Modules |
|------|--------|---------|
| Kerr Christoffel (BL coords) + ODE RHS | Done | `phase3_christoffel.py` |
| Kerr null geodesic integrator (Python RK4) | Done | `phase3_geodesic.py` |
| `KerrRenderConfig` + dispatcher | Done | `phase3_types.py`, `phase3_render.py` |
| Kerr parity / E/L conservation tests | Done | `tests/test_phase3_kerr.py` |
| Disk detection in native batch path (`eq_r_cross`) | Done | `kernel/.../batch.c`, `bridge/`, `phase2_render.py` |
| Anti-aliasing (supersample + box-average) | Done | `phase2_types.py` supersample field, `phase2_render.py`, `--aa` flag |
| ML scaffold (schema, dataset, surrogate, gate) | Done | `ml/` |
| ML end-to-end tests | Done | `tests/test_ml.py` |

**Acceptance**

- `uv run pytest` passes Phase 3 + ML tests.
- `a=0` Kerr traces agree qualitatively with Phase 2 Schwarzschild.
- E/L conservation drift < 1% over 20 λ-steps away from the horizon.

---

### Phase 4 — GPU port — partially scaffolded

| Item | Goal | Status |
|------|------|--------|
| CUDA batch kernel | Preview-oriented throughput | `.cu` exists; build path via `kernel/cuda/Makefile` |
| Python ctypes bridge | Load `libbh_rt_phase2_cuda.so` at runtime | `native_phase2_cuda.py` |
| `setup.py` `nvcc` target | Pip-installable GPU path | Planned |

**Acceptance**

- `make -C kernel/cuda cuda_batch ARCH=sm_75` produces `build/libbh_rt_phase2_cuda.so` (requires CUDA toolkit).
- Optional CI step (`CUDA_AVAILABLE=1` repo variable) validates the build.

---

## CLI cheat sheet

### `blackhole-ray-tracer` (`src/blackhole_ray_tracer/main.py`)

| Flag | Effect |
|------|--------|
| `--phase1-ab` | Step A + B summary |
| `--phase1-step-b` | Step B log |
| `--phase1-step-d` | Step D table |
| `--phase1-step-e` | Step E PPM (`--ppm-out`, `--img-width`, `--img-height`, `--preset`) |
| `--phase1-step-f` | Step F report |
| `--phase2-report` | Phase 2 presets + benchmark |
| `--phase2-render` | Phase 2 PPM (`--phase2-out`, `--phase2-preset`, image size flags; `--phase2-native` for C per-ray) |

### `python -m blackhole_ray_tracer.phase1_driver`

| `--step` | Purpose |
|-----------|---------|
| `a`, `b`, `ab`, `d`, `e`, `f` | As module docstring; extra flags for CSV, presets, \(\phi\) step, \(r_\text{escape}\), etc. |

### `python -m blackhole_ray_tracer.phase2_driver`

| Flag | Purpose |
|------|---------|
| `--report` | Presets + `dlambda` benchmark |
| `--render` | 3D PPM (`--out`, `--preset` or `--width`/`--height`/`--dlambda`/…) |
| `--native` | Use C extension per ray when `_native_phase2` is installed |

Installation: typically `PYTHONPATH=src` when developing from checkout without reinstall.

---

## Where to extend code

| Change | Prefer |
|--------|--------|
| New Schwarzschild 2D experiment | `phase1.py` helpers or `phase1_*` |
| New 3D prototype before kernel | `phase2_*` modules + small driver changes |
| Shared integrator primitive | Keep `rk4_step` in `phase1.py` until kernel duplicates it deliberately |
| C performance | New files under **`kernel/`** + **`bridge/`**; doc API in this file |
| Kerr / BL coordinates | `phase3_christoffel.py`, `phase3_geodesic.py`, `phase3_render.py` |

---

## Gaps checklist (quick scan for agents)

- [x] `kernel/` generic RK4 + Phase A harmonic parity (`make -C kernel`, pytest).
- [x] Schwarzschild **2D equatorial** `u(\phi)` tracer in C + discrete parity vs `phase1.trace_single_schwarzschild_ray` (`bh_rt_schwarzschild_u_*`, pytest).
- [x] Schwarzschild / Phase **3D Christoffel** geodesics in C + parity vs `phase2_geodesic` Python.
- [x] Populate `bridge/` with PyBind Phase 2 single-ray trace.
- [x] Optional **native per-ray** Phase 2 render path (`Phase2RenderConfig.use_native_phase2`, `--phase2-native`, `phase2_driver --native`).
- [x] SoA batched C trace (`bh_rt_schwarzschild_phase2_batch.c`) + PyBind bridge + Python render wiring (`phase2_batch.py`, `native_phase2.py`).
- [x] GitHub Actions CI (ubuntu-latest: pytest + native build + kernel smoke test).
- [x] Interactive preview CLI (`preview.py`, `--phase2-preview`).
- [x] Accretion disk intersection + Keplerian Doppler redshift (`phase2_disk.py`).
- [x] Kerr / Boyer–Lindquist design doc (`docs/KERR_BOYER_LINDQUIST.md`).
- [x] ML surrogate scaffold: schema, dataset generator, MLP (pure NumPy), runtime gate (`ml/`).
- [x] CUDA batch kernel (`kernel/cuda/`) + Python ctypes bridge (`native_phase2_cuda.py`).
- [x] **Ignore `*.ppm` in `.gitignore`** — render outputs are binary and bloat history; keep them untracked (policy; see repo `.gitignore`).
- [x] Kerr Boyer–Lindquist geodesic integrator (`phase3_christoffel.py`, `phase3_geodesic.py`) + render dispatcher + parity / conservation tests.
- [x] Equatorial crossing output in C batch kernel (`out_eq_r_cross`) → disk detection in native batch path.
- [x] Anti-aliasing supersample (`Phase2RenderConfig.supersample`, `--aa` CLI flag).
- [x] ML end-to-end tests (`tests/test_ml.py`): schema, dataset shapes, surrogate forward, gate routing.
- [x] CUDA standalone build path (`kernel/cuda/Makefile`, `build/.gitkeep`, optional CI step).
