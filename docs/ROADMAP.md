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

- Phase 2 is **slow** per pixel (pure Python); no **C kernel** path yet for `phase2` rays.
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
| **`bridge/`** | pybind11 exposing batch trace | Not started |

**Acceptance**

- `make -C kernel` builds **`harmonic_demo`** and **`schwarzschild_demo`** where `cc`/`gcc`/`clang` exists.
- Full `pytest` run passes (kernel parity tests skip without a toolchain or when `SKIP_KERNEL_TESTS=1`).
- 3D Schwarzschild (Christoffel) single-ray kernel path exercises RK4 parity vs Python Phase 2; **`bridge/`** build remains **TBD**.

---

### Phase 3 — ML warm starter — planned

| Item | Goal |
|------|------|
| **`ml/`** | Generate training tuples from verified integrator outputs |
| Surrogate | Model predicts outcomes or endpoint state under budget |
| Policy | Optionally switch kernel ↔ surrogate with safety checks |

**Acceptance**

- Documented dataset format + minimal training script; tests optional but preferred.

---

### Phase 4 — GPU port — planned

| Item | Goal |
|------|------|
| CUDA (or METAL/other) batch geodesics | Preview-oriented throughput |
| Shared API with CPU kernel | Same inputs/outputs as bridge |

**Acceptance**

- Target metric documented (e.g. rays/s or FPS at fixed resolution TBD).

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
| `--phase2-render` | Phase 2 PPM (`--phase2-out`, `--phase2-preset`, image size flags) |

### `python -m blackhole_ray_tracer.phase1_driver`

| `--step` | Purpose |
|-----------|---------|
| `a`, `b`, `ab`, `d`, `e`, `f` | As module docstring; extra flags for CSV, presets, \(\phi\) step, \(r_\text{escape}\), etc. |

### `python -m blackhole_ray_tracer.phase2_driver`

| Flag | Purpose |
|------|---------|
| `--report` | Presets + `dlambda` benchmark |
| `--render` | 3D PPM (`--out`, `--preset` or `--width`/`--height`/`--dlambda`/…) |

Installation: typically `PYTHONPATH=src` when developing from checkout without reinstall.

---

## Where to extend code

| Change | Prefer |
|--------|--------|
| New Schwarzschild 2D experiment | `phase1.py` helpers or `phase1_*` |
| New 3D prototype before kernel | `phase2_*` modules + small driver changes |
| Shared integrator primitive | Keep `rk4_step` in `phase1.py` until kernel duplicates it deliberately |
| C performance | New files under **`kernel/`** + **`bridge/`**; doc API in this file |
| Kerr / BL coordinates | Future `phase3_*` or `kerr_*` naming TBD |

---

## Gaps checklist (quick scan for agents)

- [x] `kernel/` generic RK4 + Phase A harmonic parity (`make -C kernel`, pytest).
- [x] Schwarzschild **2D equatorial** `u(\phi)` tracer in C + discrete parity vs `phase1.trace_single_schwarzschild_ray` (`bh_rt_schwarzschild_u_*`, pytest).
- [x] Schwarzschild / Phase **3D Christoffel** geodesics in C + parity vs `phase2_geodesic` Python.
- [ ] Populate `bridge/` and wire optional import from Python render path.
- [x] **Ignore `*.ppm` in `.gitignore`** — render outputs are binary and bloat history; keep them untracked (policy; see repo `.gitignore`).
- [ ] Kerr: coordinate choice (BL) documented before implementation.
