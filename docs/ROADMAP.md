# Roadmap — blackhole_ray_tracer

Canonical phased plan aligned with current implementation and evidence. Update status rows only after qualification checks.

Companion: [OVERVIEW.md](./OVERVIEW.md), [STATUS.md](./STATUS.md), [PROJECT_CONTRACT.md](./PROJECT_CONTRACT.md),
[STATE_API.md](./STATE_API.md), and [IMPLEMENTATION/SPEC.md](./IMPLEMENTATION/SPEC.md).

## Vision and scope

**In scope**

- Reliable null-geodesic integration for Schwarzschild in Python and canonical C, with optional CUDA batch source.
- Clear APIs for ray batches, presets, regression tests, and documented backend boundaries.
- Educational, inspectable phases (`phase1`, `phase2`, `phase3`, …).

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

Teaching notes and execution details remain in [`tasks/plan.md`](../tasks/plan.md).

---

### Phase 2 — Full 3D Schwarzschild reference — qualified

| Item | Status | Modules |
|------|--------|---------|
| Christoffel + RK4 affine null geodesic | Done | `phase2_christoffel.py`, `phase2_geodesic.py` |
| Static observer pinhole initializer | Done | `phase2_camera.py`, `phase2_types.py` |
| PPM shadow + synthetic sky | Done | `phase2_render.py`; `phase1_image.write_ppm_rgb` |
| Disk crossing and Doppler color | Done | `phase2_disk.py`, `phase2_render.py` |
| Presets + benchmark report | Done | `phase2_report.py` |

**Known limits**

- Python remains an accuracy-oriented reference path.
- No anti-aliasing or textured/thick disk model.
- Native CPU batch and optional CUDA batch are separate backend paths.

**Acceptance (maintain)**

- `uv run pytest` passes `tests/test_phase2.py`.
- `--phase2-report` and a small `--phase2-render` complete without crash.


---

### Phase 2+ / kernel bridge — qualified

| Item | Goal | Status |
|------|------|--------|
| **State API** | Flat row-major Phase 2 arrays and `RayStatus` codes | Qualified in [STATE_API.md](./STATE_API.md) |
| **`kernel/` RK4 core** | Shared C RK4 step and Schwarzschild helpers | Implemented and parity-tested |
| **Schwarzschild \(2D equatorial\) kernel** | `u(\phi)=1/r` loop | Implemented and parity-tested |
| **Schwarzschild Phase 2 scalar/batch kernel** | 3D Christoffel trace with disk observables | Canonical implementation; scalar and batch parity-tested |
| **`bridge/`** | PyBind11 CPU scalar/batch boundary | Implemented; local native render and marshalling parity pass |

**Acceptance**

- `make -C kernel clean && make -C kernel all` passes locally.
- C parity covers captured, escaped, max-step, and numerical-error statuses.
- Native batch render is exercised locally.
- CUDA remains optional and unavailable locally; see [STATUS.md](./STATUS.md).

---

### Phase 3 — Kerr / Boyer–Lindquist prototype — validated Python reference

| Item | Status | Modules |
|------|--------|---------|
| Kerr metric + Christoffel symbols | Tested | `phase3_christoffel.py` |
| Kerr null RK4 trace loop | Tested | `phase3_geodesic.py` |
| ZAMO/local orthonormal pinhole camera | Tested | `phase3_camera.py`, `phase3_render.py` |
| Invariants, horizon, pole, spin, and convergence checks | Tested | `tests/test_phase3.py` |
| Pinhole render, disk crossing, and report | Smoke-tested | `phase3_types.py`, `phase3_render.py`, `phase3_report.py` |

**Boundary**

- Kerr disk color and rendering remain Python-only.
- T10 added C metric/RHS algebra parity.
- T12 added bounded C single-ray termination parity.
- T14 added C ZAMO seed parity.
- T16 added C first strict equatorial crossing parity.
- Future native work still requires disk color/rendering, batch, bridge, CUDA, and performance tickets.
- Pure Python rendering is accuracy-oriented, not real-time or production-quality.

### Phase 3+ — Kerr native CPU foundation

| Item | Goal | Status |
|------|------|--------|
| C BL metric | Match `phase3_christoffel.kerr_metric_matrix` | Tested for spin zero, signed moderate spin, and near-extreme spin |
| C first-order RHS | Match `null_geodesic_kerr_first_order` | Tested through ctypes |
| C single-ray tracer | Match Python termination and scalar result fields | Tested through ctypes, no samples or rendering |
| C ZAMO seed | Match `phase3_camera.zamo_null_direction` | Tested through ctypes, finite and null |
| C equatorial crossing | Match first strict Python interpolation | Tested through ctypes, geometry-only |
| C disk observable policy | Match Python radius hit and Doppler intensity | Tested through ctypes; geometry-independent |
| Error boundary | Reject invalid controls and expose numerical terminal status | Explicit integer return codes tested |

**Boundary:** No native Kerr color renderer, batch, bridge, CUDA, or performance claim exists.

---

### Phase 3+ — ML warm starter — qualified experimental backend

| Item | Goal | Status |
|------|------|--------|
| **`ml/`** | Generate reproducible tuples from verified Phase 2 output | Metadata schema and validation tested |
| Surrogate | Predict outcomes under explicit budget | Deterministic holdout measured; accuracy remains experimental |
| Policy | Route only explicit ML requests with RK4 fallback | Strict model/output validation tested |
| Artifact | Package reference weights deliberately | `ml/models/surrogate.npz` packaged with metadata |

**Acceptance**

- Dataset metadata records coordinate convention, status mapping, sentinel policy, and integrator parameters.
- Model loader rejects missing, corrupt, malformed, nonfinite, or schema-incompatible artifacts.
- Runtime gate uses requested `max_steps` and falls back on invalid output.
- Default renders use trusted Python or native integration.

---

### Phase 4 — Optional CUDA batch backend — qualified experimental path

| Item | Goal | Status |
|------|------|--------|
| CUDA batch kernel | Match CPU Phase 2 batch semantics | Source present; host error handling hardened |
| Python ctypes wrapper | Validate inputs and expose explicit errors | Tested without CUDA library |
| Preview routing | Prefer CUDA, then CPU native, then Python | Headless routing smoke passes |
| Benchmark | Record accuracy-oriented throughput evidence | CPU native evidence recorded; GPU unavailable locally |

**Acceptance**

- CUDA parity runs when compatible `nvcc` and library exist.
- Missing `nvcc` reports a precise unavailable diagnostic.
- No CUDA availability or frame-rate claim is made without hardware evidence.

## CLI cheat sheet

### `blackhole-ray-tracer` (`src/blackhole_ray_tracer/main.py`)

| Flag | Effect |
|------|--------|
| `--phase1-ab` | Step A + B summary |
| `--phase1-step-b` | Step B log |
| `--phase1-step-d` | Step D table |
| `--phase1-step-e` | Step E PPM (`--ppm-out`, `--img-width`, `--img-height`, `--preset`) |
| `--phase1-step-f` | Step F report |
| `--phase2-render` | Phase 2 PPM (`--phase2-out`, `--phase2-preset`, image size flags; `--phase2-native`, `--phase2-cuda`, `--phase2-disk`, `--phase2-ml-gate`) |
| `--phase2-report` | Phase 2 presets and single-ray benchmark |
| `--phase3-report` | Kerr presets and single-ray benchmark |
| `--phase3-render` | Kerr PPM (`--phase3-out`, `--phase3-preset`, `--spin`, `--phase3-disk`) |

### `python -m blackhole_ray_tracer.phase1_driver`

| `--step` | Purpose |
|-----------|---------|
| `a`, `b`, `ab`, `d`, `e`, `f` | As module docstring; extra flags for CSV, presets, \(\phi\) step, \(r_\text{escape}\), etc. |

### `python -m blackhole_ray_tracer.phase2_driver`

| Flag | Purpose |
|------|---------|
| `--report` | Presets + `dlambda` benchmark |
| `--render` | 3D PPM (`--out`, `--preset` or `--width`/`--height`/`--dlambda`/…) |
| `--native` | Use installed C extension; render selects native batch when available |
| `--disk`, `--ml-gate` | Enable disk overlay, or route rays through the ML runtime gate |

### `python -m blackhole_ray_tracer.phase3_driver`

| Flag | Purpose |
|------|---------|
| `--report` | Kerr presets + single-ray benchmark |
| `--render` | Kerr PPM (`--out`, `--preset`, `--spin`, `--disk`, or explicit render parameters) |

Installation: typically `PYTHONPATH=src` when developing from checkout without reinstall.

---

## Where to extend code

| Change | Prefer |
|--------|--------|
| New Schwarzschild 2D experiment | `phase1.py` helpers or `phase1_*` |
| New 3D prototype before kernel | `phase2_*` modules + small driver changes |
| Shared integrator primitive | Keep `rk4_step` in `phase1.py` until kernel duplicates it deliberately |
| C performance | New files under **`kernel/`** + **`bridge/`**; doc API in this file |
| Kerr / BL coordinates | `phase3_*` with Boyer–Lindquist names; metric/RHS C probe allowed after T06; tracer remains ticketed |

---

## Gaps checklist (quick scan for agents)

- [x] `kernel/` generic RK4 + Phase A harmonic parity (`make -C kernel`, pytest).
- [x] Schwarzschild **2D equatorial** `u(\phi)` tracer in C + discrete parity vs `phase1.trace_single_schwarzschild_ray` (`bh_rt_schwarzschild_u_*`, pytest).
- [x] Schwarzschild / Phase **3D Christoffel** geodesics in C + parity vs `phase2_geodesic` Python.
- [x] Populate `bridge/` with PyBind Phase 2 scalar and batch traces.
- [x] Optional native Phase 2 render path (`Phase2RenderConfig.use_native_phase2`, `--phase2-native`, `phase2_driver --native`) with batch-first routing.
- [x] SoA batched C trace (`bh_rt_schwarzschild_phase2_batch.c`) + PyBind bridge + Python render wiring (`phase2_batch.py`, `native_phase2.py`).
- [x] GitHub Actions CI (ubuntu-latest: pytest + native build + kernel smoke test).
- [x] Interactive preview CLI (`preview.py`, `--phase2-preview`).
- [x] Accretion disk intersection + Keplerian Doppler redshift (`phase2_disk.py`).
- [x] Kerr / Boyer–Lindquist design doc (`docs/KERR_BOYER_LINDQUIST.md`).
- [x] Kerr Python prototype: Christoffel symbols, RK4 geodesic trace, render config/driver, and `tests/test_phase3.py` (`phase3_*`).
- [x] Kerr C BL metric/RHS algebra, bounded single-ray termination, ZAMO seed, and equatorial crossing parity (`bh_rt_kerr_phase3.*`, `tests/test_kernel_phase3_algebra.py`, `tests/test_kernel_phase3_trace.py`, `tests/test_kernel_phase3_camera.py`, `tests/test_kernel_phase3_crossing.py`).
- [x] ML experimental backend: schema, dataset generator, NumPy MLP, runtime gate, and packaged model.
- [x] Phase 2 ML render gate wiring (`Phase2RenderConfig.ml_gate_model_path`, `--phase2-ml-gate`, `phase2_driver --ml-gate`) with RK4 fallback.
- [x] CUDA batch kernel (`kernel/cuda/`) + Python ctypes bridge (`native_phase2_cuda.py`).
- [x] ZAMO Kerr camera and invariant/convergence validation (`phase3_camera.py`, `tests/test_phase3.py`).
- [x] ML metadata, strict loader, holdout check, explicit fallback, and packaged reference model.
- [x] CUDA host error handling, unavailable-toolchain diagnostic, and headless preview routing.
- [x] **Ignore `*.ppm` in `.gitignore`** — render outputs are binary and bloat history; keep them untracked (policy; see repo `.gitignore`).
