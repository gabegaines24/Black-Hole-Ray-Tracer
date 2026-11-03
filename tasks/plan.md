# Implementation Plan: blackhole_ray_tracer completion

## Problem

Project has qualified Phase 1/2 reference paths, canonical C/native CPU paths, a validated Python Kerr prototype,
and bounded ML/CUDA experiments. T00–T18 execution is complete; future work remains explicitly bounded.

Initial baseline findings below explain why this plan exists. T01–T18 resolved reachable blockers;
the final matrix records remaining hardware and future-scope boundaries.

## Decision

Finish correctness and reproducibility before throughput. Keep Python as reference oracle. Keep Phase 1, Phase 2, Phase 3, `kernel/`, `bridge/`, and `ml/` boundaries. Treat ML and CUDA as opt-in experimental backends until they pass explicit parity and accuracy gates.

Do not call project complete when only Python tests pass. Completion requires verified Python behavior, native build/parity, coherent docs, and a reproducible package/CI path.

## Scope assumptions

1. Phase 1 remains an educational equatorial Schwarzschild baseline.
2. Phase 2 remains the supported Schwarzschild 3D renderer and reference backend.
3. Phase 3 remains Python-first Kerr in Boyer–Lindquist coordinates.
4. Kerr native path now has tested metric/RHS and single-ray termination; camera, disk, bridge, batch, and CUDA work stay future.
5. Current uncommitted changes belong to an active worktree. Never reset or overwrite them during implementation.

## Initial evidence snapshot (before T01–T08)

Observed in current worktree:

| Check | Result | Meaning |
|---|---|---|
| `uv run pytest -q` | 51 passed, 7 skipped | Python suite passes; optional backends are not all exercised |
| `uv run ruff check src tests` | Pass | Current lint surface is clean |
| `uv run mypy src` | Fail, 25 diagnostics | Type contract is incomplete |
| `make -C kernel all` | Pass | Standalone C demos build |
| Kernel demos | Pass | Harmonic and batch smoke output produced |
| `uv sync --extra dev --dry-run` | Fail | CI command is invalid |
| `uv sync --group dev --dry-run` | Pass | Correct local command |
| `uv lock --check` | Pass | Lockfile is internally consistent |
| Native extension status | unavailable | Bridge parity tests skip |
| `BLACKHOLE_BUILD_NATIVE=1 uv pip install -e . --reinstall` | Fail | macOS C/C++ extension build is broken |
| `uv build --wheel --sdist` | wheel fail | `setup.py` uses absolute extension source paths; sdist omits CUDA/model data |

Working tree also has 28 modified files and 2 untracked paths. Plan work must preserve that state and separate existing edits from new fixes.

## Current architecture map

```text
CLI
├── src/blackhole_ray_tracer/main.py
├── phase1_driver.py
├── phase2_driver.py
├── phase3_driver.py
└── preview.py

Python reference
├── phase1.py                         shared Python RK4 + equatorial ray
├── phase2_christoffel.py             Schwarzschild Christoffel RHS
├── phase2_geodesic.py                3D Schwarzschild trace
├── phase2_camera.py                  static-observer pinhole initializer
├── phase2_render.py                  image loop and backend routing
├── phase2_disk.py                    disk crossing and color model
└── phase3_*.py                       Kerr Boyer–Lindquist prototype

Native path
├── kernel/src/                       C RK4, Schwarzschild tracers, and Kerr metric/RHS/tracer
├── kernel/cuda/                      optional CUDA batch kernel
├── bridge/module_phase2.cpp          PyBind11 CPU bridge
├── native_phase2.py                  Python CPU façade
└── native_phase2_cuda.py             ctypes CUDA façade

ML experiment
├── ml/dataset.py                     reference-trace dataset generation
├── ml/schema.py                      feature/label contract
├── ml/surrogate.py                   NumPy MLP
├── ml/runtime_gate.py                surrogate/RK4 routing
└── ml/models/surrogate.npz           packaged reference artifact
```

## Authority hierarchy to adopt

1. A new `docs/PROJECT_CONTRACT.md` defines state layout, units, statuses, termination, backend equivalence, and supported boundaries.
2. `docs/STATE_API.md` defines the C/Python FFI shapes and dtypes.
3. Accepted ADRs define choices that are not obvious from equations or code.
4. `docs/FINAL_ARCHITECTURE.md` summarizes stable module boundaries.
5. `docs/STATUS.md` records evidence and unverified claims.
6. `docs/IMPLEMENTATION/SPEC.md` and ticket files define execution order.
7. README and comments explain usage, but cannot override the contract.

Use Chameleon’s planning discipline, not its domain contract: explicit authority, gates, vertical tickets, evidence logs, and open external questions.

## Dependency graph

```mermaid
flowchart TD
    T00[Contract, baseline, status] --> T01[Build, packaging, CI]
    T00 --> T02[Canonical state and Python types]
    T01 --> T03[Canonical Schwarzschild kernel]
    T02 --> T03
    T03 --> T04[Safe bridge and batch parity]
    T04 --> T05[Disk observables and render parity]
    T02 --> T06[Kerr camera and validation]
    T03 --> T07[ML gate contract]
    T06 --> T07
    T04 --> T08[CUDA and performance evidence]
    T05 --> T08
    T01 --> T09[Release docs and qualification]
    T05 --> T09
    T06 --> T09
    T07 --> T09
    T08 --> T09
    T09 --> T10[Kerr native algebra probe]
    T10 --> T12[Kerr native single-ray termination]
    T12 --> T14[Kerr native ZAMO camera]
    T14 --> T16[Kerr native equatorial crossing]
    T16 --> T18[Kerr native disk policy]

## Tickets

### T00 — Freeze baseline and ratify project contract

**Description:** Record current behavior and define the supported numerical/API contract before changing implementations. Mark every claim as implemented, tested, skipped, or planned.

**Acceptance criteria:**

- [x] `docs/PROJECT_CONTRACT.md` defines geometric units, Schwarzschild radius, Boyer–Lindquist naming, state order `(t, r, theta, phi, vt, vr, vtheta, vphi)`, status meanings, and termination precedence.
- [x] Contract distinguishes Python reference behavior, CPU native behavior, CUDA behavior, and experimental ML behavior.
- [x] `docs/STATUS.md` records baseline results and lists skipped native/CUDA checks as evidence gaps.
- [x] Pending worktree edits are classified; no current edit is silently discarded.

**Verification:**

- [x] Cross-link audit finds one authority for each API rule.
- [x] Baseline commands were recorded before implementation.

**Dependencies:** None.

**Files likely touched:** `docs/PROJECT_CONTRACT.md`, `docs/STATUS.md`, `docs/FINAL_ARCHITECTURE.md`, `docs/IMPLEMENTATION/README.md`.

**Estimated scope:** Medium.

### T01 — Repair build, packaging, and CI gates

**Description:** Make clean installs and wheel builds work on supported Python/compiler combinations. Align CI commands with `pyproject.toml`.

**Acceptance criteria:**

- [x] Replace CI `--extra dev` with `--group dev`; align matrix with `requires-python >= 3.12`.
- [x] Build extension source paths relative to project root, not absolute paths.
- [x] Compile C sources with C flags and C++ bridge sources with C++ flags.
- [x] `uv build --wheel --sdist` succeeds; package contents and optional model policy are explicit.
- [x] CI runs pytest, Ruff, mypy, kernel build, and native bridge smoke tests.

**Verification:**

- [x] `uv sync --group dev` dry-run passes in the current environment.
- [x] `uv run pytest -q` passes.
- [x] `uv run ruff check src tests ml` passes.
- [x] `uv run mypy src ml` passes.
- [x] Native extension builds and imports on macOS.
- [x] `uv build --wheel --sdist` passes with package-data inspection.

**Dependencies:** T00.

**Files likely touched:** `setup.py`, `pyproject.toml`, `MANIFEST.in`, `.github/workflows/ci.yml`.

**Estimated scope:** Medium.

### T02 — Canonicalize state, result, config, and validation types

**Description:** Remove Python type drift before backend work. Keep one disk configuration type and one explicit render/result contract.

**Acceptance criteria:**

- [x] Define `DiskConfig` once and use one render-stats/result contract.
- [x] Validate positive dimensions, finite physical parameters, valid step budgets, observer location, FOV, spin bounds, and array shapes.
- [x] Keep endpoint state and crossing velocity aligned in `GeodesicTraceResult`.
- [x] `mypy src ml` has no project-owned type errors.

**Verification:**

- [x] Boundary tests cover invalid dimensions, horizon/pole observers, negative steps, invalid spin, malformed `(N, 8)` arrays, and duplicate config types.
- [x] Full pytest, Ruff, and mypy pass.

**Dependencies:** T00.

**Files likely touched:** `phase2_types.py`, `phase2_disk.py`, `phase2_geodesic.py`, `phase3_types.py`, `phase2_render.py`.

**Estimated scope:** Medium.

### T03 — Establish one canonical Schwarzschild C implementation

**Description:** Eliminate the two overlapping 3D C paths. Keep one implementation as the parity oracle and migrate demos/tests/build rules to it.

**Acceptance criteria:**

- [x] Choose Phase 2 scalar/batch implementation as canonical.
- [x] Remove duplicate Christoffel, RK4-loop, termination, and renormalization logic.
- [x] Use fixed-width integer types where the FFI contract promises `int32`.
- [x] Preserve Python termination precedence and RK4 projection cadence.
- [x] Kernel demos and tests exercise only canonical code.

**Verification:**

- [x] `make -C kernel clean && make -C kernel all`.
- [x] C parity tests cover captured, escaped, max-step, and numerical-error paths.
- [x] Python scalar, C scalar, C batch, and final states match across tested rays.

**Dependencies:** T01, T02.

**Files likely touched:** `kernel/include/bh_rt_schwarzschild_phase2*.h`, `kernel/include/bh_rt_status.h`, `kernel/src/bh_rt_schwarzschild_phase2*.c`, `kernel/Makefile`, and kernel parity tests.

**Estimated scope:** Large; split implementation into scalar and batch substeps if needed.

### T04 — Harden PyBind11 bridge and native batch path

**Description:** Make FFI marshalling safe and prove installed native rendering matches the Python reference.

**Acceptance criteria:**

- [x] Single-ray inputs use contiguous float64 buffers; malformed inputs fail or normalize safely.
- [x] Batch inputs are C-contiguous `(N, 8)` float64 with bounded `N`; empty and oversized batches have defined behavior.
- [x] Output arrays are explicitly `int32` or `float64` as promised by `docs/STATE_API.md`.
- [x] Long native calls release the GIL.
- [x] Native scalar and batch paths match Python status and observable contracts.
- [x] Bridge tests run when the compiler is available and skip only when native build is unavailable.

**Verification:**

- [x] Build the extension in the current environment.
- [x] Run bridge parity and native render tests.
- [x] Run non-contiguous, dtype, empty-batch, and large-batch boundary tests.

**Dependencies:** T01, T02, T03.

**Files likely touched:** `bridge/module_phase2.cpp`, `src/blackhole_ray_tracer/native_phase2.py`, `tests/test_bridge_native_phase2.py`, `tests/test_phase2.py`.

**Estimated scope:** Medium.

### T05 — Make disk observables physically and backend correct

**Description:** Fix equatorial crossing interpolation and use velocity at crossing, not initial velocity. Keep disk coloring outside geodesic integration.

**Acceptance criteria:**

- [x] Python trace exposes consecutive `(theta, r, vt, vphi)` samples.
- [x] CPU and CUDA batch paths detect first sign-changing crossing using pre-step and post-step states.
- [x] Python, C, and CUDA use the same linear interpolation formula.
- [x] Disk tests use guaranteed crossings and fail on missed crossing behavior.
- [x] Doppler and color outputs are finite and clamped; approximation limits are documented.

**Verification:**

- [x] Deterministic disk crossing tests compare Python/native observables.
- [x] Small Python and native disk renders exercise hit masks and colors.
- [x] Capture-before-crossing, no-crossing, boundary, and multiple-crossing tests pass.

**Dependencies:** T02, T03, T04.

**Files likely touched:** `phase2_geodesic.py`, `phase2_disk.py`, `phase2_render.py`, `kernel/src/bh_rt_schwarzschild_phase2_batch.c`, `tests/test_phase2.py`.

**Estimated scope:** Medium.

### T06 — Validate Kerr prototype before any Kerr port

**Description:** Strengthen Phase 3 from a smoke prototype into a numerically justified Python reference. Keep C/Kerr port out of scope until these gates pass.

**Acceptance criteria:**

- [x] Add a Kerr-compatible ZAMO/local orthonormal tetrad.
- [x] Validate null initialization, horizon termination, spin range, pole behavior, and near-extreme spin.
- [x] Test spin-zero parity, `E`/`L`, off-equatorial motion, and Carter drift.
- [x] Add step-size convergence and Schwarzschild limiting-case checks.
- [x] Phase 3 disk crossing uses crossing-state velocity and remains Python-only.

**Verification:**

- [x] Run Phase 3 tests for zero, moderate positive/negative, and near-extreme spin.
- [x] Compare coarse/fine `dlambda` trajectories and invariant drift.
- [x] Run tiny render/report smoke without production-quality claims.

**Dependencies:** T02, T05.

**Files likely touched:** `phase3_camera.py` (new), `phase3_christoffel.py`, `phase3_geodesic.py`, `phase3_render.py`, `tests/test_phase3.py`.

**Estimated scope:** Large; split camera and invariant work if needed.

### T07 — Bound ML as an experimental, reproducible backend

**Description:** Make surrogate use explicit, validated, and unable to silently replace the trusted integrator on malformed or unverified outputs.

**Acceptance criteria:**

- [x] Dataset schema records integrator parameters, coordinate convention, status mapping, sentinel policy, and model version.
- [x] Model loader validates presence, arrays, shapes, dtype, finite weights, and schema/version metadata.
- [x] Runtime gate passes requested `max_steps`, validates predictions, and falls back on errors.
- [x] Default renders use Python/native integration; ML remains explicit opt-in.
- [x] Reference weights are packaged deliberately with deterministic regeneration instructions.
- [x] Held-out surrogate accuracy is measured before any quality-sensitive route.

**Verification:**

- [x] Test missing, corrupt, shape-invalid, and finite-but-inaccurate models.
- [x] Generate deterministic train/holdout data with fixed seed.
- [x] Report status accuracy and finite prediction checks.
- [x] Run ML-gated fallback behavior tests.

**Dependencies:** T02, T06.

**Files likely touched:** `ml/schema.py`, `ml/dataset.py`, `ml/runtime_gate.py`, `ml/surrogate.py`, `tests/test_ml.py`.

**Estimated scope:** Medium.

### T08 — Qualify CUDA, throughput, and preview behavior

**Description:** Keep GPU support optional but make its contract honest and measurable. Do not promise frame rates without a measured target.

**Acceptance criteria:**

- [x] CUDA host wrapper checks allocation, copy, launch, synchronization, and free errors.
- [x] CUDA output dtypes, statuses, termination values, and disk observables use the CPU contract.
- [x] CUDA build path reports a precise unavailable reason without `nvcc`.
- [x] Benchmark records rays/s, image size, step settings, backend, and hardware.
- [x] Preview flags propagate native/CUDA/disk choices with a headless smoke path.

**Verification:**

- [x] CPU Python and native parity run on fixed tests; CUDA parity remains skipped without library.
- [x] `make -C kernel cuda` reports `nvcc not found (nvcc)` locally.
- [x] Headless preview smoke passes.
- [x] Benchmark output is recorded without treating unavailable GPU hardware as failure.

**Dependencies:** T04, T05.

**Files likely touched:** `kernel/cuda/bh_rt_schwarzschild_phase2_batch.cu`, `src/blackhole_ray_tracer/native_phase2_cuda.py`, `src/blackhole_ray_tracer/preview.py`, `kernel/Makefile`, `tests/test_cuda_batch_parity.py`.

**Estimated scope:** Large.

### T09 — Reconcile documentation and publish qualification evidence

**Description:** Make documentation match code and evidence, using Chameleon’s status/contract/ticket/decision pattern.

**Acceptance criteria:**

- [x] Add `docs/IMPLEMENTATION/README.md`, `docs/IMPLEMENTATION/SPEC.md`, and dependency-sequenced ticket files.
- [x] Add `docs/DECISIONS/` ADRs for canonical C API, observer tetrad, disk observable semantics, ML status, and CUDA acceptance.
- [x] Update README, overview, roadmap, Kerr, FFI, bridge, persistent rules, and session prompt.
- [x] Record final tests, skips, native build, package build, kernel smoke, CLI smoke, and benchmark evidence in `docs/STATUS.md`.
- [x] Keep PPM, cache, binary, and model-artifact policies explicit.

**Verification:**

- [x] Markdown link/path audit finds all linked targets.
- [x] Run the final verification matrix below.
- [x] Review unchecked or planned claims against actual files.

**Dependencies:** T01, T05, T06, T07, T08.

**Files likely touched:** `docs/OVERVIEW.md`, `docs/ROADMAP.md`, `docs/STATE_API.md`, `docs/KERR_BOYER_LINDQUIST.md`, `README.md`, `.cursorrules`, `AGENTS.md`, new `docs/STATUS.md` and decision/spec files.

**Estimated scope:** Large documentation change; keep behavior changes out of this ticket.

### T10 — Incremental Kerr native-port foundation

**Description:** Port only Kerr BL metric and first-order geodesic RHS evaluation.
Keep Python as reference until algebra parity passes.

**Acceptance criteria:**

- [x] C ABI preserves coordinate/state order and dimensionless `spin = a/M`.
- [x] C metric output matches `phase3_christoffel.kerr_metric_matrix`.
- [x] C RHS output matches `null_geodesic_kerr_first_order` for exterior off-pole states.
- [x] Invalid, non-finite, horizon, and pole inputs return explicit error codes.
- [x] No renderer, batch tracer, PyBind11 bridge, or CUDA claim is added in this slice.

**Verification:**

- [x] `uv run pytest -q tests/test_kernel_phase3_algebra.py tests/test_kernel_phase3_trace.py`.
- [x] `uv run ruff check src tests ml`.
- [x] `uv run mypy src ml`.

**Dependencies:** T06, T09.

**Files:** `kernel/include/bh_rt_kerr_phase3.h`, `kernel/src/bh_rt_kerr_phase3.c`, `tests/test_kernel_phase3_algebra.py`.

**Estimated scope:** Small foundation slice.

### T12 — Bounded Kerr native single-ray tracer

**Description:** Extend the T10 Kerr C module with one bounded RK4 tracer and shared termination result.
Keep camera, disk, bridge, batch, CUDA, and rendering out of scope.

**Acceptance criteria:**

- [x] Preserve the eight-state order and dimensionless `spin = a/M`.
- [x] Mirror Python pre-step capture/escape checks and periodic null-cone projection.
- [x] Return shared `BH_RT_STATUS_*` values and scalar termination fields.
- [x] Reject invalid control parameters with explicit `BH_RT_KERR_*` errors.
- [x] Match Python terminal status, step count, termination values, and minimum radius.
- [x] Keep non-finite initial state behavior explicit as numerical-error status.

**Verification:**

- [x] `uv run pytest -q tests/test_kernel_phase3_algebra.py tests/test_kernel_phase3_trace.py`.
- [x] C99 compile with `-Wall -Wextra -Werror`.
- [x] Full post-T12 suite, lint, type, package, and documentation audit.

**Dependencies:** T10.

**Files:** `kernel/include/bh_rt_kerr_phase3.h`, `kernel/src/bh_rt_kerr_phase3.c`, `tests/test_kernel_phase3_trace.py`.

**Estimated scope:** Small vertical slice.

### T14 — Kerr native ZAMO camera initializer

**Description:** Port one validated Python ZAMO camera seed into the C Kerr module.
Keep disk, batch, bridge, CUDA, and rendering out of scope.

**Acceptance criteria:**

- [x] Preserve normalized screen mapping, aspect ratio, FOV, lapse, and frame dragging.
- [x] Return the complete eight-state `(t, r, theta, phi, vt, vr, vtheta, vphi)` seed.
- [x] Match Python camera seeds for zero, signed moderate, and near-extreme spin.
- [x] Reject invalid pointers, controls, poles, horizon, FOV, and dimensions explicitly.
- [x] Confirm generated states are finite and null.

**Verification:**

- [x] `uv run pytest -q tests/test_kernel_phase3_camera.py`.
- [x] C99 compile with `-Wall -Wextra -Werror`.
- [x] Full post-T14 suite, lint, type, package, and documentation audit.

**Dependencies:** T12.

**Files:** `kernel/include/bh_rt_kerr_phase3.h`, `kernel/src/bh_rt_kerr_phase3.c`, `tests/test_kernel_phase3_camera.py`.

**Estimated scope:** Small vertical slice.

### T16 — Kerr native equatorial crossing observables

**Description:** Add a separate C trace entrypoint for the first strict equatorial crossing.
Preserve the T12 trace result ABI and keep disk policy/rendering out of scope.

**Acceptance criteria:**

- [x] Preserve the existing scalar trace result layout.
- [x] Return first strict `theta = pi/2` crossing only.
- [x] Interpolate `r`, `v^t`, and `v^phi` with Python's alpha semantics.
- [x] Return NaN crossing fields when no crossing exists.
- [x] Reject missing crossing output explicitly.

**Verification:**

- [x] `uv run pytest -q tests/test_kernel_phase3_crossing.py`.
- [x] C99 compile with `-Wall -Wextra -Werror`.
- [x] Full post-T16 suite, lint, type, package, and documentation audit.

**Dependencies:** T14.

**Files:** `kernel/include/bh_rt_kerr_phase3.h`, `kernel/src/bh_rt_kerr_phase3.c`, `tests/test_kernel_phase3_crossing.py`.

**Estimated scope:** Small vertical slice.

### T18 — Kerr native disk hit and Doppler policy

**Description:** Add geometry-independent C disk observables after the T16 crossing API.
Match Python radius gating and Doppler intensity without adding color, rendering, or backend ownership.

**Acceptance criteria:**

- [x] Preserve T16 crossing and T12 trace ABIs.
- [x] Return inclusive radius hit policy and Python-compatible no-crossing sentinel.
- [x] Match Doppler intensity fallback, sign handling, scale handling, and clamp.
- [x] Reject invalid disk controls and missing output explicitly.
- [x] Keep ISCO calculation, color, rendering, batch, bridge, and CUDA out of scope.

**Verification:**

- [x] `uv run pytest -q tests/test_kernel_phase3_disk.py`.
- [x] Full post-T18 suite, lint, type, package, kernel, and documentation audit.

**Dependencies:** T16.

**Files:** `kernel/include/bh_rt_kerr_phase3.h`, `kernel/src/bh_rt_kerr_phase3.c`, `tests/test_kernel_phase3_disk.py`.

**Estimated scope:** Small vertical slice.




## Checkpoints

### Checkpoint A — contract and toolchain

- [x] T00 complete.
- [x] Clean dev install succeeds with `uv sync --group dev`.
- [x] Wheel and sdist build succeeds.
- [x] CI command uses supported `uv` syntax.
- [x] Ruff and mypy pass.

### Checkpoint B — trusted Schwarzschild path

- [x] T02–T04 complete.
- [x] Python, C scalar, C batch, and bridge parity pass.
- [x] No duplicate 3D C implementation remains.
- [x] Native render is verified, not skipped.

### Checkpoint C — observables and Kerr

- [x] T05–T06 complete.
- [x] Disk crossing tests prove hit and miss cases.
- [x] Kerr invariants and limiting cases pass.
- [x] No Kerr C port begins before this checkpoint.

### Checkpoint D — optional backends

- [x] T07–T08 complete; ML/CUDA boundaries remain explicit.
- [x] ML never silently replaces trusted integration.
- [x] CUDA parity and benchmark evidence are recorded when hardware exists.

### Checkpoint I — native Kerr observables

- [x] T10, T12, T14, T16, and T18 native Kerr slices have focused parity evidence.
- [x] Native disk hit and Doppler policy matches Python before color or render work.

### Final completion gate

- [x] `uv run pytest -q` passes with only intentional, documented skips.
- [x] `uv run ruff check src tests ml` passes.
- [x] `uv run mypy src ml` passes with no project-owned diagnostics.
- [x] `make -C kernel all` and kernel smoke demos pass.
- [x] Native extension builds locally; CI contains the supported native smoke gate.
- [x] `uv build --wheel --sdist` succeeds and contains required package data.
- [x] CLI help, Phase 1, Phase 2 report/render, Phase 3 report/render, and preview smoke paths work.
- [x] Documentation status matches observed evidence.

## Required versus optional scope

### Required for a credible current release

- T00, T01, T02, T03, T04, T05, T06, and T09.
- ML and CUDA are either completed under T07/T08 or explicitly labeled experimental/unavailable everywhere.
- No claim of real-time or production-quality rendering for pure Python.

### Optional future work

- Kerr disk color/rendering, batch API, bridge, and CUDA kernels.
- Adaptive RK methods or symplectic integrators after profiling and error studies.
- Radiative transfer, textured/thick disks, lensing backgrounds, and physically calibrated color.
- Metal backend or other GPU backend.
- High-resolution preview optimization and multi-GPU batching.
- Surrogate use in quality-sensitive paths without a measured accuracy budget.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Two C implementations drift | False parity confidence | Select one canonical implementation and delete duplicate hot-path math |
| Native/CUDA tests remain skipped | Unsupported backend claims | Make native build a CI gate; keep CUDA optional but report explicit availability |
| Coordinate singularities corrupt rays | Silent physics errors | Validate observer/range inputs and terminate before horizon/pole singularities |
| Surrogate predicts plausible but wrong statuses | Wrong images with no exception | Opt-in gate, model validation, holdout accuracy, strict fallback |
| Pending worktree changes conflict | User work lost | Never reset; classify and integrate existing edits first |
| Documentation overstates completion | Wrong downstream work | Maintain `STATUS.md` evidence and update roadmap only after gates pass |

## Resolved decisions and residual boundaries

1. Canonical C API: [ADR-001](../docs/DECISIONS/ADR-001-canonical-phase2-c-api.md) selects the Phase 2 scalar/batch implementation.
2. Kerr observer frame: [ADR-002](../docs/DECISIONS/ADR-002-kerr-zamo-camera.md) selects a ZAMO/local orthonormal frame.
3. Reference model policy: [ADR-004](../docs/DECISIONS/ADR-004-ml-experimental-boundary.md) keeps ML opt-in with strict fallback.
4. CUDA target: [ADR-005](../docs/DECISIONS/ADR-005-cuda-acceptance.md) rejects unmeasured frame-rate guarantees.
5. Supported Python versions: `>=3.12`, enforced by project metadata and CI.
6. Kerr native termination boundary: [ADR-007](../docs/DECISIONS/ADR-007-kerr-native-tracer.md) keeps C tracing scalar and camera/render-free.
7. Kerr native camera boundary: [ADR-008](../docs/DECISIONS/ADR-008-kerr-native-camera.md) keeps C initialization scalar and render-free.
8. Kerr native crossing boundary: [ADR-009](../docs/DECISIONS/ADR-009-kerr-crossing.md) keeps C geometry-only and disk-policy-free.
9. Kerr native disk boundary: [ADR-010](../docs/DECISIONS/ADR-010-kerr-disk-policy.md) keeps C geometry-independent and color/render-free.

## Final verification matrix

```bash
uv sync --group dev
uv lock --check
uv run pytest -q
uv run ruff check src tests ml
uv run mypy src ml
make -C kernel clean && make -C kernel all
./kernel/harmonic_demo 0.02 8 1
./kernel/phase2_batch_demo
uv build --wheel --sdist
BLACKHOLE_BUILD_NATIVE=1 uv pip install -e . --reinstall
uv run blackhole-ray-tracer --help
uv run blackhole-ray-tracer --phase2-report
uv run blackhole-ray-tracer --phase3-report
uv run blackhole-ray-tracer --phase2-render --phase2-native --img-width 4 --img-height 4 --phase2-out /tmp/phase2_smoke.ppm
uv run blackhole-ray-tracer --phase3-render --img-width 3 --img-height 3 --phase3-out /tmp/phase3_smoke.ppm
```

Run CUDA-specific commands only when `nvcc` and a compatible GPU library exist. Never treat a skipped optional backend as parity evidence.
