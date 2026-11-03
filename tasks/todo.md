# blackhole_ray_tracer implementation queue

Plan: [`tasks/plan.md`](plan.md)

Status: T00–T19 complete. Future Kerr color/rendering, batch, bridge, and CUDA work remains bounded.

## Gate 0 — contract and baseline

- [x] T00: Freeze current worktree and record baseline evidence.
- [x] Add project contract, status, architecture, and implementation index.

## Gate 1 — build and types

- [x] T01: Repair `uv` group syntax, Python matrix, native source paths, C/C++ flags, wheel/sdist contents, and CI gates.
- [x] T02: Canonicalize `DiskConfig`, render stats, result types, and public parameter validation.

### Checkpoint A

- [x] Clean `uv sync --group dev` succeeds in current environment.
- [x] `uv run pytest -q` passes with native extension available.
- [x] `uv run ruff check src tests ml` passes.
- [x] `uv run mypy src ml` passes.
- [x] `uv build --wheel --sdist` succeeds.

## Gate 2 — trusted Schwarzschild path

- [x] T03: Select one canonical C Schwarzschild implementation and remove duplicate 3D math.
- [x] T04: Harden PyBind11 marshalling, output dtypes, GIL behavior, and native parity.

### Checkpoint B

- [x] Python, C scalar, C batch, and bridge parity pass.
- [x] Native extension builds on supported CI.
- [x] Native render is exercised, not skipped.

## Gate 3 — observables and Kerr

- [x] T05: Correct disk crossing state and backend-equivalent observables.
- [x] T06: Add Kerr camera, invariant, convergence, and edge-case validation.

### Checkpoint C

- [x] Disk hit and miss tests prove crossing behavior.
- [x] Kerr spin-zero, invariant, off-equatorial, and near-extreme tests pass.
- [x] No Kerr C port starts before validation passes.

## Gate 4 — optional backends

- [x] T07: Make ML schema, model loading, fallback, accuracy, and packaging explicit.
- [x] T08: Qualify CUDA parity boundaries, error handling, benchmark, and preview routing.

### Checkpoint D

- [x] ML is labeled experimental with explicit fallback and packaged metadata.
- [x] CUDA is optional and unavailable locally without `nvcc`; no silent fallback claim.
- [x] Benchmark evidence records platform, backend, dimensions, and step settings.

## Gate 5 — documentation and release evidence

- [x] T09: Reconcile README, roadmap, rules, FFI docs, Kerr docs, bridge docs, ADRs, tickets, and status.
- [x] Run final verification matrix from `tasks/plan.md`.
- [x] Record intentional skips and unverified platform/GPU claims.

## Current evidence

- Full pytest final: 111 passed, 2 skipped; CUDA parity skip is intentional without library.
- Phase 2 boundary tests and native malformed-batch tests pass.
- Kerr algebra/tracer/camera/crossing/disk ctypes tests pass: 16 focused tests.
- Kerr camera ctypes tests pass: 2 focused tests.
- Kerr crossing ctypes tests pass: 5 focused tests.
- Ruff passes for `src tests ml`.
- Mypy passes for `src ml`.
- Kernel rebuild and demos pass after canonical Phase 2 cutover.
- `uv sync --group dev` completes in current environment.
- Native CPU scalar and batch extension rebuild and import successfully on macOS.
- CUDA library remains unavailable locally because `nvcc` is absent.
- Wheel and sdist include the Kerr C module and algebra/tracer/camera/crossing/disk tests.

## Gate 6 — incremental Kerr native foundation

- [x] T10: Add tested C Kerr BL metric/RHS probe with explicit errors and no renderer claim.
- [x] T11: Run final post-T10 verification and record current test count.

## Gate 7 — bounded Kerr native termination

- [x] T12: Add C Kerr BL single-ray RK4 tracing with shared statuses and ctypes parity.
- [x] T13: Run post-T12 full verification and record current test count.

## Gate 8 — bounded Kerr native camera

- [x] T14: Add C Kerr ZAMO seed initializer with Python parity and explicit domain errors.
- [x] T15: Run post-T14 full verification and record current test count.

## Gate 9 — Kerr native crossing observables

- [x] T16: Add C first strict equatorial crossing output with Python interpolation parity.
- [x] T17: Run post-T16 full verification and record current test count.

## Gate 10 — Kerr native disk observables

- [x] T18: Add geometry-independent C disk hit and Doppler policy with Python parity.
- [x] T19: Run post-T18 full verification and record current test count.
