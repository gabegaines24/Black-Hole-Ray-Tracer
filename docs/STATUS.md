# Project status and qualification evidence

This file records observed evidence. It separates source presence from tested support.

## Current qualification

| Area | Source state | Evidence state | Claim boundary |
|---|---|---|---|
| Phase 1 Python | Present | Tested | Educational/reference path |
| Phase 2 Python | Present | Tested | Schwarzschild reference; deterministic disk crossing coverage |
| Phase 3 Python | Present | Validated Python reference | Kerr Boyer–Lindquist prototype; not production-quality |
| Standalone C kernel | Present | Built and parity-tested | Canonical Phase 2 scalar/batch implementation plus Kerr algebra, camera, tracer, crossing, and disk policy |
| Kerr C CPU path | Present | Metric/RHS/camera/tracer/crossing/disk-policy ctypes parity pass | No color rendering, bridge, batch, CUDA, or performance claim |
| PyBind11 CPU bridge | Present | Installed locally; marshalling and render parity pass | Optional acceleration boundary |
| CUDA batch path | Present | Input/error boundaries pass; unavailable locally | Experimental; parity skipped without CUDA |
| ML runtime gate | Present | Metadata, fallback, and deterministic holdout tests pass | Experimental; not quality-sensitive |
| Packaging | Present | Clean wheel/sdist build includes reference model | Model artifact is deliberately packaged |
| Documentation | Present | Reconciled; link audit and final matrix pass | T10–T18 evidence recorded below |

## Baseline commands

These results were captured before T00/T01 implementation changes.

| Command | Result | Evidence |
|---|---|---|
| `uv run pytest -q` | Pass | 51 passed, 7 skipped |
| `uv run ruff check src tests` | Pass | `All checks passed!` |
| `uv run mypy src` | Fail | 25 project-owned diagnostics |
| `make -C kernel all` | Pass | All kernel demos rebuilt |
| `./kernel/harmonic_demo 0.02 8 1` | Pass | RK4 error output emitted |
| `./kernel/phase2_batch_demo` | Pass | Batch result emitted |
| `uv sync --extra dev --dry-run` | Fail | Extra `dev` is undefined; project uses dependency groups |
| `uv sync --group dev --dry-run` | Pass | No changes required |
| `uv lock --check` | Pass | Lockfile valid |
| `uv build --wheel --sdist` | Fail | Wheel rejects absolute `bridge/module_phase2.cpp` path |
| `BLACKHOLE_BUILD_NATIVE=1 uv pip install -e . --reinstall` | Fail | macOS C compiler rejects `-std=c++17` on `.c` sources |
| Native availability probes | Unavailable | CPU extension not installed |
| CUDA availability probe | Unavailable | `BLACKHOLE_CUDA_LIB` not available |
| CLI help/report smoke | Pass | Phase 2 and Phase 3 report commands ran |

## T01 evidence

| Command | Result | Evidence |
|---|---|---|
| `BLACKHOLE_BUILD_NATIVE=1 uv pip install -e . --reinstall` | Pass | macOS native extension built with separate C/C++ flags |
| Native availability probes | Pass | CPU scalar and batch extension import successfully |
| `uv run pytest -q tests/test_bridge_native_phase2.py tests/test_phase2.py` | Pass | 19 passed, 1 skipped |
| `uv sync --group dev --dry-run` | Pass | Dependency group resolves |
| `uv lock --check` | Pass | Lockfile valid |
| Clean `uv build --wheel --sdist` | Pass | Wheel contains native extension; sdist contains CUDA source |
| `uv run ruff check src tests` | Pass | `All checks passed!` |
| `uv run mypy src` | Fail | 25 project-owned diagnostics remain for T02 |


## T02 evidence

| Command | Result | Evidence |
|---|---|---|
| Phase 2 config boundary tests | Pass | 9 passed |
| Phase 3 config boundary tests | Pass | 3 passed |
| Native malformed-batch boundary test | Pass | 1 passed with installed extension |
| `uv run ruff check src tests ml` | Pass | `All checks passed!` |
| `uv run mypy src ml` | Pass | No issues found in 30 source files |
| `uv run pytest -q` | Pass | 69 passed, 2 skipped |

## T03 evidence

| Command | Result | Evidence |
|---|---|---|
| `make -C kernel clean && make -C kernel all` | Pass | Canonical Phase 2 demos build |
| Scalar Phase 2 C parity | Pass | 8 passed, including four terminal statuses |
| Batch Phase 2 C parity | Pass | 2 passed |
| `./kernel/phase2_batch_demo` | Pass | `BH_PHASE2_BATCH_RESULT` emitted |

## T04 focused evidence

| Command | Result | Evidence |
|---|---|---|
| Native bridge rebuild | Pass | macOS PyBind11 extension rebuilt |
| Native bridge marshalling tests | Pass | 7 passed |
| Native render parity tests | Pass | 4 passed, 1 intentional missing-extension skip |

## T05 evidence

| Command | Result | Evidence |
|---|---|---|
| Deterministic Python/C disk crossing | Pass | Crossing radius and `v^t`/`v^phi` matched within floating-point tolerance |
| Phase 2 disk tests | Pass | 10 focused tests; capture-before-crossing and multiple-crossing cases included |
| Full pytest before T06 | Pass | 78 passed, 2 skipped |
| `uv run ruff check src tests ml` | Pass | `All checks passed!` |
| `uv run mypy src ml` | Pass | No issues found in 30 source files |

## T06 focused evidence

| Command | Result | Evidence |
|---|---|---|
| `uv run pytest -q tests/test_phase3.py` | Pass | 21 passed |
| Kerr camera and invariant subset | Pass | ZAMO null initialization, spin-zero parity, Carter drift, pole, horizon, and convergence tests pass |
| Near-extreme camera/trace cases | Pass | `spin=±0.999999` finite null initialization and finite 100-step traces |
| Kerr CLI render smoke | Pass | 3x3 Python render with disk and `spin=0.999999` completed |

## Pre-existing worktree classification

The worktree was already dirty before this plan. This classification uses paths, not authorship or correctness. Existing edits must be preserved and integrated deliberately.

| Path group | Observed scope | Plan treatment |
|---|---|---|
| `src/blackhole_ray_tracer/phase2_*`, `phase3_*`, `main.py`, `preview.py` | Python rendering, CLI, Kerr, native/CUDA routing | T02, T05, T06, T08 |
| `src/blackhole_ray_tracer/native_phase2*.py` | CPU/CUDA loader façades | T04, T08 |
| `kernel/`, `bridge/` | C kernels, batch path, PyBind11 bridge, CUDA source | T01, T03, T04, T08 |
| `ml/`, `ml/models/` | Dataset, schema, surrogate, runtime gate, reference model | T07 |
| `tests/` | Phase, kernel, bridge, CUDA, ML coverage | T02–T08 |
| `README.md`, `docs/`, `.gitignore`, build files | Claims, commands, packaging, session rules | T00, T01, T09 |
| Untracked `tests/test_cuda_batch_parity.py` | CUDA parity coverage | T08 |
| Reference `ml/models/surrogate.npz` | Present; baseline artifact was untracked | Packaged with schema metadata | T07 policy applied; accuracy remains experimental |

## Known evidence gaps

- Native bridge tests may skip in clean environments without optional extension; local scalar and batch parity pass.
- CUDA parity skips because no compatible library is available.
- Kerr disk color/rendering, bridge, batch, and CUDA remain unqualified; T10/T12/T14/T16/T18 C parity passed.
- ML accuracy remains experimental; deterministic holdout tests pass, but no quality-sensitive route uses surrogate output.

## T07 focused evidence

| Command | Result | Evidence |
|---|---|---|
| ML qualification tests | Pass | 9 passed; metadata, corrupt artifacts, fallback, and holdout checks included |
| `uv run mypy src ml` | Pass | No issues found in 31 source files |
| Deterministic holdout | Pass | 40 rays split 30/10; status accuracy threshold met; finite error vectors |
| `uv build --wheel --sdist` | Pass | Wheel and sdist include `ml/models/surrogate.npz` |
| Full pytest after T07 | Pass | 92 passed, 2 skipped |

## T08 focused evidence

| Command | Result | Evidence |
|---|---|---|
| CUDA input boundary tests | Pass | 1 passed; malformed, nonfinite, empty, and oversized cases covered |
| CUDA parity test | Skipped | No CUDA library available locally |
| `make -C kernel/cuda cuda_batch` | Expected unavailable | CUDA compiler unavailable locally; CUDA parity remains skipped |
| Headless preview routing | Pass | Native/disk/CUDA flags propagated; 4x4 frame rendered without GUI |
| CPU native benchmark | Pass | macOS arm64, 8x8, 300 steps, native batch, 17,905 rays/s observed |

## T09 evidence

| Command or check | Result | Evidence |
|---|---|---|
| Markdown relative-link audit | Pass | All linked file and directory targets exist |
| `uv lock --check` | Pass | Lockfile valid |
| `uv build --wheel --sdist` | Pass | Wheel contains Phase 3, CUDA façade, and ML model; sdist contains CUDA source and model |
| Native editable rebuild | Pass | `BLACKHOLE_BUILD_NATIVE=1 uv pip install -e . --reinstall` completed on macOS |
| Kernel rebuild and demos | Pass | Harmonic and Phase 2 batch outputs emitted |
| CLI help/report smoke | Pass | Main help, Phase 2 report, and Phase 3 report completed |
| Tiny Phase 2 render | Pass | 4x4 native batch render completed; backend `native_batch` |
| Tiny Phase 3 render | Pass | 3x3 Python Kerr render completed; backend `python_kerr`, `spin=0.999999` |
| CUDA build | Expected unavailable | `nvcc not found (nvcc)` remains local boundary |
| `uv run pytest -q` | Pass | 94 passed, 2 skipped; CUDA parity skipped because no library is available |

## T10/T12/T14/T16/T18 focused evidence

| Command or check | Result | Evidence |
|---|---|---|
| `uv run pytest -q tests/test_kernel_phase3_algebra.py` | Pass | 3 passed; metric, RHS, and invalid-domain checks |
| Phase 2 sign regression | Pass | `Gamma^r_rr = -m / (r^2 f)` asserted analytically |
| Phase 2/native parity after rebuild | Pass | 43 passed, 1 intentional skip |
| Kerr algebra compiler | Pass | C99, warnings-as-errors, shared-library ctypes load |
| `uv run pytest -q tests/test_kernel_phase3_trace.py` | Pass | 4 passed; terminal parity, near-horizon capture, and control errors |
| `uv run pytest -q tests/test_kernel_phase3_camera.py` | Pass | 2 passed; ZAMO seed parity, null invariant, and invalid-domain checks |
| `uv run pytest -q tests/test_kernel_phase3_crossing.py` | Pass | 5 passed; first strict crossing interpolation and empty-output semantics |
| `uv run pytest -q tests/test_kernel_phase3_disk.py` | Pass | 2 passed; radius hit policy, Python Doppler parity, sentinels, and invalid controls |
| `uv run pytest -q` | Pass | 111 passed, 2 skipped; optional CUDA parity skipped without library |
| `uv run ruff check src tests ml` | Pass | All checks passed |
| `uv run mypy src ml` | Pass | No issues found in 31 source files |
| `uv build --wheel --sdist` | Pass | Kerr module header/source and algebra/tracer/camera/crossing/disk tests included in sdist |
| Tiny Phase 2 render after sign fix | Pass | 4x4 `native_batch` render completed |
| Kernel build and smoke demos | Pass | Harmonic and Phase 2 batch demos completed |
| Tiny Phase 3 CLI render | Pass | 3x3 Python Kerr render with `spin=0.25` completed |


## Claim policy

Use these labels in future documentation:

- **tested:** command or test ran and passed in a named environment;
- **implemented:** source exists, but qualification may be incomplete;
- **experimental:** behavior exists without release-level evidence;
- **unavailable:** required compiler, library, or hardware was absent;
- **planned:** no usable implementation exists yet.

Do not convert `implemented`, `experimental`, or `unavailable` into `tested` without new evidence.
