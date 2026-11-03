# blackhole_ray_tracer

Black hole ray tracer for null geodesic integration, with Python reference paths and optional native acceleration.

## Goals

- Simulate curved light paths around a black hole using numerical integration.
- Start with Schwarzschild behavior and validate a Python Kerr Boyer–Lindquist prototype.
- Keep physics compute inspectable in Python and portable in C/CUDA backends.

## Project Layout

- `kernel/`: pure C integration core (no Python headers).
- `bridge/`: pybind11 bridge layer between NumPy arrays and kernel calls.
- `ml/`: surrogate model and data generation/training code.
- `src/blackhole_ray_tracer/`: Python package entrypoints and orchestration helpers.

## Documentation

- **`docs/OVERVIEW.md`** — implemented layers and module map.
- **`docs/ROADMAP.md`** — phased milestones, acceptance criteria, CLI reference, and gaps.
- **`docs/STATUS.md`** — observed verification, skips, and claim boundaries.
- **`docs/PROJECT_CONTRACT.md`** — numerical, state, disk, Kerr, and backend contract.
- **`docs/STATE_API.md`** — Phase 2 C/Python array layout and FFI dtypes.
- **`docs/IMPLEMENTATION/`** — ticket order, specification, and evidence index.
- **`.cursorrules`** and **`AGENTS.md`** — persistent contributor rules.

## Quickstart

```bash
uv sync
uv run blackhole-ray-tracer
uv run blackhole-ray-tracer --phase1-ab
```

## Phase 2 — Schwarzschild 3D

Phase 2 traces null geodesics in spherical Schwarzschild coordinates with a static-observer pinhole camera.
Python remains the inspectable reference. Qualified CPU batch and optional CUDA paths use the same documented
status, termination, minimum-radius, and disk-crossing fields; CUDA parity remains unavailable locally.

```bash
# Presets and single-ray timing
uv run blackhole-ray-tracer --phase2-report

# Small Python render
uv run blackhole-ray-tracer --phase2-render --img-width 32 --img-height 32 --phase2-out shadow32.ppm

# Native CPU batch render
uv run blackhole-ray-tracer --phase2-render --phase2-preset fast --phase2-native --phase2-out shadow_native.ppm

# Explicit ML experiment; invalid or missing models fall back to RK4
uv run blackhole-ray-tracer --phase2-render --phase2-ml-gate ml/models/surrogate.npz
```

Pure Python renders are accuracy-oriented and slow at high resolution.
PPM outputs remain ignored and untracked.

## Phase 3 — Kerr Boyer–Lindquist prototype

Phase 3 uses a ZAMO/local orthonormal camera and Python RK4 integration.
Tests cover spin-zero parity, Killing quantities, Carter drift, off-equatorial motion,
horizon and pole boundaries, near-extreme spin, and step-size convergence.

```bash
uv run blackhole-ray-tracer --phase3-report
uv run blackhole-ray-tracer --phase3-render --img-width 3 --img-height 3 --spin 0.5 --phase3-out phase3.ppm
```

Native C now covers Kerr metric/RHS algebra, bounded single-ray termination, ZAMO seed initialization,
first equatorial crossing observables, and geometry-independent disk hit/Doppler observables.
Kerr disk color and rendering remain Python-only. No production-quality or real-time claim applies.

## Optional preview and CUDA

```bash
uv run python -m blackhole_ray_tracer.preview --help
uv run blackhole-ray-tracer --phase2-render --phase2-native --phase2-cuda --img-width 4 --img-height 4
```

CUDA requires `nvcc`, a compatible GPU library, and `BLACKHOLE_CUDA_LIB`.
Missing `nvcc` reports an explicit unavailable diagnostic.

Install optional groups as needed:

```bash
uv sync --group dev
uv sync --group viz
uv sync --group ml
```

## Architecture Notes

- Use Boyer–Lindquist coordinates for Kerr Phase 3; Schwarzschild Phase 2 keeps its spherical contract.
- Keep batch compute SoA-friendly for native acceleration.
- Apply documented horizon, escape, capture, and step limits instead of implicit termination.

## Milestones

1. **Phase 1** — Equatorial and toy Schwarzschild demos in Python.
2. **Phase 2** — 3D Schwarzschild reference, canonical C scalar/batch backend, disk observables, optional native/CUDA routing.
3. **Phase 3** — Python-only Kerr prototype with ZAMO camera, invariants, convergence checks, and disk rendering.
4. **ML gate** — Experimental schema-validated surrogate with explicit RK4 fallback.
5. **CUDA** — Optional host-checked batch backend; local qualification depends on `nvcc` and GPU availability.

## Development

- **Git contributor email:** use **`gabegaines24@gmail.com`** (verified on GitHub for contribution graph credit). Repo-local: `git config user.email gabegaines24@gmail.com`.
- **Native Phase 2 extension:** On **Linux/macOS**, `uv sync` builds **`blackhole_ray_tracer._native_phase2`** when compilers are installed. On **Windows**, compiling is **opt‑in**: install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), then `BLACKHOLE_BUILD_NATIVE=1 uv sync`. Otherwise the package installs without the extension; `tests/test_bridge_native_phase2.py` skips accordingly. Unix users can disable with `BLACKHOLE_SKIP_NATIVE=1 uv sync`.
- Lint: `uv run ruff check src tests ml`
- Type check: `uv run mypy src ml`
- Tests: `uv run pytest`
- Kernel (optional): `make -C kernel && ./kernel/harmonic_demo 0.02 8 1` — parity exercised by `tests/test_kernel_harmonic_parity.py` when a C compiler is available (`SKIP_KERNEL_TESTS=1` to skip).
