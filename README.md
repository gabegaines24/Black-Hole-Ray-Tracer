# blackhole_ray_tracer

Black hole ray tracer focused on null geodesic integration with a high-performance C kernel and Python orchestration.

## Goals

- Simulate curved light paths around a black hole using numerical integration.
- Start with Schwarzschild behavior and evolve toward Kerr frame-dragging support.
- Keep physics compute in C while using Python for orchestration, training, and visualization.

## Project Layout

- `kernel/`: pure C integration core (no Python headers).
- `bridge/`: pybind11 bridge layer between NumPy arrays and kernel calls.
- `ml/`: surrogate model and data generation/training code.
- `src/blackhole_ray_tracer/`: Python package entrypoints and orchestration helpers.

## Documentation (canonical references)

- **`docs/OVERVIEW.md`** — layered architecture (Python vs planned C/bridge/ml), conventions, commands.
- **`docs/ROADMAP.md`** — phased milestones, acceptance criteria, CLI cheat sheet, gap checklist.
- **`docs/STATE_API.md`** — draft SoA / `RayStatus` FFI contract for kernel + bridge.
- **`docs/AGENT_PROMPT.md`** — per-session starter (paste at top of Composer / Claude threads).
- **`.cursorrules`** — persistent Cursor rules (always on).
- **`AGENTS.md`** — short rules for humans and agents.

## Quickstart

```bash
uv sync
uv run blackhole-ray-tracer
uv run blackhole-ray-tracer --phase1-ab
```

## Phase 2 — Schwarzschild 3D (Python prototype)

A separable **Phase 2** path traces null geodesics in full **Spherical Schwarzschild** with a **static-observer pinhole** camera: Christoffel + RK4 affine integration, per-pixel background for escaped rays, black for captured (shadow) pixels. This is a correctness/foundation path; high resolutions are **slow** (pure Python, one ray per pixel).

```bash
# Presets and single-ray timing
uv run blackhole-ray-tracer --phase2-report

# Small render (default size uses img width/height flags)
uv run blackhole-ray-tracer --phase2-render --img-width 32 --img-height 32 --phase2-out shadow32.ppm

# Faster preset
uv run blackhole-ray-tracer --phase2-render --phase2-preset fast --phase2-out shadow_fast.ppm
```

Or use the dedicated module (requires `PYTHONPATH=src` if running from source without install):

```bash
PYTHONPATH=src uv run python -m blackhole_ray_tracer.phase2_driver --render --preset balanced --out phase2.ppm
PYTHONPATH=src uv run python -m blackhole_ray_tracer.phase2_driver --report
```

**Note:** `kernel/` contains a standalone C **RK4** core plus a **harmonic** Phase A parity demo (`make -C kernel`); Schwarzschild geodesics are **not** in C yet. Phase 2 rendering still uses NumPy + `rk4_step` from Phase 1.

Install optional groups as needed:

```bash
uv sync --group dev
uv sync --group viz
uv sync --group ml
```

## Architecture Notes

- Use Boyer-Lindquist coordinates as the baseline so Schwarzschild (`a = 0`) and Kerr share one coordinate framework.
- Keep the compute loop SoA-friendly for SIMD expansion (`px[]`, `py[]`, `pz[]`, `vx[]`, ...).
- Add early-exit rules in the kernel (`r < r_s` or `r > r_max`) to avoid wasted integration steps.

## Milestones

1. **Phase 1** — Equatorial / toy Schwarzschild demos in Python (RK4, ring image prototype).
2. **Phase 2 (this repo, Python layer)** — 3D Schwarzschild null geodesics + pinhole image (Christoffel form); future: hook C kernel, accretion texture, Doppler.
3. **Phase 3 - ML Warm Starter**: train surrogate model and integrate switching logic with kernel.
4. **Phase 4 - GPU Port**: CUDA port targeting real-time preview (30+ FPS at 720p).

## Development

- Lint: `uv run --group dev ruff check .`
- Type check: `uv run --group dev mypy src`
- Tests: `uv run --group dev pytest`
- Kernel (optional): `make -C kernel && ./kernel/harmonic_demo 0.02 8 1` — parity exercised by `tests/test_kernel_harmonic_parity.py` when a C compiler is available (`SKIP_KERNEL_TESTS=1` to skip).