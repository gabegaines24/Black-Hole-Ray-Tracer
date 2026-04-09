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

## Quickstart

```bash
uv sync
uv run blackhole-ray-tracer
```

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

1. **Phase 1 - C Foundation**: 2D Schwarzschild null geodesics in C and Einstein ring verification plot.
2. **Phase 2 - 3D Accretion Disk**: textured disk and relativistic Doppler/redshift handling.
3. **Phase 3 - ML Warm Starter**: train surrogate model and integrate switching logic with kernel.
4. **Phase 4 - GPU Port**: CUDA port targeting real-time preview (30+ FPS at 720p).

## Development

- Lint: `uv run --group dev ruff check .`
- Type check: `uv run --group dev mypy src`
- Tests: `uv run --group dev pytest`
