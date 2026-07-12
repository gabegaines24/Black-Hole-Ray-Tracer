# blackhole_ray_tracer

Black hole ray tracer focused on null geodesic integration with a high-performance C kernel and Python orchestration.

## Goals

- Simulate curved light paths around a black hole using numerical integration.
- Schwarzschild and Kerr (Boyer–Lindquist) null geodesics with optional accretion disk overlay.
- Heavy integration in C / GPU; Python for orchestration, ML tooling, and visualization.

## Project Layout

- `kernel/`: C integration core (RK4, Schwarzschild, Kerr) + optional `cuda/`.
- `bridge/`: pybind11 extensions `_native_phase2` and `_native_phase3`.
- `ml/`: NumPy MLP surrogate, dataset generator, `RuntimeGate`.
- `src/blackhole_ray_tracer/`: Python package, render backends, CLI entrypoints.

## Documentation (canonical references)

- **`docs/OVERVIEW.md`** — layered architecture, conventions, commands.
- **`docs/ROADMAP.md`** — phased milestones, acceptance criteria, CLI cheat sheet.
- **`docs/STATE_API.md`** — SoA / `RayStatus` FFI contract for kernel + bridge.
- **`docs/AGENT_PROMPT.md`** — per-session starter for agent threads.
- **`.cursorrules`** — persistent Cursor rules.
- **`AGENTS.md`** — short rules for humans and agents.

## Quickstart

```bash
uv sync --group dev
uv run blackhole-ray-tracer --phase1-ab
uv run pytest
```

## Phase 2 — Schwarzschild 3D render

Pinhole camera + Christoffel RK4 integration. Default path is a **Python per-pixel loop** (accuracy-oriented, slow at high resolution). Batch backends:

| Backend | Flag | Requires |
|---------|------|----------|
| C CPU batch | `--phase2-native` | `_native_phase2` (built on `uv sync` with compilers) |
| GPU batch | `--phase2-cuda` | `make -C kernel/cuda cuda_batch` |
| ML surrogate | `--phase2-ml` | trained `ml/models/surrogate.npz` |

```bash
uv run blackhole-ray-tracer --phase2-report
uv run blackhole-ray-tracer --phase2-render --img-width 32 --img-height 32 --phase2-out shadow.ppm
uv run blackhole-ray-tracer --phase2-render --phase2-native --phase2-preset fast --phase2-out shadow_native.ppm
uv run blackhole-ray-tracer --phase2-render --phase2-ml --img-width 32 --phase2-out shadow_ml.ppm
```

Phase 3 Kerr renders: `uv run python -m blackhole_ray_tracer.phase3_driver --render --spin 0.5 --native`

Optional groups: `uv sync --group viz` (matplotlib preview + Streamlit dashboard), `uv sync --group ml` (PyTorch for future training).

### Preview and dashboard

```bash
uv sync --group viz
uv run python -m blackhole_ray_tracer.preview --preset fast --disk --backend auto
uv run blackhole-ray-tracer --phase2-preview --phase2-preset fast --phase2-disk
uv run blackhole-ray-tracer-dashboard
# or: uv run streamlit run src/blackhole_ray_tracer/streamlit_app.py
```

Preview flags: `--backend auto|python|native|cuda|ml`, `--disk`, `--sky gradient|flat`, `--r-observer`, `--fov`, `--aa`.

## Milestones (current)

1. **Phase 1** — Equatorial Schwarzschild Python demos (RK4, Einstein ring).
2. **Phase 2** — 3D Schwarzschild shadow + disk + anti-aliasing; C/GPU/ML batch backends.
3. **Phase 3** — Kerr Python + C batch kernel; ML surrogate wired into render.
4. **Phase 4 (partial)** — CUDA batch kernel + render wiring; pip `nvcc` target still planned.

## Development

- **Git contributor email:** `gabegaines24@gmail.com`
- **Native extensions:** Linux/macOS build `_native_phase2` and `_native_phase3` on `uv sync` when compilers are present. Windows: `BLACKHOLE_BUILD_NATIVE=1 uv sync` + MSVC.
- **Tests:** `uv run pytest`
- **Kernel smoke:** `make -C kernel`
