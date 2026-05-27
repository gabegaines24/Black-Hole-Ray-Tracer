"""Phase 2 benchmarks: presets and timing for 3D Schwarzschild rendering / tracing."""

from __future__ import annotations

import time
from typing import TypedDict

import numpy as np

from .phase2_camera import make_camera_from_config, initial_position_observer, static_observer_null_direction
from .phase2_geodesic import trace_null_geodesic_3d
from .phase2_types import Phase2RenderConfig


class Phase2Preset(TypedDict):
    width: int
    height: int
    dlambda: float
    max_steps: int
    r_escape: float
    fov_deg: float
    r_observer: float


PRESETS: dict[str, Phase2Preset] = {
    "fast": {
        "width": 24,
        "height": 24,
        "dlambda": 0.1,
        "max_steps": 4000,
        "r_escape": 80.0,
        "fov_deg": 70.0,
        "r_observer": 30.0,
    },
    "balanced": {
        "width": 48,
        "height": 48,
        "dlambda": 0.06,
        "max_steps": 8000,
        "r_escape": 80.0,
        "fov_deg": 60.0,
        "r_observer": 30.0,
    },
    "quality": {
        "width": 72,
        "height": 72,
        "dlambda": 0.04,
        "max_steps": 14_000,
        "r_escape": 100.0,
        "fov_deg": 60.0,
        "r_observer": 35.0,
    },
}


def format_phase2_presets_table() -> str:
    lines = [
        "Phase 2 presets (3D Schwarzschild render)",
        f"  {'name':<10}  {'WxH':>9}  {'dlambda':>10}  {'max_steps':>10}  r_esc  fov°  r_obs",
    ]
    for name, p in PRESETS.items():
        wh = f"{p['width']}x{p['height']}"
        lines.append(
            f"  {name:<10}  {wh:>9}  {p['dlambda']:10.4f}  {p['max_steps']:10d}  "
            f"{p['r_escape']:5.0f}  {p['fov_deg']:4.0f}  {p['r_observer']:5.1f}"
        )
    return "\n".join(lines)


def phase2_single_ray_benchmark(
    m: float = 1.0,
    dlambda_values: tuple[float, ...] = (0.12, 0.08, 0.04),
) -> str:
    """Time one center-pixel geodesic at several ``dlambda`` values."""
    cam = make_camera_from_config(
        m=m, r=30.0, theta=np.pi / 2, phi=0.0, fov_deg=60.0, width=32, height=32
    )
    x0 = initial_position_observer(cam)
    v0 = static_observer_null_direction(cam, 0.0, 0.0)
    lines: list[str] = [
        "Phase 2 benchmark (single null geodesic, center pixel)",
        f"- M={m}, observer r=30, equatorial",
        f"  {'dlambda':>10}  {'status':>12}  {'steps':>8}  {'r_min':>10}  {'time_ms':>10}",
    ]
    for dlam in dlambda_values:
        t0 = time.perf_counter()
        r = trace_null_geodesic_3d(
            x0,
            v0,
            m=m,
            dlambda=dlam,
            max_steps=12_000,
            r_escape=80.0,
            store_samples=False,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        rmin = r.r_min if np.isfinite(r.r_min) else float("nan")
        lines.append(
            f"  {dlam:10.4f}  {r.status.value:>12}  {r.steps_taken:8d}  {rmin:10.4f}  {ms:10.1f}"
        )
    lines.append(
        "- Smaller dlambda follows the null cone more accurately but costs more steps at fixed max_steps."
    )
    return "\n".join(lines)


def format_phase2_report() -> str:
    return format_phase2_presets_table() + "\n\n" + phase2_single_ray_benchmark()


def render_config_from_preset(
    name: str,
    m: float = 1.0,
    sky_mode: str = "gradient",
    *,
    use_native_phase2: bool = False,
) -> Phase2RenderConfig:
    p = PRESETS[name]
    return Phase2RenderConfig(
        width=p["width"],
        height=p["height"],
        m=m,
        r_observer=p["r_observer"],
        observer_theta=np.pi / 2,
        observer_phi=0.0,
        fov_deg=p["fov_deg"],
        dlambda=p["dlambda"],
        max_steps=p["max_steps"],
        r_escape=p["r_escape"],
        sky_mode=sky_mode,
        use_native_phase2=use_native_phase2,
    )
