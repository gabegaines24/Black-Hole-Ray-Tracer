"""Phase 3 Kerr presets and single-ray benchmark report."""

from __future__ import annotations

import time
from typing import TypedDict

import numpy as np

from .phase3_camera import (
    initial_position_observer,
    make_kerr_camera_from_config,
    zamo_null_direction,
)
from .phase3_geodesic import trace_kerr_null_geodesic
from .phase3_types import KerrRenderConfig


class Phase3Preset(TypedDict):
    width: int
    height: int
    dlambda: float
    max_steps: int
    r_escape: float
    fov_deg: float
    r_observer: float
    observer_theta: float


PRESETS: dict[str, Phase3Preset] = {
    "fast": {
        "width": 18,
        "height": 18,
        "dlambda": 0.12,
        "max_steps": 4_000,
        "r_escape": 80.0,
        "fov_deg": 70.0,
        "r_observer": 30.0,
        "observer_theta": float(np.pi / 3.0),
    },
    "balanced": {
        "width": 32,
        "height": 32,
        "dlambda": 0.08,
        "max_steps": 8_000,
        "r_escape": 80.0,
        "fov_deg": 60.0,
        "r_observer": 30.0,
        "observer_theta": float(np.pi / 3.0),
    },
    "quality": {
        "width": 48,
        "height": 48,
        "dlambda": 0.05,
        "max_steps": 14_000,
        "r_escape": 100.0,
        "fov_deg": 60.0,
        "r_observer": 35.0,
        "observer_theta": float(np.pi / 3.0),
    },
}


def render_config_from_preset(
    name: str,
    m: float = 1.0,
    spin: float = 0.5,
    sky_mode: str = "gradient",
) -> KerrRenderConfig:
    p = PRESETS[name]
    return KerrRenderConfig(
        width=p["width"],
        height=p["height"],
        m=m,
        spin=spin,
        r_observer=p["r_observer"],
        observer_theta=p["observer_theta"],
        fov_deg=p["fov_deg"],
        dlambda=p["dlambda"],
        max_steps=p["max_steps"],
        r_escape=p["r_escape"],
        sky_mode=sky_mode,
    )


def format_phase3_presets_table() -> str:
    lines = [
        "Phase 3 presets (Kerr BL render)",
        f"  {'name':<10}  {'WxH':>9}  {'dlambda':>10}  {'max_steps':>10}  r_esc  fov°  theta°",
    ]
    for name, p in PRESETS.items():
        wh = f"{p['width']}x{p['height']}"
        theta_deg = np.degrees(p["observer_theta"])
        lines.append(
            f"  {name:<10}  {wh:>9}  {p['dlambda']:10.4f}  {p['max_steps']:10d}  "
            f"{p['r_escape']:5.0f}  {p['fov_deg']:4.0f}  {theta_deg:6.1f}"
        )
    return "\n".join(lines)


def phase3_single_ray_benchmark(
    m: float = 1.0,
    spin: float = 0.5,
    dlambda_values: tuple[float, ...] = (0.14, 0.1, 0.06),
) -> str:
    cam = make_kerr_camera_from_config(
        m=m,
        spin=spin,
        r=30.0,
        theta=np.pi / 3.0,
        phi=0.0,
        fov_deg=60.0,
        width=24,
        height=24,
    )
    x0 = initial_position_observer(cam)
    v0 = zamo_null_direction(cam, 0.0, 0.0)
    lines = [
        "Phase 3 benchmark (single Kerr null geodesic, center pixel)",
        f"- M={m}, spin a/M={spin}, observer r=30, theta=60deg",
        f"  {'dlambda':>10}  {'status':>12}  {'steps':>8}  {'r_min':>10}  {'time_ms':>10}",
    ]
    for dlam in dlambda_values:
        t0 = time.perf_counter()
        res = trace_kerr_null_geodesic(
            x0,
            v0,
            m=m,
            spin=spin,
            dlambda=dlam,
            max_steps=12_000,
            r_escape=80.0,
            store_samples=False,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        rmin = res.r_min if np.isfinite(res.r_min) else float("nan")
        lines.append(
            f"  {dlam:10.4f}  {res.status.value:>12}  {res.steps_taken:8d}  {rmin:10.4f}  {ms:10.1f}"
        )
    return "\n".join(lines)


def format_phase3_report(m: float = 1.0, spin: float = 0.5) -> str:
    return format_phase3_presets_table() + "\n\n" + phase3_single_ray_benchmark(
        m=m, spin=spin
    )
