"""Step F: presets and benchmarks for trading quality vs speed (h, r_escape, FOV via b range).

``dphi`` is the RK4 step in phi (same role as plan's ``h`` for this integrator).
"""

from __future__ import annotations

import time
from typing import TypedDict

import numpy as np

from .phase1 import trace_single_schwarzschild_ray


class StepEPreset(TypedDict):
    width: int
    height: int
    dphi: float
    phi_max: float
    r_escape: float
    b_min: float
    b_max: float


PRESETS: dict[str, StepEPreset] = {
    "fast": {
        "width": 48,
        "height": 48,
        "dphi": 0.018,
        "phi_max": 8.0,
        "r_escape": 80.0,
        "b_min": 2.5,
        "b_max": 10.0,
    },
    "balanced": {
        "width": 72,
        "height": 72,
        "dphi": 0.012,
        "phi_max": 8.0,
        "r_escape": 80.0,
        "b_min": 2.5,
        "b_max": 10.0,
    },
    "quality": {
        "width": 128,
        "height": 128,
        "dphi": 0.008,
        "phi_max": 10.0,
        "r_escape": 100.0,
        "b_min": 2.5,
        "b_max": 10.0,
    },
}


def format_presets_table() -> str:
    lines = [
        "Step F presets (Step E render)",
        f"  {'name':<10}  {'WxH':>9}  {'dphi':>8}  {'phi_max':>8}  {'r_esc':>8}  b_min..b_max",
    ]
    for name, p in PRESETS.items():
        wh = f"{p['width']}x{p['height']}"
        lines.append(
            f"  {name:<10}  {wh:>9}  {p['dphi']:8.4f}  {p['phi_max']:8.1f}  "
            f"{p['r_escape']:8.0f}  {p['b_min']:.1f}..{p['b_max']:.1f}"
        )
    return "\n".join(lines)


def step_f_benchmark(
    b_ref: float = 6.0,
    m: float = 1.0,
    dphi_values: tuple[float, ...] = (0.02, 0.012, 0.006),
) -> str:
    """Time one reference ray at several ``dphi`` values; compare coarse vs fine r_min."""
    lines: list[str] = [
        "Step F benchmark (single ray, reference b)",
        f"- b={b_ref}, M={m}",
        f"  {'dphi':>8}  {'status':>12}  {'steps':>6}  {'r_min':>10}  {'time_ms':>10}",
    ]
    r_mins: list[float] = []
    for dphi in dphi_values:
        t0 = time.perf_counter()
        r = trace_single_schwarzschild_ray(
            b=b_ref, m=m, dphi=dphi, phi_max=8.0, r_escape=80.0
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if len(r.rs) > 0:
            r_min = float(np.min(r.rs))
        else:
            r_min = float("nan")
        r_mins.append(r_min)
        lines.append(
            f"  {dphi:8.4f}  {r.status.value:>12}  {r.steps_taken:6d}  {r_min:10.4f}  {elapsed_ms:10.1f}"
        )
    if len(r_mins) >= 2:
        d = abs(r_mins[-1] - r_mins[-2])
        lines.append(
            f"- |r_min(dphi={dphi_values[-1]}) - r_min(dphi={dphi_values[-2]})| = {d:.4e} "
            "(smaller => step size less critical for this ray)"
        )
    lines.append(
        "- Tip: smaller dphi and larger r_escape sharpen the shadow edge but cost more steps per ray."
    )
    return "\n".join(lines)


def format_step_f_report() -> str:
    """Full Step F text: presets + benchmark."""
    return format_presets_table() + "\n\n" + step_f_benchmark()
