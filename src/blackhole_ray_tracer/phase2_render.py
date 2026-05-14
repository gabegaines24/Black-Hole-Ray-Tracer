"""Schwarzschild 3D pinhole image: trace null geodesics per pixel, color by outcome + sky."""

from __future__ import annotations

import math

import numpy as np

from .phase1 import RayStatus
from .phase2_camera import make_camera_from_config, initial_position_observer, static_observer_null_direction
from .phase2_geodesic import trace_null_geodesic_3d
from .phase2_types import Phase2RenderConfig
from .native_phase2 import native_phase2_available, ray_status_from_native_phase2, schwarzschild_phase2_trace_native


def _sky_rgb_from_direction(
    sx: float, sy: float, mode: str
) -> tuple[float, float, float]:
    """Map screen space to a simple synthetic background color (escaped rays)."""
    if mode == "flat":
        return (0.15, 0.18, 0.28)
    th = math.atan2(sy, sx)
    g = 0.5 + 0.5 * math.sin(th)
    b = 0.5 + 0.5 * math.cos(th + 0.7)
    r = 0.35 + 0.25 * g
    return (min(r, 1.0), min(g, 1.0), min(b, 1.0))


def render_schwarzschild_3d_image(
    cfg: Phase2RenderConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return RGB float32 (H, W, 3) in [0,1] and simple stats dict."""
    h, w = cfg.height, cfg.width
    cam = make_camera_from_config(
        m=cfg.m,
        r=cfg.r_observer,
        theta=cfg.observer_theta,
        phi=cfg.observer_phi,
        fov_deg=cfg.fov_deg,
        width=w,
        height=h,
    )
    x0 = initial_position_observer(cam)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    n_cap = n_esc = n_other = 0

    use_native = cfg.use_native_phase2 and native_phase2_available()
    if cfg.use_native_phase2 and not native_phase2_available():
        raise RuntimeError(
            "Phase2RenderConfig.use_native_phase2 is True but extension "
            "`blackhole_ray_tracer._native_phase2` is not available. "
            "On Windows set BLACKHOLE_BUILD_NATIVE=1 and install MSVC Build Tools, then `uv sync`."
        )

    for j in range(h):
        for i in range(w):
            sx = 2.0 * (i + 0.5) / w - 1.0
            sy = 1.0 - 2.0 * (j + 0.5) / h
            v0 = static_observer_null_direction(cam, sx, sy)
            if use_native:
                y0 = np.concatenate([x0, v0])
                nt = schwarzschild_phase2_trace_native(
                    y0,
                    m=cfg.m,
                    dlambda=cfg.dlambda,
                    max_steps=cfg.max_steps,
                    r_escape=cfg.r_escape,
                    r_horizon_epsilon=cfg.r_horizon_epsilon,
                )
                status = ray_status_from_native_phase2(nt)
            else:
                result = trace_null_geodesic_3d(
                    x0,
                    v0,
                    m=cfg.m,
                    dlambda=cfg.dlambda,
                    max_steps=cfg.max_steps,
                    r_escape=cfg.r_escape,
                    r_horizon_epsilon=cfg.r_horizon_epsilon,
                    store_samples=False,
                )
                status = result.status
            if status == RayStatus.CAPTURED:
                rgb[j, i, :] = 0.0
                n_cap += 1
            elif status == RayStatus.ESCAPED:
                br, bg, bb = _sky_rgb_from_direction(sx, sy, cfg.sky_mode)
                rgb[j, i, 0] = br
                rgb[j, i, 1] = bg
                rgb[j, i, 2] = bb
                n_esc += 1
            else:
                rgb[j, i, :] = (0.12, 0.1, 0.15)
                n_other += 1

    total = max(h * w, 1)
    stats = {
        "captured": n_cap,
        "escaped": n_esc,
        "other": n_other,
        "frac_captured": n_cap / total,
        "backend": "native" if use_native else "python",
    }
    return rgb, stats
