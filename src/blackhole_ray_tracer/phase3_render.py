"""Kerr pinhole render: trace Phase 3 null geodesics per pixel."""

from __future__ import annotations

import numpy as np

from .phase1 import RayStatus
from .phase2_disk import (
    detect_equatorial_crossing_state,
    disk_color_at_r,
    disk_hit_from_equatorial_crossing,
)
from .phase2_render import _sky_rgb_from_direction
from .phase3_camera import (
    initial_position_observer,
    make_kerr_camera_from_config,
    zamo_null_direction,
)
from .phase3_geodesic import trace_kerr_null_geodesic
from .phase2_types import RenderStats
from .phase3_types import KerrRenderConfig


def render_kerr_image(cfg: KerrRenderConfig) -> tuple[np.ndarray, RenderStats]:
    h, w = cfg.height, cfg.width
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    n_cap = n_esc = n_other = 0
    if cfg.spin is None:
        raise ValueError("KerrRenderConfig must define spin or a.")
    spin = cfg.spin

    cam = make_kerr_camera_from_config(
        m=cfg.m,
        spin=spin,
        r=cfg.r_observer,
        theta=cfg.observer_theta,
        phi=cfg.observer_phi,
        fov_deg=cfg.fov_deg,
        width=w,
        height=h,
    )
    x0 = initial_position_observer(cam)
    disk = cfg.disk
    need_samples = disk is not None

    for j in range(h):
        for i in range(w):
            sx = 2.0 * (i + 0.5) / w - 1.0
            sy = 1.0 - 2.0 * (j + 0.5) / h
            v0 = zamo_null_direction(cam, sx, sy)
            res = trace_kerr_null_geodesic(
                x0,
                v0,
                m=cfg.m,
                spin=cfg.spin,
                dlambda=cfg.dlambda,
                max_steps=cfg.max_steps,
                r_escape=cfg.r_escape,
                r_horizon_epsilon=cfg.r_horizon_epsilon,
                store_samples=need_samples,
            )
            status = res.status

            disk_drawn = False
            if disk is not None and status != RayStatus.CAPTURED and res.theta_samples:
                crossing = detect_equatorial_crossing_state(
                    res.theta_samples, res.r_samples, res.vt_samples, res.vphi_samples
                )
                if crossing is not None:
                    dh = disk_hit_from_equatorial_crossing(
                        crossing.r_cross,
                        disk,
                        cfg.m,
                        vt=crossing.vt_cross,
                        vph=crossing.vph_cross,
                    )
                    if dh.hit:
                        rgb[j, i, :] = disk_color_at_r(
                            crossing.r_cross, dh.z_factor, disk
                        )
                        n_esc += 1
                        disk_drawn = True

            if disk_drawn:
                continue
            if status == RayStatus.CAPTURED:
                rgb[j, i, :] = 0.0
                n_cap += 1
            elif status == RayStatus.ESCAPED:
                rgb[j, i, :] = _sky_rgb_from_direction(sx, sy, cfg.sky_mode)
                n_esc += 1
            else:
                rgb[j, i, :] = (0.12, 0.1, 0.15)
                n_other += 1

    total = max(h * w, 1)
    return rgb, {
        "captured": n_cap,
        "escaped": n_esc,
        "other": n_other,
        "frac_captured": n_cap / total,
        "backend": "python_kerr",
        "spin": spin,
    }


def render_kerr_3d_image(cfg: KerrRenderConfig) -> tuple[np.ndarray, RenderStats]:
    """Render through legacy Phase 3 entry point and preserve backend label."""
    rgb, stats = render_kerr_image(cfg)
    legacy_stats = stats.copy()
    legacy_stats["backend"] = "kerr_python"
    return rgb, legacy_stats
