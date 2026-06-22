"""Kerr 3D null-geodesic pinhole image renderer.

Dispatches to the Phase 2 (Schwarzschild) path when spin a=0 (and
`use_native_phase2` is set), otherwise uses the Python Kerr integrator.
"""

from __future__ import annotations

import math

__all__ = ["render_kerr_3d_image"]

import numpy as np

from .phase1 import RayStatus
from .phase2_camera import (
    initial_position_observer,
    make_camera_from_config,
    static_observer_null_direction,
)
from .phase2_disk import (
    detect_equatorial_crossing,
    disk_color_at_r,
    disk_hit_from_equatorial_crossing,
)
from .phase2_render import render_schwarzschild_3d_image, _sky_rgb_from_direction
from .phase3_geodesic import trace_kerr_null_geodesic
from .phase3_types import KerrRenderConfig


def render_kerr_3d_image(
    cfg: KerrRenderConfig,
) -> tuple[np.ndarray, dict]:
    """Render a Kerr black-hole image.

    When ``cfg.a == 0`` and ``cfg.use_native_phase2`` is True the call
    is forwarded to the optimised Phase 2 C batch path.  Otherwise the
    Python Kerr integrator is used.

    Supports ``cfg.supersample``: render at ``(W*s, H*s)`` and box-average
    down to ``(W, H)`` for anti-aliasing.
    """
    # ── a=0 fast-path: forward to optimised Phase 2 renderer ─────────────────
    if cfg.a == 0.0 and cfg.use_native_phase2:
        return render_schwarzschild_3d_image(cfg.to_phase2_config())

    # ── Kerr Python path ──────────────────────────────────────────────────────
    s = max(cfg.supersample, 1)
    render_w = cfg.width * s
    render_h = cfg.height * s

    cam = make_camera_from_config(
        m=cfg.m,
        r=cfg.r_observer,
        theta=cfg.observer_theta,
        phi=cfg.observer_phi,
        fov_deg=cfg.fov_deg,
        width=render_w,
        height=render_h,
    )
    x0 = initial_position_observer(cam)

    rgb_full = np.zeros((render_h, render_w, 3), dtype=np.float32)
    n_cap = n_esc = n_other = 0
    disk = cfg.disk
    need_samples = disk is not None

    for j in range(render_h):
        for i in range(render_w):
            sx = 2.0 * (i + 0.5) / render_w - 1.0
            sy = 1.0 - 2.0 * (j + 0.5) / render_h
            v0 = static_observer_null_direction(cam, sx, sy)

            res = trace_kerr_null_geodesic(
                x0, v0,
                m=cfg.m, a=cfg.a,
                dlambda=cfg.dlambda, max_steps=cfg.max_steps,
                r_escape=cfg.r_escape, r_horizon_epsilon=cfg.r_horizon_epsilon,
                store_samples=need_samples,
            )
            status = res.status

            disk_drawn = False
            if disk is not None and status != RayStatus.CAPTURED and res.theta_samples:
                found, r_cross = detect_equatorial_crossing(res.theta_samples, res.r_samples)
                if found:
                    dh = disk_hit_from_equatorial_crossing(
                        r_cross, disk, cfg.m, vt=float(v0[0]), vph=float(v0[3])
                    )
                    if dh.hit:
                        dr, dg, db = disk_color_at_r(r_cross, dh.z_factor, disk)
                        rgb_full[j, i, 0] = dr
                        rgb_full[j, i, 1] = dg
                        rgb_full[j, i, 2] = db
                        n_esc += 1
                        disk_drawn = True

            if not disk_drawn:
                if status == RayStatus.CAPTURED:
                    rgb_full[j, i, :] = 0.0
                    n_cap += 1
                elif status == RayStatus.ESCAPED:
                    br, bg, bb = _sky_rgb_from_direction(sx, sy, cfg.sky_mode)
                    rgb_full[j, i, 0] = br
                    rgb_full[j, i, 1] = bg
                    rgb_full[j, i, 2] = bb
                    n_esc += 1
                else:
                    rgb_full[j, i, :] = (0.12, 0.1, 0.15)
                    n_other += 1

    # ── box-average supersample ───────────────────────────────────────────────
    if s > 1:
        rgb = rgb_full.reshape(cfg.height, s, cfg.width, s, 3).mean(axis=(1, 3))
        rgb = rgb.astype(np.float32)
    else:
        rgb = rgb_full

    total = max(cfg.height * cfg.width, 1)
    stats = {
        "captured": n_cap,
        "escaped": n_esc,
        "other": n_other,
        "frac_captured": n_cap / (total * s * s),
        "backend": "kerr_python",
        "spin": cfg.a,
    }
    return rgb, stats
