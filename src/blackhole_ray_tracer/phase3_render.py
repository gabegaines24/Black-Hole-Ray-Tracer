"""Kerr 3D null-geodesic pinhole image renderer.

Dispatches to the Phase 2 (Schwarzschild) path when spin a=0 (and
`use_native_phase2` is set), otherwise uses the Python Kerr integrator
or the native Kerr batch kernel when `use_native_phase3` is set.
"""

from __future__ import annotations

import math

__all__ = ["render_kerr_3d_image"]

import numpy as np

from .phase1 import RayStatus
from .phase2_batch import build_camera_y0
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
from .native_phase3 import (
    batch_native_phase3_available,
    kerr_phase3_batch_native,
    native_phase3_available,
    ray_status_array_from_native,
)
from .phase3_geodesic import trace_kerr_null_geodesic
from .phase3_types import KerrRenderConfig


def render_kerr_3d_image(
    cfg: KerrRenderConfig,
) -> tuple[np.ndarray, dict]:
    """Render a Kerr black-hole image.

    When ``cfg.a == 0`` and ``cfg.use_native_phase2`` is True the call
    is forwarded to the optimised Phase 2 C batch path.  When ``cfg.a != 0``
    and ``cfg.use_native_phase3`` is True the native Kerr C batch path is used.
    Otherwise the Python Kerr integrator is used.

    Supports ``cfg.supersample``: render at ``(W*s, H*s)`` and box-average
    down to ``(W, H)`` for anti-aliasing.
    """
    s = max(cfg.supersample, 1)
    if s > 1:
        import dataclasses

        hi_cfg = dataclasses.replace(
            cfg, width=cfg.width * s, height=cfg.height * s, supersample=1
        )
        rgb_full, stats = render_kerr_3d_image(hi_cfg)
        rgb = (
            rgb_full.reshape(cfg.height, s, cfg.width, s, 3)
            .mean(axis=(1, 3))
            .astype(np.float32)
        )
        stats["supersample"] = s
        return rgb, stats

    # ── a=0 fast-path: forward to optimised Phase 2 renderer ─────────────────
    if cfg.a == 0.0 and cfg.use_native_phase2:
        return render_schwarzschild_3d_image(cfg.to_phase2_config())

    h, w = cfg.height, cfg.width

    if cfg.use_native_phase3 and not native_phase3_available():
        raise RuntimeError(
            "KerrRenderConfig.use_native_phase3 is True but extension "
            "`blackhole_ray_tracer._native_phase3` is not available. "
            "On Windows set BLACKHOLE_BUILD_NATIVE=1 and install MSVC Build Tools, then `uv sync`."
        )

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    n_cap = n_esc = n_other = 0
    backend = "kerr_python"

    use_batch = cfg.use_native_phase3 and batch_native_phase3_available()

    if use_batch:
        backend = "kerr_native_batch"
        y0 = build_camera_y0(cfg.to_phase2_config())
        result = kerr_phase3_batch_native(
            y0,
            m=cfg.m,
            a=cfg.a,
            dlambda=cfg.dlambda,
            max_steps=cfg.max_steps,
            r_escape=cfg.r_escape,
            r_horizon_epsilon=cfg.r_horizon_epsilon,
        )
        statuses = ray_status_array_from_native(result["status"])
        eq_r_cross_arr: np.ndarray | None = result.get("eq_r_cross")
        disk = cfg.disk

        for j in range(h):
            for i in range(w):
                idx = j * w + i
                sx = 2.0 * (i + 0.5) / w - 1.0
                sy = 1.0 - 2.0 * (j + 0.5) / h
                status = statuses[idx]

                disk_drawn = False
                if (
                    disk is not None
                    and status != RayStatus.CAPTURED
                    and eq_r_cross_arr is not None
                ):
                    r_cross = float(eq_r_cross_arr[idx])
                    if math.isfinite(r_cross):
                        y0_ray = y0[idx]
                        dh = disk_hit_from_equatorial_crossing(
                            r_cross,
                            disk,
                            cfg.m,
                            vt=float(y0_ray[4]),
                            vph=float(y0_ray[7]),
                        )
                        if dh.hit:
                            dr, dg, db = disk_color_at_r(r_cross, dh.z_factor, disk)
                            rgb[j, i, 0] = dr
                            rgb[j, i, 1] = dg
                            rgb[j, i, 2] = db
                            n_esc += 1
                            disk_drawn = True

                if not disk_drawn:
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
    else:
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
        disk = cfg.disk
        need_samples = disk is not None

        for j in range(h):
            for i in range(w):
                sx = 2.0 * (i + 0.5) / w - 1.0
                sy = 1.0 - 2.0 * (j + 0.5) / h
                v0 = static_observer_null_direction(cam, sx, sy)

                res = trace_kerr_null_geodesic(
                    x0,
                    v0,
                    m=cfg.m,
                    a=cfg.a,
                    dlambda=cfg.dlambda,
                    max_steps=cfg.max_steps,
                    r_escape=cfg.r_escape,
                    r_horizon_epsilon=cfg.r_horizon_epsilon,
                    store_samples=need_samples,
                )
                status = res.status

                disk_drawn = False
                if disk is not None and status != RayStatus.CAPTURED and res.theta_samples:
                    found, r_cross = detect_equatorial_crossing(
                        res.theta_samples, res.r_samples
                    )
                    if found:
                        dh = disk_hit_from_equatorial_crossing(
                            r_cross, disk, cfg.m, vt=float(v0[0]), vph=float(v0[3])
                        )
                        if dh.hit:
                            dr, dg, db = disk_color_at_r(r_cross, dh.z_factor, disk)
                            rgb[j, i, 0] = dr
                            rgb[j, i, 1] = dg
                            rgb[j, i, 2] = db
                            n_esc += 1
                            disk_drawn = True

                if not disk_drawn:
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
        "backend": backend,
        "spin": cfg.a,
        "supersample": s,
    }
    return rgb, stats
