"""Schwarzschild 3D pinhole image: trace null geodesics per pixel, color by outcome + sky."""

from __future__ import annotations

import math

__all__ = ["render_schwarzschild_3d_image"]

import numpy as np

from .phase1 import RayStatus
from .phase2_batch import build_camera_y0
from .phase2_camera import make_camera_from_config, initial_position_observer, static_observer_null_direction
from .phase2_disk import (
    detect_equatorial_crossing,
    disk_color_at_r,
    disk_hit_from_equatorial_crossing,
)
from .phase2_geodesic import trace_null_geodesic_3d
from .phase2_types import Phase2RenderConfig
from .native_phase2 import (
    batch_native_available,
    native_phase2_available,
    ray_status_array_from_native,
    ray_status_from_native_phase2,
    schwarzschild_phase2_batch_native,
    schwarzschild_phase2_trace_native,
)


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
    """Return RGB float32 (H, W, 3) in [0,1] and simple stats dict.

    When ``cfg.use_native_phase2`` is True the native C batch kernel is used
    (one call for the entire pixel grid), replacing the Python ray-by-ray loop.

    When ``cfg.supersample > 1`` renders at ``(W*s, H*s)`` then box-averages
    down to ``(W, H)`` for anti-aliasing.
    """
    s = max(getattr(cfg, "supersample", 1), 1)
    if s > 1:
        import dataclasses
        hi_cfg = dataclasses.replace(cfg, width=cfg.width * s, height=cfg.height * s, supersample=1)
        rgb_full, stats = render_schwarzschild_3d_image(hi_cfg)
        rgb = rgb_full.reshape(cfg.height, s, cfg.width, s, 3).mean(axis=(1, 3)).astype(np.float32)
        stats["supersample"] = s
        return rgb, stats

    h, w = cfg.height, cfg.width

    if cfg.use_native_phase2 and not native_phase2_available():
        raise RuntimeError(
            "Phase2RenderConfig.use_native_phase2 is True but extension "
            "`blackhole_ray_tracer._native_phase2` is not available. "
            "On Windows set BLACKHOLE_BUILD_NATIVE=1 and install MSVC Build Tools, then `uv sync`."
        )

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    n_cap = n_esc = n_other = 0
    backend = "python"

    use_batch = cfg.use_native_phase2 and batch_native_available()

    if use_batch:
        # ── native batch path ────────────────────────────────────────────────
        backend = "native_batch"
        y0 = build_camera_y0(cfg)   # (N, 8)
        result = schwarzschild_phase2_batch_native(
            y0,
            m=cfg.m,
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
                            r_cross, disk, cfg.m,
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
        # ── Python (or single-ray native) fallback ───────────────────────────
        use_native = cfg.use_native_phase2  # single-ray native, native_phase2 checked above
        if use_native:
            backend = "native"

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

        disk = cfg.disk  # DiskConfig | None
        need_samples = disk is not None and not use_native

        for j in range(h):
            for i in range(w):
                sx = 2.0 * (i + 0.5) / w - 1.0
                sy = 1.0 - 2.0 * (j + 0.5) / h
                v0 = static_observer_null_direction(cam, sx, sy)
                if use_native:
                    y0_ray = np.concatenate([x0, v0])
                    nt = schwarzschild_phase2_trace_native(
                        y0_ray,
                        m=cfg.m,
                        dlambda=cfg.dlambda,
                        max_steps=cfg.max_steps,
                        r_escape=cfg.r_escape,
                        r_horizon_epsilon=cfg.r_horizon_epsilon,
                    )
                    status = ray_status_from_native_phase2(nt)
                    samples_theta: list[float] = []
                    samples_r: list[float] = []
                else:
                    res = trace_null_geodesic_3d(
                        x0, v0,
                        m=cfg.m,
                        dlambda=cfg.dlambda,
                        max_steps=cfg.max_steps,
                        r_escape=cfg.r_escape,
                        r_horizon_epsilon=cfg.r_horizon_epsilon,
                        store_samples=need_samples,
                    )
                    status = res.status
                    if need_samples:
                        samples_theta = res.theta_samples
                        samples_r = res.r_samples
                    else:
                        samples_theta = []
                        samples_r = res.r_samples

                # Disk detection (Python path only; batch path skips for now)
                disk_drawn = False
                if disk is not None and status != RayStatus.CAPTURED and samples_theta:
                    found, r_cross = detect_equatorial_crossing(samples_theta, samples_r)
                    if found:
                        dh = disk_hit_from_equatorial_crossing(
                            r_cross, disk, cfg.m, vt=float(v0[0]), vph=float(v0[3])
                        )
                        if dh.hit:
                            dr, dg, db = disk_color_at_r(r_cross, dh.z_factor, disk)
                            rgb[j, i, 0] = dr
                            rgb[j, i, 1] = dg
                            rgb[j, i, 2] = db
                            n_esc += 1  # disk hits counted as "escaped" (not captured)
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
        "supersample": max(getattr(cfg, "supersample", 1), 1),
    }
    return rgb, stats
