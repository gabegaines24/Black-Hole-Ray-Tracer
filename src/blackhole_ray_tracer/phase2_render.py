"""Schwarzschild 3D pinhole image: trace null geodesics per pixel, color by outcome + sky."""

from __future__ import annotations

import math

import numpy as np
from ml.schema import STATUS_CAPTURED, STATUS_ESCAPED, STATUS_MAX_STEPS, STATUS_NUMERIC

from .phase1 import RayStatus
from .phase2_batch import build_camera_y0
from .phase2_camera import make_camera_from_config, initial_position_observer, static_observer_null_direction
from .phase2_disk import (
    detect_equatorial_crossing_state,
    disk_color_at_r,
    disk_hit_from_equatorial_crossing,
)
from .phase2_geodesic import trace_null_geodesic_3d
from .phase2_types import Phase2RenderConfig, RenderStats
from .native_phase2 import (
    batch_native_available,
    native_phase2_available,
    ray_status_array_from_native,
    ray_status_from_native_phase2,
    schwarzschild_phase2_batch_native,
    schwarzschild_phase2_trace_native,
)
from .native_phase2_cuda import cuda_batch_available, schwarzschild_phase2_batch_cuda


def _ray_status_from_ml_gate(status: int) -> RayStatus:
    if status == STATUS_CAPTURED:
        return RayStatus.CAPTURED
    if status == STATUS_ESCAPED:
        return RayStatus.ESCAPED
    if status == STATUS_MAX_STEPS:
        return RayStatus.MAX_STEPS
    if status == STATUS_NUMERIC:
        return RayStatus.NUMERICAL_ERROR
    return RayStatus.MAX_STEPS


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


def _color_batch_pixels(
    cfg: Phase2RenderConfig,
    result: dict,
    statuses: list[RayStatus],
    rgb: np.ndarray,
) -> tuple[int, int, int]:
    """Color an image from a batch trace result dict. Returns (n_cap, n_esc, n_other)."""
    h, w = cfg.height, cfg.width
    disk = cfg.disk
    n_cap = n_esc = n_other = 0
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
                and int(result["disk_hit"][idx]) != 0
            ):
                r_cross = float(result["disk_r_cross"][idx])
                dh = disk_hit_from_equatorial_crossing(
                    r_cross,
                    disk,
                    cfg.m,
                    vt=float(result["disk_vt"][idx]),
                    vph=float(result["disk_vph"][idx]),
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
    return n_cap, n_esc, n_other


def render_schwarzschild_3d_image(
    cfg: Phase2RenderConfig,
) -> tuple[np.ndarray, RenderStats]:
    """Return RGB float32 (H, W, 3) in [0,1] and simple stats dict.

    When ``cfg.use_native_phase2`` is True the native C batch kernel is used
    (one call for the entire pixel grid), replacing the Python ray-by-ray loop.
    """
    h, w = cfg.height, cfg.width

    if cfg.use_native_phase2 and not native_phase2_available():
        raise RuntimeError(
            "Phase2RenderConfig.use_native_phase2 is True but extension "
            "`blackhole_ray_tracer._native_phase2` is not available. "
            "On Windows set BLACKHOLE_BUILD_NATIVE=1 and install MSVC Build Tools, then `uv sync`."
        )

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    n_cap = n_esc = n_other = 0
    n_surrogate = 0
    backend = "python"

    use_cuda = cfg.use_native_phase2 and cfg.use_cuda_batch and cuda_batch_available()
    use_batch = cfg.use_native_phase2 and batch_native_available() and not use_cuda
    use_ml_gate = cfg.ml_gate_model_path is not None
    if use_ml_gate and cfg.use_native_phase2:
        raise ValueError("ML gate and native Phase 2 backend cannot be enabled together.")

    if use_cuda or use_batch:
        backend = "cuda_batch" if use_cuda else "native_batch"
        y0 = build_camera_y0(cfg)
        batch_fn = schwarzschild_phase2_batch_cuda if use_cuda else schwarzschild_phase2_batch_native
        result = batch_fn(
            y0,
            m=cfg.m,
            dlambda=cfg.dlambda,
            max_steps=cfg.max_steps,
            r_escape=cfg.r_escape,
            r_horizon_epsilon=cfg.r_horizon_epsilon,
        )
        statuses = ray_status_array_from_native(result["status"])
        n_cap, n_esc, n_other = _color_batch_pixels(cfg, result, statuses, rgb)

    else:
        # ── Python (or single-ray native) fallback ───────────────────────────
        use_native = cfg.use_native_phase2  # single-ray native, native_phase2 checked above
        if use_native:
            backend = "native"
        ml_gate = None
        if use_ml_gate:
            from ml.runtime_gate import RuntimeGate

            ml_gate = RuntimeGate.from_model_path(str(cfg.ml_gate_model_path))
            backend = "ml_gate" if ml_gate.surrogate_loaded else "ml_gate_fallback"

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
                samples_theta: list[float] = []
                samples_r: list[float] = []
                samples_vt: list[float] = []
                samples_vphi: list[float] = []
                if ml_gate is not None:
                    y0_ray = np.concatenate([x0, v0])
                    gate_result = ml_gate.infer_or_integrate(
                        y0_ray,
                        m=cfg.m,
                        dlambda=cfg.dlambda,
                        max_steps=cfg.max_steps,
                        r_escape=cfg.r_escape,
                        r_horizon_epsilon=cfg.r_horizon_epsilon,
                    )
                    status = _ray_status_from_ml_gate(gate_result.status)
                    if gate_result.from_surrogate:
                        n_surrogate += 1

                elif use_native:
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
                        samples_vt = res.vt_samples
                        samples_vphi = res.vphi_samples
                    else:
                        samples_theta = []
                        samples_r = []
                        samples_vt = []
                        samples_vphi = []

                # Disk detection (Python path; batch path handles disk above)
                disk_drawn = False
                if disk is not None and status != RayStatus.CAPTURED and samples_theta:
                    crossing = detect_equatorial_crossing_state(
                        samples_theta, samples_r, samples_vt, samples_vphi
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
                            dr, dg, db = disk_color_at_r(
                                crossing.r_cross, dh.z_factor, disk
                            )
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
    stats: RenderStats = {
        "captured": n_cap,
        "escaped": n_esc,
        "other": n_other,
        "frac_captured": n_cap / total,
        "backend": backend,
        "surrogate_frac": n_surrogate / total,
    }
    return rgb, stats
