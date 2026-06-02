"""Phase 2 ray-batch helpers: SoA dataclass and (N,8) bridge-compatible builder.

The SoA `Phase2RayBatch` / `prepare_phase2_ray_batch` path predates the PyBind11
batch bridge and is kept for Python-only use.  `build_camera_y0` returns the
row-major `(N, 8)` array consumed by the C batch kernel via the bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phase2_camera import (
    initial_position_observer,
    make_camera_from_config,
    static_observer_null_direction,
)
from .phase2_geodesic import trace_null_geodesic_3d
from .phase2_types import GeodesicTraceResult, Phase2RenderConfig, StaticObserverCamera


# ── SoA dataclass (Python-only path) ─────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Phase2RayBatch:
    """Structure-of-arrays initial conditions for a render-sized ray batch."""

    width: int
    height: int
    sx: np.ndarray
    sy: np.ndarray
    t0: np.ndarray
    r0: np.ndarray
    theta0: np.ndarray
    phi0: np.ndarray
    vt0: np.ndarray
    vr0: np.ndarray
    vtheta0: np.ndarray
    vphi0: np.ndarray

    @property
    def count(self) -> int:
        return int(self.t0.size)


def prepare_phase2_ray_batch(cfg: Phase2RenderConfig) -> Phase2RayBatch:
    """Create one initial null ray per pixel in SoA layout.

    Pixel order is row-major and matches `phase2_render`: index = `j * width + i`.
    """
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
    count = h * w
    sx = np.empty(count, dtype=float)
    sy = np.empty(count, dtype=float)
    t0 = np.full(count, float(x0[0]), dtype=float)
    r0 = np.full(count, float(x0[1]), dtype=float)
    theta0 = np.full(count, float(x0[2]), dtype=float)
    phi0 = np.full(count, float(x0[3]), dtype=float)
    vt0 = np.empty(count, dtype=float)
    vr0 = np.empty(count, dtype=float)
    vtheta0 = np.empty(count, dtype=float)
    vphi0 = np.empty(count, dtype=float)

    for j in range(h):
        for i in range(w):
            idx = j * w + i
            sx_i = 2.0 * (i + 0.5) / w - 1.0
            sy_i = 1.0 - 2.0 * (j + 0.5) / h
            v0 = static_observer_null_direction(cam, sx_i, sy_i)
            sx[idx] = sx_i
            sy[idx] = sy_i
            vt0[idx] = float(v0[0])
            vr0[idx] = float(v0[1])
            vtheta0[idx] = float(v0[2])
            vphi0[idx] = float(v0[3])

    return Phase2RayBatch(
        width=w, height=h, sx=sx, sy=sy,
        t0=t0, r0=r0, theta0=theta0, phi0=phi0,
        vt0=vt0, vr0=vr0, vtheta0=vtheta0, vphi0=vphi0,
    )


def trace_phase2_ray_batch_python(
    batch: Phase2RayBatch,
    cfg: Phase2RenderConfig,
) -> list[GeodesicTraceResult]:
    """Trace a prepared batch with the Python geodesic implementation."""
    results: list[GeodesicTraceResult] = []
    for idx in range(batch.count):
        x0 = np.array(
            [batch.t0[idx], batch.r0[idx], batch.theta0[idx], batch.phi0[idx]], dtype=float
        )
        v0 = np.array(
            [batch.vt0[idx], batch.vr0[idx], batch.vtheta0[idx], batch.vphi0[idx]], dtype=float
        )
        results.append(
            trace_null_geodesic_3d(
                x0, v0,
                m=cfg.m, dlambda=cfg.dlambda, max_steps=cfg.max_steps,
                r_escape=cfg.r_escape, r_horizon_epsilon=cfg.r_horizon_epsilon,
                store_samples=False,
            )
        )
    return results


# ── (N, 8) bridge-compatible builder ─────────────────────────────────────────

def build_camera_y0(config: Phase2RenderConfig) -> np.ndarray:
    """Return y0 array of shape (N, 8) for every pixel in the render grid.

    Pixels are laid out row-major: row j, column i → flat index j*width + i.
    Screen coords: sx ∈ [-1, 1] left→right, sy ∈ [-1, 1] bottom→top.
    """
    cam = StaticObserverCamera(
        m=config.m,
        r=config.r_observer,
        theta=config.observer_theta,
        phi=config.observer_phi,
        fov_deg=config.fov_deg,
        width=config.width,
        height=config.height,
    )
    x0 = initial_position_observer(cam)
    w, h = config.width, config.height
    n = w * h
    y0 = np.empty((n, 8), dtype=np.float64)
    flat_idx = 0
    for j in range(h):
        for i in range(w):
            sx = 2.0 * (i + 0.5) / w - 1.0
            sy = 1.0 - 2.0 * (j + 0.5) / h
            v0 = static_observer_null_direction(cam, sx, sy)
            y0[flat_idx, :4] = x0
            y0[flat_idx, 4:] = v0
            flat_idx += 1
    return y0
