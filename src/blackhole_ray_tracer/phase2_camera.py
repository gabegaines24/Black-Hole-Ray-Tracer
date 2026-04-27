"""Pinhole camera for a static observer in Schwarzschild: local tetrad to coordinate 4-velocity."""

from __future__ import annotations

import math

import numpy as np

from .phase2_christoffel import f_schwarzschild
from .phase2_types import StaticObserverCamera


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return v * 0.0
    return v / n


def static_observer_null_direction(
    camera: StaticObserverCamera, sx: float, sy: float
) -> np.ndarray:
    r"""Map normalized screen coordinates (sx, sy) in [-1,1]^2 to initial coordinate 4-velocity v^\\mu.

    Local Minkowski frame: e_0 = static observer, e_1 = outward radial, e_2 = +\\theta, e_3 = +\\phi.
    Boresight is **inward** (toward the origin), i.e. -e_1, matching a camera looking at the black hole.
    """
    m = camera.m
    r = camera.r
    th = camera.theta
    fov_rad = math.radians(camera.fov_deg)
    # Vertical FOV = fov_deg; scale horizontal by aspect ratio
    aspect = float(camera.width) / max(float(camera.height), 1.0)
    tan_h = math.tan(fov_rad * 0.5)
    tan_w = aspect * tan_h
    # "Right" in image = +phi, "up" = +theta; flip sy so y pixel down => up is positive ey
    # spatial local: e_rad_out = e1, e_th = e2, e_ph = e3. Inward boresight = -e1.
    b = np.array([-1.0, 0.0, 0.0], dtype=float)
    right = np.array([0.0, 1.0, 0.0], dtype=float)  # +theta
    up = np.array([0.0, 0.0, 1.0], dtype=float)  # +phi
    d_local = b + sx * tan_w * right + sy * tan_h * up
    d_local = _normalize(d_local)
    # null in local: p^0=1, p^i = d_local^i, need -1 + |d|^2 =0 => |d|=1
    p0 = 1.0
    p1, p2, p3 = float(d_local[0]), float(d_local[1]), float(d_local[2])

    f = f_schwarzschild(r, m)
    if f <= 0.0 or not math.isfinite(f):
        raise ValueError("Observer r must be outside the horizon (f > 0).")
    s_th = math.sin(th)
    if abs(s_th) < 1e-10:
        raise ValueError("Observer theta must avoid poles for this tetrad (use theta != 0, pi).")
    f_sqrt = math.sqrt(f)
    inv_f_sqrt = 1.0 / f_sqrt

    vt = p0 * inv_f_sqrt
    vr = p1 * f_sqrt
    vth = p2 / r
    vph = p3 / (r * s_th)
    v = np.array([vt, vr, vth, vph], dtype=float)
    return v


def initial_position_observer(camera: StaticObserverCamera) -> np.ndarray:
    return np.array(
        [0.0, camera.r, camera.theta, camera.phi], dtype=float
    )  # t=0 at first crossing


def make_camera_from_config(
    m: float,
    r: float,
    theta: float,
    phi: float,
    fov_deg: float,
    width: int,
    height: int,
) -> StaticObserverCamera:
    return StaticObserverCamera(
        m=m,
        r=r,
        theta=theta,
        phi=phi,
        fov_deg=fov_deg,
        width=width,
        height=height,
    )
