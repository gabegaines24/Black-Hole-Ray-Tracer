"""Kerr Boyer–Lindquist camera using a local ZAMO orthonormal frame."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .phase3_christoffel import kerr_metric_matrix, kerr_outer_horizon


@dataclass(frozen=True, slots=True)
class KerrObserverCamera:
    """Pinhole camera parameters for a Kerr observer outside the outer horizon."""

    m: float
    spin: float
    r: float
    theta: float
    phi: float
    fov_deg: float
    width: int
    height: int


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-15 or not math.isfinite(norm):
        raise ValueError("Camera direction must have finite non-zero norm.")
    return vector / norm


def make_kerr_camera_from_config(
    m: float,
    spin: float,
    r: float,
    theta: float,
    phi: float,
    fov_deg: float,
    width: int,
    height: int,
) -> KerrObserverCamera:
    """Build a Kerr camera after validating its local coordinate frame domain."""
    if not all(math.isfinite(value) for value in (m, spin, r, theta, phi, fov_deg)):
        raise ValueError("Kerr camera parameters must be finite.")
    if m <= 0.0 or abs(spin) > 1.0:
        raise ValueError("Kerr camera requires m > 0 and |spin| <= 1.")
    if r <= kerr_outer_horizon(m, spin):
        raise ValueError("Kerr camera must lie outside the outer horizon.")
    if r <= 0.0 or not 0.0 < theta < math.pi:
        raise ValueError("Kerr camera must lie at positive r and 0 < theta < pi.")
    if abs(math.sin(theta)) < 1e-10:
        raise ValueError("Kerr observer theta must avoid coordinate poles.")
    if not 0.0 < fov_deg < 180.0:
        raise ValueError("Kerr camera FOV must lie strictly between 0 and 180 degrees.")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("Kerr camera width must be a positive integer.")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("Kerr camera height must be a positive integer.")
    metric = kerr_metric_matrix(r, theta, m, spin)
    g_rr = float(metric[1, 1])
    g_theta_theta = float(metric[2, 2])
    g_phi_phi = float(metric[3, 3])
    lapse_squared = -(
        float(metric[0, 0])
        - float(metric[0, 3]) ** 2 / g_phi_phi
    ) if g_phi_phi > 0.0 else float("nan")
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (g_rr, g_theta_theta, g_phi_phi, lapse_squared)
    ):
        raise ValueError("Kerr camera must lie outside a valid ZAMO frame domain.")
    return KerrObserverCamera(m, spin, r, theta, phi, fov_deg, width, height)


def initial_position_observer(camera: KerrObserverCamera) -> np.ndarray:
    """Return initial Boyer–Lindquist position at affine parameter zero."""
    return np.array([0.0, camera.r, camera.theta, camera.phi], dtype=float)


def zamo_null_direction(
    camera: KerrObserverCamera, sx: float, sy: float
) -> np.ndarray:
    """Map screen coordinates to a null coordinate 4-velocity in a ZAMO frame."""
    fov_rad = math.radians(camera.fov_deg)
    aspect = float(camera.width) / float(camera.height)
    tan_half_fov = math.tan(0.5 * fov_rad)
    local_direction = _normalize(
        np.array(
            [-1.0, sx * aspect * tan_half_fov, sy * tan_half_fov],
            dtype=float,
        )
    )
    metric = kerr_metric_matrix(camera.r, camera.theta, camera.m, camera.spin)
    g_rr = float(metric[1, 1])
    g_theta_theta = float(metric[2, 2])
    g_phi_phi = float(metric[3, 3])
    lapse_squared = -(
        float(metric[0, 0])
        - float(metric[0, 3]) ** 2 / g_phi_phi
    )
    lapse = math.sqrt(lapse_squared)
    frame_dragging = -float(metric[0, 3]) / (lapse * g_phi_phi)
    return np.array(
        [
            1.0 / lapse,
            local_direction[0] / math.sqrt(g_rr),
            local_direction[1] / math.sqrt(g_theta_theta),
            frame_dragging + local_direction[2] / math.sqrt(g_phi_phi),
        ],
        dtype=float,
    )
