"""Phase 2 domain types: camera, render config, and 3D geodesic trace results."""

import math
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np

from .phase1 import RayStatus as RayStatus  # re-export for convenience
from .phase2_disk import DiskConfig as DiskConfig


class RenderStats(TypedDict, total=False):
    """Common render counters and backend metadata."""

    captured: int
    escaped: int
    other: int
    frac_captured: float
    backend: str
    surrogate_frac: float
    spin: float


def _validate_render_parameters(
    *,
    width: int,
    height: int,
    m: float,
    r_observer: float,
    observer_theta: float,
    observer_phi: float,
    fov_deg: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    r_horizon_epsilon: float,
    horizon_radius: float,
) -> None:
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("Render width must be a positive integer.")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("Render height must be a positive integer.")
    finite_values = {
        "m": m,
        "r_observer": r_observer,
        "observer_theta": observer_theta,
        "observer_phi": observer_phi,
        "fov_deg": fov_deg,
        "dlambda": dlambda,
        "r_escape": r_escape,
        "r_horizon_epsilon": r_horizon_epsilon,
        "horizon_radius": horizon_radius,
    }
    if any(not math.isfinite(value) for value in finite_values.values()):
        raise ValueError("Render parameters must be finite.")
    if m <= 0.0:
        raise ValueError("Black hole mass m must be positive.")
    if r_horizon_epsilon < 0.0:
        raise ValueError("Horizon epsilon must be non-negative.")
    if r_observer <= horizon_radius + r_horizon_epsilon:
        raise ValueError("Observer radius must be outside the horizon.")
    if not 0.0 < observer_theta < math.pi:
        raise ValueError("Observer theta must lie strictly between 0 and pi.")
    if not 0.0 < fov_deg < 180.0:
        raise ValueError("Field of view must lie strictly between 0 and 180 degrees.")
    if dlambda <= 0.0:
        raise ValueError("Affine step dlambda must be positive.")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")
    if r_escape <= horizon_radius + r_horizon_epsilon:
        raise ValueError("Escape radius must be outside the horizon.")

@dataclass(frozen=True, slots=True)
class Phase2RenderConfig:
    """Parameters for a Schwarzschild 3D image render."""

    width: int
    height: int
    m: float = 1.0
    r_observer: float = 30.0
    observer_theta: float = 1.5707963267948966  # π/2, equatorial
    observer_phi: float = 0.0
    fov_deg: float = 60.0
    dlambda: float = 0.06
    max_steps: int = 8000
    r_escape: float = 80.0
    r_horizon_epsilon: float = 1e-3
    # sky mapping for escaped rays (see phase2_render)
    sky_mode: str = "gradient"  # "gradient" | "flat"
    # If True and `blackhole_ray_tracer._native_phase2` is installed, use C RK4 per pixel.
    use_native_phase2: bool = False
    # If True (with use_native_phase2), prefer CUDA batch when lib is available.
    use_cuda_batch: bool = False
    # Optional thin accretion disk.  None = disabled.
    disk: DiskConfig | None = None
    # Optional Phase 3 ML surrogate gate model path. Missing path falls back to RK4.
    ml_gate_model_path: str | None = None
    def __post_init__(self) -> None:
        _validate_render_parameters(
            width=self.width,
            height=self.height,
            m=self.m,
            r_observer=self.r_observer,
            observer_theta=self.observer_theta,
            observer_phi=self.observer_phi,
            fov_deg=self.fov_deg,
            dlambda=self.dlambda,
            max_steps=self.max_steps,
            r_escape=self.r_escape,
            r_horizon_epsilon=self.r_horizon_epsilon,
            horizon_radius=2.0 * self.m,
        )
        if self.sky_mode not in {"gradient", "flat"}:
            raise ValueError("sky_mode must be 'gradient' or 'flat'.")


@dataclass(frozen=True, slots=True)
class StaticObserverCamera:
    """Static observer at fixed (r, θ, φ) with pinhole boresight toward coordinate origin."""

    m: float
    r: float
    theta: float
    phi: float
    fov_deg: float
    width: int
    height: int
    # boresight: toward r=0; image plane: +theta = "up" on screen, +phi = "right"
    boresight_inward: bool = True


@dataclass
class GeodesicTraceResult:
    """Result of one 3D null geodesic (affine parameter integration)."""

    status: RayStatus
    steps_taken: int
    max_steps: int
    termination_r: float
    termination_lambda: float
    r_samples: list[float] = field(default_factory=list)
    t_samples: list[float] = field(default_factory=list)
    theta_samples: list[float] = field(default_factory=list)
    vt_samples: list[float] = field(default_factory=list)
    vphi_samples: list[float] = field(default_factory=list)
    r_min: float = float("nan")

    def as_numpy(self) -> dict[str, np.ndarray]:
        return {
            "r": np.asarray(self.r_samples, dtype=float),
            "t": np.asarray(self.t_samples, dtype=float),
            "theta": np.asarray(self.theta_samples, dtype=float),
            "vt": np.asarray(self.vt_samples, dtype=float),
            "vphi": np.asarray(self.vphi_samples, dtype=float),
        }
