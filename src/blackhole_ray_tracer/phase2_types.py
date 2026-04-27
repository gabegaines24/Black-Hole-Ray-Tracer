"""Phase 2 domain types: camera, render config, and 3D geodesic trace results."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .phase1 import RayStatus as RayStatus  # re-export for convenience


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
    r_min: float = float("nan")

    def as_numpy(self) -> dict[str, np.ndarray]:
        return {
            "r": np.asarray(self.r_samples, dtype=float),
            "t": np.asarray(self.t_samples, dtype=float),
        }
