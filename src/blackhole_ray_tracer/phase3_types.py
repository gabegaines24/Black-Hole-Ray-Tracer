"""Phase 3 Kerr render configuration types."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .phase2_types import DiskConfig, _validate_render_parameters
from .phase3_christoffel import kerr_outer_horizon


@dataclass(frozen=True, slots=True)
class KerrRenderConfig:
    """Parameters for a Kerr Boyer-Lindquist pinhole render."""

    width: int
    height: int
    m: float = 1.0
    spin: float = 0.5
    r_observer: float = 30.0
    observer_theta: float = math.pi / 3.0
    observer_phi: float = 0.0
    fov_deg: float = 60.0
    dlambda: float = 0.06
    max_steps: int = 10_000
    r_escape: float = 80.0
    r_horizon_epsilon: float = 1e-3
    sky_mode: str = "gradient"
    disk: DiskConfig | None = None
    def __post_init__(self) -> None:
        if not math.isfinite(self.spin) or abs(self.spin) > 1.0:
            raise ValueError("Kerr spin must be finite with |spin| <= 1.")
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
            horizon_radius=kerr_outer_horizon(self.m, self.spin),
        )
        if abs(math.sin(self.observer_theta)) < 1e-10:
            raise ValueError("Kerr observer theta must avoid coordinate poles.")
        if self.sky_mode not in {"gradient", "flat"}:
            raise ValueError("sky_mode must be 'gradient' or 'flat'.")
