"""Phase 3 domain types: Kerr render configuration."""

from __future__ import annotations

from dataclasses import dataclass

from .phase2_types import DiskConfig, Phase2RenderConfig


@dataclass(frozen=True, slots=True)
class KerrRenderConfig:
    """Parameters for a Kerr 3D null-geodesic image render.

    Extends Phase2RenderConfig by adding a spin parameter `a`.
    When `a=0` the Kerr integrator is numerically equivalent to the
    Schwarzschild Phase 2 tracer.
    """

    width: int
    height: int
    m: float = 1.0
    a: float = 0.0             # spin parameter (|a| <= M)
    r_observer: float = 30.0
    observer_theta: float = 1.5707963267948966   # π/2
    observer_phi: float = 0.0
    fov_deg: float = 60.0
    dlambda: float = 0.06
    max_steps: int = 8000
    r_escape: float = 80.0
    r_horizon_epsilon: float = 1e-3
    sky_mode: str = "gradient"
    use_native_phase2: bool = False   # if True and a=0, use C batch path
    disk: DiskConfig | None = None
    supersample: int = 1

    def to_phase2_config(self) -> Phase2RenderConfig:
        """Return equivalent Phase2RenderConfig for a=0 dispatch."""
        return Phase2RenderConfig(
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
            sky_mode=self.sky_mode,
            use_native_phase2=self.use_native_phase2,
            disk=self.disk,
            supersample=self.supersample,
        )
