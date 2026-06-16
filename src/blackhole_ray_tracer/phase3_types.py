"""Phase 3 Kerr render configuration types."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .phase2_types import DiskConfig


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
