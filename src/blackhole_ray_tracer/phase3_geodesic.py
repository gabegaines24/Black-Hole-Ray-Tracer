"""Kerr null geodesic integration (Boyer-Lindquist, affine parameter) using RK4.

Mirrors `phase2_geodesic.trace_null_geodesic_3d` but for the full Kerr metric.
Setting spin `a=0` gives numerically identical results to the Schwarzschild tracer.
"""

from __future__ import annotations

import math

__all__ = ["trace_kerr_null_geodesic"]

import numpy as np

from .phase1 import RayStatus, rk4_step
from .phase2_types import GeodesicTraceResult
from .phase3_christoffel import (
    kerr_conserved,
    kerr_null_geodesic_rhs,
    kerr_null_invariant,
    renormalize_vr_kerr,
    kerr_delta,
)


def trace_kerr_null_geodesic(
    x0: np.ndarray,
    v0: np.ndarray,
    m: float,
    a: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    r_horizon_epsilon: float = 1e-3,
    *,
    store_samples: bool = True,
    sample_stride: int = 1,
    monitor_conserved: bool = False,
) -> GeodesicTraceResult:
    """Integrate one Kerr null geodesic.

    Parameters
    ----------
    x0 : (4,) array  — initial (t, r, θ, φ)
    v0 : (4,) array  — initial (v^t, v^r, v^θ, v^φ)
    m  : black-hole mass (geometric units)
    a  : spin parameter (|a| ≤ M; a=0 → Schwarzschild)
    dlambda, max_steps, r_escape, r_horizon_epsilon : integration params
    store_samples : collect r, t, θ along trajectory
    monitor_conserved : print E/L drift warnings (debug only)

    Returns
    -------
    GeodesicTraceResult  (same type as Phase 2, theta_samples populated)
    """
    y = np.concatenate([x0.astype(float), v0.astype(float)])
    y = renormalize_vr_kerr(y, m, a)

    # Capture radius: outer horizon r_+ = M + √(M²−a²) + ε
    disc = m * m - a * a
    r_horizon = m + math.sqrt(max(disc, 0.0))
    r_cap = r_horizon + r_horizon_epsilon

    r_min = float("inf")
    r_samples: list[float] = []
    t_samples: list[float] = []
    theta_samples: list[float] = []

    status = RayStatus.MAX_STEPS
    termination_r = float("nan")
    termination_lambda = 0.0
    steps_taken = 0

    lam = 0.0
    for step_idx in range(max_steps):
        _, r, th, _, vt, vr, vth, vph = (float(x) for x in y)

        if not all(math.isfinite(x) for x in (r, th, vt, vr, vth, vph)):
            status = RayStatus.NUMERICAL_ERROR
            termination_r = r
            termination_lambda = lam
            steps_taken = step_idx
            break
        if r < r_cap:
            status = RayStatus.CAPTURED
            termination_r = r
            termination_lambda = lam
            steps_taken = step_idx
            break
        if r > r_escape:
            status = RayStatus.ESCAPED
            termination_r = r
            termination_lambda = lam
            steps_taken = step_idx
            break

        r_min = min(r_min, r)
        if store_samples and step_idx % max(sample_stride, 1) == 0:
            r_samples.append(r)
            t_samples.append(float(y[0]))
            theta_samples.append(th)

        # RK4 step: pass (m, a) as extra args to the RHS
        y = rk4_step(kerr_null_geodesic_rhs, lam, y, dlambda, m, a)

        if step_idx % 4 == 0:
            y = renormalize_vr_kerr(y, m, a)

        lam += dlambda
        steps_taken = step_idx + 1
    else:
        termination_r = float(y[1])
        termination_lambda = lam

    if status == RayStatus.MAX_STEPS:
        termination_r = float(y[1])
        termination_lambda = lam
        if math.isfinite(float(y[1])):
            r_min = min(r_min, float(y[1]))

    if not r_samples and math.isfinite(float(y[1])):
        r_samples = [float(y[1])]
        t_samples = [float(y[0])]
        theta_samples = [float(y[2])]

    r_min = r_min if math.isfinite(r_min) else float("nan")

    return GeodesicTraceResult(
        status=status,
        steps_taken=steps_taken,
        max_steps=max_steps,
        termination_r=termination_r,
        termination_lambda=termination_lambda,
        r_samples=r_samples,
        t_samples=t_samples,
        theta_samples=theta_samples,
        r_min=r_min,
    )
