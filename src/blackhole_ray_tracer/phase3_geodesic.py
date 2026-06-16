"""Kerr null geodesic integration in Boyer-Lindquist coordinates."""

from __future__ import annotations

import math

import numpy as np

from .phase1 import RayStatus, rk4_step
from .phase2_types import GeodesicTraceResult
from .phase3_christoffel import (
    I_PH,
    I_T,
    kerr_metric_matrix,
    kerr_outer_horizon,
    null_geodesic_kerr_first_order,
)


def metric_invariant_kerr(
    t: float,
    r: float,
    th: float,
    ph: float,
    vt: float,
    vr: float,
    vth: float,
    vph: float,
    m: float,
    spin: float,
) -> float:
    """Return g_mu_nu v^mu v^nu (zero for a null ray)."""
    del t, ph
    g = kerr_metric_matrix(r, th, m, spin)
    v = np.array([vt, vr, vth, vph], dtype=float)
    return float(v @ g @ v)


def kerr_killing_energy(y: np.ndarray, m: float, spin: float) -> float:
    """Return E = -p_t for a Kerr geodesic state."""
    _, r, th, _, vt, _, _, vph = (float(x) for x in y)
    g = kerr_metric_matrix(r, th, m, spin)
    return float(-g[I_T, I_T] * vt - g[I_T, I_PH] * vph)


def kerr_axial_angular_momentum(y: np.ndarray, m: float, spin: float) -> float:
    """Return L = p_phi for a Kerr geodesic state."""
    _, r, th, _, vt, _, _, vph = (float(x) for x in y)
    g = kerr_metric_matrix(r, th, m, spin)
    return float(g[I_PH, I_PH] * vph + g[I_T, I_PH] * vt)


def renormalize_kerr_momentum_from_null_constraint(
    y: np.ndarray, m: float, spin: float, preserve_vr_sign: bool = True
) -> np.ndarray:
    r"""Enforce g(v,v)=0 by adjusting v^r while preserving its sign."""
    _, r, th, _, vt, vr, vth, vph = (float(x) for x in y)
    g = kerr_metric_matrix(r, th, m, spin)
    g_rr = float(g[1, 1])
    if abs(g_rr) < 1e-30 or not math.isfinite(g_rr):
        return y
    other = (
        g[0, 0] * vt * vt
        + 2.0 * g[0, 3] * vt * vph
        + g[2, 2] * vth * vth
        + g[3, 3] * vph * vph
    )
    inner = -float(other) / g_rr
    if inner < 0.0 and inner > -1e-8:
        inner = 0.0
    if inner < 0.0 or not math.isfinite(inner):
        return y
    sign = 1.0 if vr >= 0.0 else -1.0
    if not preserve_vr_sign:
        sign = 1.0
    y2 = y.copy()
    y2[5] = sign * math.sqrt(inner)
    return y2


def trace_kerr_null_geodesic(
    x0: np.ndarray,
    v0: np.ndarray,
    m: float,
    spin: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    r_horizon_epsilon: float = 1e-3,
    *,
    store_samples: bool = True,
    sample_stride: int = 1,
) -> GeodesicTraceResult:
    """Trace one Kerr null geodesic with RK4 in affine parameter."""
    y = np.concatenate([x0.astype(float), v0.astype(float)], axis=0)
    y = renormalize_kerr_momentum_from_null_constraint(y, m, spin)
    r_cap = kerr_outer_horizon(m, spin) + r_horizon_epsilon

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
        t, r, th, ph, vt, vr, vth, vph = (float(x) for x in y)
        if not all(math.isfinite(x) for x in (t, r, th, ph, vt, vr, vth, vph)):
            status = RayStatus.NUMERICAL_ERROR
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
        if store_samples and (step_idx % max(sample_stride, 1) == 0):
            r_samples.append(r)
            t_samples.append(t)
            theta_samples.append(th)

        y = rk4_step(null_geodesic_kerr_first_order, lam, y, dlambda, m, spin)
        if step_idx % 4 == 0:
            y = renormalize_kerr_momentum_from_null_constraint(y, m, spin)
        lam += dlambda
        steps_taken = step_idx + 1
    else:
        termination_r = float(y[1]) if len(y) > 1 else float("nan")
        termination_lambda = lam

    if status == RayStatus.MAX_STEPS:
        termination_r = float(y[1]) if len(y) > 1 else float("nan")
        termination_lambda = lam
        r = float(y[1]) if len(y) > 1 else float("nan")
        r_min = min(r_min, r) if math.isfinite(r) else r_min

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
