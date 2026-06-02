"""3D null geodesic integration in Schwarzschild (affine parameter) using RK4 + Christoffel form."""

from __future__ import annotations

import math

import numpy as np

from .phase1 import RayStatus, rk4_step
from .phase2_christoffel import f_schwarzschild, null_geodesic_first_order
from .phase2_types import GeodesicTraceResult


def metric_invariant_schwarzschild(
    t: float, r: float, th: float, ph: float, vt: float, vr: float, vth: float, vph: float, m: float
) -> float:
    """Return g_\\mu\\nu v^\\mu v^\\nu (0 for a physical null direction)."""
    f = f_schwarzschild(r, m)
    s = math.sin(th)
    return -f * vt**2 + vr**2 / f + r**2 * vth**2 + r**2 * s**2 * vph**2


def renormalize_momentum_from_null_constraint(
    y: np.ndarray, m: float, preserve_vr_sign: bool = True
) -> np.ndarray:
    r"""Enforce g(v,v)=0 by adjusting v^r given v^t, v^\\theta, v^\\phi (sign from current v^r)."""
    t, r, th, ph, vt, vr, vth, vph = (float(x) for x in y)
    f = f_schwarzschild(r, m)
    if f <= 0.0 or not math.isfinite(f):
        return y
    s = math.sin(th)
    a = (r**2) * (vth**2 + s**2 * vph**2)
    inner = f**2 * vt**2 - f * a
    if inner < 0.0 and inner > -1e-6:
        inner = 0.0
    if inner < 0.0 or not math.isfinite(inner):
        return y
    sgn = 1.0 if vr >= 0.0 else -1.0
    if not preserve_vr_sign:
        sgn = 1.0
    vr_new = sgn * math.sqrt(inner)
    y2 = y.copy()
    y2[5] = vr_new
    return y2


def trace_null_geodesic_3d(
    x0: np.ndarray,
    v0: np.ndarray,
    m: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    r_horizon_epsilon: float = 1e-3,
    *,
    store_samples: bool = True,
    sample_stride: int = 1,
) -> GeodesicTraceResult:

    y = np.concatenate([x0.astype(float), v0.astype(float)], axis=0)
    y = renormalize_momentum_from_null_constraint(y, m)
    r_h = 2.0 * m
    r_cap = r_h + r_horizon_epsilon

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

        y = rk4_step(null_geodesic_first_order, lam, y, dlambda, m)
        # Periodically project back onto null cone
        if step_idx % 4 == 0:
            y = renormalize_momentum_from_null_constraint(y, m)
        lam += dlambda
        steps_taken = step_idx + 1
    else:
        termination_r = float(y[1]) if len(y) > 1 else float("nan")
        termination_lambda = lam

    if status == RayStatus.MAX_STEPS:
        termination_r = float(y[1]) if len(y) > 1 else float("nan")
        termination_lambda = lam
        t, r, th, ph, vt, vr, vth, vph = (float(x) for x in y)
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
