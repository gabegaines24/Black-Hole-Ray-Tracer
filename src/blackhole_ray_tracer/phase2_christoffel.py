"""Schwarzschild metric in spherical coordinates: Christoffel symbols for geodesic integration.

Signature (-+++). x^0=t, x^1=r, x^2=theta, x^3=phi. f = 1 - 2M/r.
"""

from __future__ import annotations

import math

import numpy as np

# index names: t=0, r=1, theta=2, phi=3
I_T, I_R, I_TH, I_PH = 0, 1, 2, 3


def f_schwarzschild(r: float, m: float) -> float:
    return 1.0 - 2.0 * m / r


def christoffel_schwarzschild(mu: int, a: int, b: int, r: float, th: float, m: float) -> float:
    r"""Return symmetric lower pair \Gamma^\mu_{ab} = \Gamma^\mu_{ba} for Schwarzschild."""
    if r <= 2.0 * m or not (math.isfinite(r) and math.isfinite(th)):
        return 0.0
    f = f_schwarzschild(r, m)
    s = math.sin(th)
    c = math.cos(th)
    s2 = s * s
    r2 = r * r
    m_r2f = m / (r2 * f)

    # Enforce a <= b for lookup
    if a > b:
        a, b = b, a

    if mu == I_T:
        if a == I_T and b == I_R:
            return m_r2f
        return 0.0

    if mu == I_R:
        if a == I_T and b == I_T:
            return m * f / r2
        if a == I_R and b == I_R:
            return m / (r2 * f)
        if a == I_TH and b == I_TH:
            return -r * f
        if a == I_PH and b == I_PH:
            return -r * f * s2
        return 0.0

    if mu == I_TH:
        if a == I_R and b == I_TH:
            return 1.0 / r
        if a == I_PH and b == I_PH:
            return -s * c
        return 0.0

    if mu == I_PH:
        if a == I_R and b == I_PH:
            return 1.0 / r
        if a == I_TH and b == I_PH:
            return c / s if abs(s) > 1e-12 else 0.0
        return 0.0

    return 0.0


def geodesic_acceleration_schwarzschild(y: np.ndarray, m: float) -> np.ndarray:
    """Return dv^mu/d\\lambda = -\\sum_{a,b} \\Gamma^mu_{a b} v^a v^b."""
    t, r, th, ph, vt, vr, vth, vph = (float(x) for x in y)
    v = (vt, vr, vth, vph)
    acc = np.zeros(4, dtype=float)
    for mu in range(4):
        s = 0.0
        for a in range(4):
            for b in range(4):
                s += christoffel_schwarzschild(mu, a, b, r, th, m) * v[a] * v[b]
        acc[mu] = -s
    return acc


def null_geodesic_first_order(_lam: float, y: np.ndarray, m: float) -> np.ndarray:
    """First-order ODE: y = (x^mu, v^mu) with x = (t,r,theta,phi)."""
    t, r, th, ph, vt, vr, vth, vph = (float(x) for x in y)
    dpos = np.array([vt, vr, vth, vph], dtype=float)
    dvel = geodesic_acceleration_schwarzschild(y, m)
    return np.concatenate([dpos, dvel])
