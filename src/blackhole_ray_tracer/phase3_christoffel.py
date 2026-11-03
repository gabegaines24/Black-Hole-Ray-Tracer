"""Kerr metric in Boyer-Lindquist coordinates for Phase 3 geodesics.

Coordinates follow the Phase 2 convention: x^0=t, x^1=r, x^2=theta,
x^3=phi, signature (-+++).  The ``spin`` parameter is dimensionless a/M;
``spin=0`` reduces to Schwarzschild.
"""

from __future__ import annotations

import math

import numpy as np

I_T, I_R, I_TH, I_PH = 0, 1, 2, 3


def kerr_spin_a(m: float, spin: float) -> float:
    """Return dimensional Kerr spin parameter a from dimensionless a/M."""
    if m <= 0.0 or not math.isfinite(m):
        raise ValueError("Black hole mass m must be positive and finite.")
    if abs(spin) > 1.0 or not math.isfinite(spin):
        raise ValueError("Kerr spin must be finite with |spin| <= 1.")
    return spin * m


def kerr_sigma(
    r: float,
    th: float,
    m: float | None = None,
    spin: float | None = None,
    *,
    a: float | None = None,
) -> float:
    """Return Σ for either dimensional ``a`` or ``(m, spin)`` inputs."""
    if a is not None:
        dimensional_a = a
    elif spin is None:
        dimensional_a = 0.0 if m is None else m
    else:
        if m is None:
            raise ValueError("Mass m is required with dimensionless spin.")
        dimensional_a = kerr_spin_a(m, spin)
    c = math.cos(th)
    return r * r + dimensional_a * dimensional_a * c * c

def kerr_delta(
    r: float,
    m: float,
    spin: float | None = None,
    *,
    a: float | None = None,
) -> float:
    """Return Δ for either dimensional ``a`` or dimensionless ``spin``."""
    dimensional_a = a if a is not None else (
        0.0 if spin is None else kerr_spin_a(m, spin)
    )
    return r * r - 2.0 * m * r + dimensional_a * dimensional_a


def kerr_outer_horizon(m: float, spin: float) -> float:
    """Return r_+ = M + sqrt(M^2 - a^2)."""
    a = kerr_spin_a(m, spin)
    return m + math.sqrt(max(m * m - a * a, 0.0))


def kerr_metric_matrix(r: float, th: float, m: float, spin: float) -> np.ndarray:
    """Return covariant Kerr metric g_mu_nu in BL coordinates."""
    a = kerr_spin_a(m, spin)
    s = math.sin(th)
    c = math.cos(th)
    s2 = s * s
    sigma = r * r + a * a * c * c
    delta = r * r - 2.0 * m * r + a * a
    if sigma <= 0.0 or delta == 0.0:
        return np.zeros((4, 4), dtype=float)

    g = np.zeros((4, 4), dtype=float)
    g[I_T, I_T] = -1.0 + 2.0 * m * r / sigma
    g[I_T, I_PH] = g[I_PH, I_T] = -2.0 * m * a * r * s2 / sigma
    g[I_R, I_R] = sigma / delta
    g[I_TH, I_TH] = sigma
    g[I_PH, I_PH] = (r * r + a * a + 2.0 * m * a * a * r * s2 / sigma) * s2
    return g


def kerr_inverse_metric_matrix(r: float, th: float, m: float, spin: float) -> np.ndarray:
    """Return contravariant Kerr metric g^mu_nu using the sparse BL block form."""
    g = kerr_metric_matrix(r, th, m, spin)
    inv = np.zeros((4, 4), dtype=float)
    denom = g[I_T, I_T] * g[I_PH, I_PH] - g[I_T, I_PH] * g[I_T, I_PH]
    if abs(denom) < 1e-30:
        return inv
    inv[I_T, I_T] = g[I_PH, I_PH] / denom
    inv[I_T, I_PH] = inv[I_PH, I_T] = -g[I_T, I_PH] / denom
    inv[I_PH, I_PH] = g[I_T, I_T] / denom
    if abs(g[I_R, I_R]) > 1e-30:
        inv[I_R, I_R] = 1.0 / g[I_R, I_R]
    if abs(g[I_TH, I_TH]) > 1e-30:
        inv[I_TH, I_TH] = 1.0 / g[I_TH, I_TH]
    return inv


def kerr_metric_derivative(
    coord_idx: int, r: float, th: float, m: float, spin: float
) -> np.ndarray:
    """Return partial derivative d_coord g_mu_nu.

    Kerr in BL coordinates is stationary and axisymmetric, so only r and theta
    derivatives are non-zero.
    """
    if coord_idx not in (I_R, I_TH):
        return np.zeros((4, 4), dtype=float)

    a = kerr_spin_a(m, spin)
    s = math.sin(th)
    c = math.cos(th)
    s2 = s * s
    ds2_dth = 2.0 * s * c
    sigma = r * r + a * a * c * c
    delta = r * r - 2.0 * m * r + a * a
    d_sigma_dr = 2.0 * r
    d_sigma_dth = -2.0 * a * a * s * c
    d_delta_dr = 2.0 * r - 2.0 * m

    dg = np.zeros((4, 4), dtype=float)
    if sigma <= 0.0:
        return dg

    if coord_idx == I_R:
        sigma_minus_r_dsigma = sigma - r * d_sigma_dr
        dg[I_T, I_T] = 2.0 * m * sigma_minus_r_dsigma / (sigma * sigma)
        dg[I_T, I_PH] = (
            -2.0 * m * a * s2 * sigma_minus_r_dsigma / (sigma * sigma)
        )
        dg[I_PH, I_T] = dg[I_T, I_PH]
        if abs(delta) > 1e-30:
            dg[I_R, I_R] = (
                d_sigma_dr * delta - sigma * d_delta_dr
            ) / (delta * delta)
        dg[I_TH, I_TH] = d_sigma_dr
        dg[I_PH, I_PH] = s2 * (
            2.0 * r
            + 2.0 * m * a * a * s2 * sigma_minus_r_dsigma / (sigma * sigma)
        )
        return dg

    dg[I_T, I_T] = -2.0 * m * r * d_sigma_dth / (sigma * sigma)
    dg[I_T, I_PH] = -2.0 * m * a * r * (
        ds2_dth / sigma - s2 * d_sigma_dth / (sigma * sigma)
    )
    dg[I_PH, I_T] = dg[I_T, I_PH]
    if abs(delta) > 1e-30:
        dg[I_R, I_R] = d_sigma_dth / delta
    dg[I_TH, I_TH] = d_sigma_dth
    base = r * r + a * a + 2.0 * m * a * a * r * s2 / sigma
    d_base_dth = 2.0 * m * a * a * r * (
        ds2_dth / sigma - s2 * d_sigma_dth / (sigma * sigma)
    )
    dg[I_PH, I_PH] = ds2_dth * base + s2 * d_base_dth
    return dg


def christoffel_kerr(
    mu: int, a_idx: int, b_idx: int, r: float, th: float, m: float, spin: float
) -> float:
    r"""Return symmetric lower pair \Gamma^\mu_{ab} for Kerr BL coordinates."""
    if not (math.isfinite(r) and math.isfinite(th)):
        return 0.0
    if r <= kerr_outer_horizon(m, spin):
        return 0.0
    g_inv = kerr_inverse_metric_matrix(r, th, m, spin)
    d_a = kerr_metric_derivative(a_idx, r, th, m, spin)
    d_b = kerr_metric_derivative(b_idx, r, th, m, spin)

    total = 0.0
    for rho in range(4):
        d_rho = kerr_metric_derivative(rho, r, th, m, spin)
        total += g_inv[mu, rho] * (
            d_a[b_idx, rho] + d_b[a_idx, rho] - d_rho[a_idx, b_idx]
        )
    return 0.5 * total


def geodesic_acceleration_kerr(y: np.ndarray, m: float, spin: float) -> np.ndarray:
    """Return dv^mu/dlambda = -Gamma^mu_ab v^a v^b."""
    _, r, th, _, vt, vr, vth, vph = (float(x) for x in y)
    v = (vt, vr, vth, vph)
    acc = np.zeros(4, dtype=float)
    for mu in range(4):
        total = 0.0
        for a_idx in range(4):
            for b_idx in range(4):
                total += (
                    christoffel_kerr(mu, a_idx, b_idx, r, th, m, spin)
                    * v[a_idx]
                    * v[b_idx]
                )
        acc[mu] = -total
    return acc


def null_geodesic_kerr_first_order(
    _lam: float, y: np.ndarray, m: float, spin: float
) -> np.ndarray:
    """First-order Kerr geodesic ODE: y = (x^mu, v^mu)."""
    _, _, _, _, vt, vr, vth, vph = (float(x) for x in y)
    dpos = np.array([vt, vr, vth, vph], dtype=float)
    dvel = geodesic_acceleration_kerr(y, m, spin)
    return np.concatenate([dpos, dvel])


def _dimensional_a_to_spin(m: float, a: float) -> float:
    if not math.isfinite(m) or m == 0.0:
        return a
    return a / m


def kerr_conserved(
    y: np.ndarray | list[float], m: float, a: float
) -> tuple[float, float]:
    """Return conserved energy and axial angular momentum for dimensional ``a``."""
    state = np.asarray(y, dtype=float)
    _, r, th, _, vt, _, _, vph = (float(value) for value in state)
    g = kerr_metric_matrix(r, th, m, _dimensional_a_to_spin(m, a))
    return (
        float(-g[I_T, I_T] * vt - g[I_T, I_PH] * vph),
        float(g[I_PH, I_PH] * vph + g[I_T, I_PH] * vt),
    )


def kerr_null_invariant(
    y: np.ndarray | list[float], m: float, a: float
) -> float:
    """Return g(v, v) for a state using dimensional ``a``."""
    state = np.asarray(y, dtype=float)
    _, r, th, _, vt, vr, vth, vph = (float(value) for value in state)
    g = kerr_metric_matrix(r, th, m, _dimensional_a_to_spin(m, a))
    velocity = np.array([vt, vr, vth, vph], dtype=float)
    return float(velocity @ g @ velocity)


def renormalize_vr_kerr(
    y: np.ndarray | list[float], m: float, a: float
) -> np.ndarray:
    """Adjust ``v^r`` onto the null cone using dimensional ``a``."""
    from .phase3_geodesic import renormalize_kerr_momentum_from_null_constraint

    state = np.asarray(y, dtype=float)
    return renormalize_kerr_momentum_from_null_constraint(
        state, m, _dimensional_a_to_spin(m, a)
    )


def kerr_null_geodesic_rhs(
    lam: float, y: np.ndarray | list[float], m: float, a: float
) -> np.ndarray:
    """Return the first-order RHS using dimensional ``a``."""
    return null_geodesic_kerr_first_order(
        lam,
        np.asarray(y, dtype=float),
        m,
        _dimensional_a_to_spin(m, a),
    )
