"""Kerr metric in Boyer-Lindquist (BL) coordinates: Christoffel symbols and ODE RHS.

Signature (-+++).  Coordinates x^0=t, x^1=r, x^2=θ, x^3=φ.
Setting spin a=0 recovers the Schwarzschild metric exactly.

References
----------
- Bardeen, Press & Teukolsky (1972), ApJ 178, 347
- MTW §33, Chandrasekhar (1983) Ch. 6
"""

from __future__ import annotations

import math

__all__ = [
    "kerr_sigma",
    "kerr_delta",
    "geodesic_acceleration_kerr",
    "kerr_null_geodesic_rhs",
    "kerr_conserved",
    "kerr_null_invariant",
    "renormalize_vr_kerr",
]

import numpy as np

I_T, I_R, I_TH, I_PH = 0, 1, 2, 3


# ── metric helpers ────────────────────────────────────────────────────────────

def kerr_sigma(r: float, theta: float, a: float) -> float:
    """Σ = r² + a² cos²θ."""
    c = math.cos(theta)
    return r * r + a * a * c * c


def kerr_delta(r: float, m: float, a: float) -> float:
    """Δ = r² − 2Mr + a²."""
    return r * r - 2.0 * m * r + a * a


def _kerr_geometry(r: float, theta: float, m: float, a: float) -> dict:
    """Compute all Kerr metric quantities needed for Christoffel symbols.

    Returns a dict containing metric components, their r/θ derivatives,
    and the inverse metric (upper-index) components.
    """
    if not (math.isfinite(r) and math.isfinite(theta) and r > 0.0):
        return {}

    s = math.sin(theta)
    c = math.cos(theta)
    s2 = s * s
    sc = s * c
    a2 = a * a

    Sig = r * r + a2 * c * c            # Σ
    Dlt = r * r - 2.0 * m * r + a2     # Δ
    A = r * r + a2                      # r² + a²  (shorthand)
    Sig2 = Sig * Sig
    Dlt2 = Dlt * Dlt

    # ── metric components ────────────────────────────────────────────────────
    # Ξ = A·Σ + 2Mra²sin²θ  (appears in g_φφ numerator)
    Xi = A * Sig + 2.0 * m * r * a2 * s2

    g_tt   = -(Sig - 2.0 * m * r) / Sig
    g_tph  = -2.0 * m * r * a * s2 / Sig
    g_rr   = Sig / Dlt
    g_thth = Sig
    g_phph = Xi * s2 / Sig

    # ── inverse metric (t-φ block inversion, trivial r,θ) ────────────────────
    # det(t-φ block) = g_tt·g_φφ − g_tφ² = −Δ·sin²θ
    det_tph = -Dlt * s2
    if abs(det_tph) < 1e-40:
        # degenerate: on the axis (s=0) or at the horizon (Δ=0)
        return {}

    gup_tt   =  g_phph / det_tph
    gup_tph  = -g_tph  / det_tph
    gup_phph =  g_tt   / det_tph
    gup_rr   =  Dlt / Sig
    gup_thth =  1.0 / Sig

    # ── metric derivatives (∂_r and ∂_θ) ─────────────────────────────────────
    dSig_dr  = 2.0 * r
    dSig_dth = -2.0 * a2 * sc

    # g_tt = −(Σ−2Mr)/Σ
    dg_tt_dr  = 2.0 * m * (a2 * c * c - r * r) / Sig2
    dg_tt_dth = 4.0 * m * r * a2 * sc / Sig2

    # g_tφ = −2Mra·sin²θ/Σ
    dg_tph_dr  =  2.0 * m * a * s2 * (r * r - a2 * c * c) / Sig2
    dg_tph_dth = -4.0 * m * r * a * A * sc / Sig2   # uses Σ + a²s² = A

    # g_rr = Σ/Δ
    dg_rr_dr  = (2.0 * r * Dlt - Sig * 2.0 * (r - m)) / Dlt2
    dg_rr_dth = -2.0 * a2 * sc / Dlt

    # g_θθ = Σ
    dg_thth_dr  = 2.0 * r
    dg_thth_dth = -2.0 * a2 * sc

    # g_φφ = s²·Ξ/Σ
    # ∂_r Ξ = 2r(Σ+A) + 2Ma²s²
    dXi_dr  = 2.0 * r * (Sig + A) + 2.0 * m * a2 * s2
    # ∂_θ Ξ = −2a²sc·Δ
    dXi_dth = -2.0 * a2 * sc * Dlt

    # ∂_r g_φφ = s²(Σ·∂_r Ξ − 2r·Ξ)/Σ²
    dg_phph_dr  = s2 * (Sig * dXi_dr - 2.0 * r * Xi) / Sig2
    # ∂_θ g_φφ = 2sc/Σ² · (Ξ·A − Δ·Σ·a²·s²)   uses Σ+a²s²=A
    dg_phph_dth = 2.0 * sc / Sig2 * (Xi * A - Dlt * Sig * a2 * s2)

    return {
        "s": s, "c": c, "s2": s2, "sc": sc,
        "Sig": Sig, "Dlt": Dlt, "A": A,
        "g_tt": g_tt, "g_tph": g_tph, "g_rr": g_rr,
        "g_thth": g_thth, "g_phph": g_phph,
        "gup_tt": gup_tt, "gup_tph": gup_tph, "gup_phph": gup_phph,
        "gup_rr": gup_rr, "gup_thth": gup_thth,
        "dSig_dr": dSig_dr, "dSig_dth": dSig_dth,
        "dg_tt_dr": dg_tt_dr, "dg_tt_dth": dg_tt_dth,
        "dg_tph_dr": dg_tph_dr, "dg_tph_dth": dg_tph_dth,
        "dg_rr_dr": dg_rr_dr, "dg_rr_dth": dg_rr_dth,
        "dg_thth_dr": dg_thth_dr, "dg_thth_dth": dg_thth_dth,
        "dg_phph_dr": dg_phph_dr, "dg_phph_dth": dg_phph_dth,
    }


# ── geodesic acceleration ─────────────────────────────────────────────────────

def geodesic_acceleration_kerr(y: np.ndarray, m: float, a: float) -> np.ndarray:
    r"""Return dv^μ/dλ = −Γ^μ_{αβ} v^α v^β for null geodesics in Kerr.

    Uses the analytic Christoffel symbols derived from the BL metric.
    """
    t_, r, th, ph, vt, vr, vth, vph = (float(x) for x in y)

    if not all(math.isfinite(x) for x in (r, th, vt, vr, vth, vph)):
        return np.zeros(4)

    geo = _kerr_geometry(r, th, m, a)
    if not geo:
        return np.zeros(4)

    # Unpack geometry
    gup_rr   = geo["gup_rr"]
    gup_thth = geo["gup_thth"]
    gup_tt   = geo["gup_tt"]
    gup_tph  = geo["gup_tph"]
    gup_phph = geo["gup_phph"]

    dg_tt_dr    = geo["dg_tt_dr"]
    dg_tt_dth   = geo["dg_tt_dth"]
    dg_tph_dr   = geo["dg_tph_dr"]
    dg_tph_dth  = geo["dg_tph_dth"]
    dg_rr_dr    = geo["dg_rr_dr"]
    dg_rr_dth   = geo["dg_rr_dth"]
    dg_thth_dr  = geo["dg_thth_dr"]
    dg_thth_dth = geo["dg_thth_dth"]
    dg_phph_dr  = geo["dg_phph_dr"]
    dg_phph_dth = geo["dg_phph_dth"]

    # ── Non-zero Γ^r_{αβ} (diagonal inverse for r) ───────────────────────────
    # Γ^r_{αβ} = (g^{rr}/2)(∂_α g_{rβ} + ∂_β g_{rα} − ∂_r g_{αβ})
    # simplifies because g_{rα}=0 for α≠r:
    #   Γ^r_{rr}  = (g^{rr}/2) ∂_r g_{rr}
    #   Γ^r_{rθ}  = (g^{rr}/2) ∂_θ g_{rr}
    #   Γ^r_{AB}  = −(g^{rr}/2) ∂_r g_{AB}  for A,B ∈ {t,θ,φ}
    gr = gup_rr * 0.5
    G_r_rr   =  gr * dg_rr_dr
    G_r_rth  =  gr * dg_rr_dth
    G_r_tt   = -gr * dg_tt_dr
    G_r_tph  = -gr * dg_tph_dr
    G_r_thth = -gr * dg_thth_dr
    G_r_phph = -gr * dg_phph_dr

    # ── Non-zero Γ^θ_{αβ} ────────────────────────────────────────────────────
    gth = gup_thth * 0.5
    G_th_thth =  gth * dg_thth_dth
    G_th_thr  =  gth * dg_thth_dr     # Γ^θ_{θr} = Γ^θ_{rθ}
    G_th_tt   = -gth * dg_tt_dth
    G_th_tph  = -gth * dg_tph_dth
    G_th_rr   = -gth * dg_rr_dth
    G_th_phph = -gth * dg_phph_dth

    # ── Non-zero Γ^t_{αβ} (t-φ off-diagonal block) ───────────────────────────
    # Γ^t_{αβ} = (1/2)(g^{tt}(∂_α g_{tβ} + ∂_β g_{tα}) + g^{tφ}(∂_α g_{φβ} + ∂_β g_{φα}))
    # Non-zero pairs: (t,r), (t,θ), (r,φ), (θ,φ)
    G_t_tr   = 0.5 * (gup_tt * dg_tt_dr    + gup_tph * dg_tph_dr)
    G_t_tth  = 0.5 * (gup_tt * dg_tt_dth   + gup_tph * dg_tph_dth)
    G_t_rph  = 0.5 * (gup_tt * dg_tph_dr   + gup_tph * dg_phph_dr)
    G_t_thph = 0.5 * (gup_tt * dg_tph_dth  + gup_tph * dg_phph_dth)

    # ── Non-zero Γ^φ_{αβ} ────────────────────────────────────────────────────
    G_ph_tr   = 0.5 * (gup_tph * dg_tt_dr    + gup_phph * dg_tph_dr)
    G_ph_tth  = 0.5 * (gup_tph * dg_tt_dth   + gup_phph * dg_tph_dth)
    G_ph_rph  = 0.5 * (gup_tph * dg_tph_dr   + gup_phph * dg_phph_dr)
    G_ph_thph = 0.5 * (gup_tph * dg_tph_dth  + gup_phph * dg_phph_dth)

    # ── a^μ = −Γ^μ_{αβ} v^α v^β  (Γ symmetric ⟹ cross-terms factor of 2) ───
    at = -(
        2.0 * G_t_tr   * vt * vr  +
        2.0 * G_t_tth  * vt * vth +
        2.0 * G_t_rph  * vr * vph +
        2.0 * G_t_thph * vth * vph
    )
    ar = -(
        G_r_tt   * vt  * vt  +
        2.0 * G_r_tph  * vt  * vph +
        G_r_rr   * vr  * vr  +
        2.0 * G_r_rth  * vr  * vth +
        G_r_thth * vth * vth +
        G_r_phph * vph * vph
    )
    ath = -(
        G_th_tt   * vt  * vt  +
        2.0 * G_th_tph  * vt  * vph +
        G_th_rr   * vr  * vr  +
        2.0 * G_th_thr  * vr  * vth +
        G_th_thth * vth * vth +
        G_th_phph * vph * vph
    )
    aph = -(
        2.0 * G_ph_tr   * vt  * vr  +
        2.0 * G_ph_tth  * vt  * vth +
        2.0 * G_ph_rph  * vr  * vph +
        2.0 * G_ph_thph * vth * vph
    )

    return np.array([at, ar, ath, aph], dtype=float)


def kerr_null_geodesic_rhs(_lam: float, y: np.ndarray, m: float, a: float) -> np.ndarray:
    """First-order ODE: dy/dλ for Kerr null geodesic.

    y = (t, r, θ, φ, v^t, v^r, v^θ, v^φ)
    Reuse from Python callers via `phase1.rk4_step`.
    """
    _, _, _, _, vt, vr, vth, vph = (float(x) for x in y)
    dpos = np.array([vt, vr, vth, vph], dtype=float)
    dvel = geodesic_acceleration_kerr(y, m, a)
    return np.concatenate([dpos, dvel])


# ── conserved quantities ──────────────────────────────────────────────────────

def kerr_conserved(y: np.ndarray, m: float, a: float) -> tuple[float, float]:
    """Return (E, L): energy and z-angular momentum along the geodesic.

    E = −g_{tμ} v^μ = −g_tt v^t − g_tφ v^φ
    L =  g_{φμ} v^μ =  g_φφ v^φ + g_tφ v^t
    """
    _, r, th, _, vt, _, _, vph = (float(x) for x in y)
    geo = _kerr_geometry(r, th, m, a)
    if not geo:
        return float("nan"), float("nan")
    E = -geo["g_tt"] * vt - geo["g_tph"] * vph
    L =  geo["g_phph"] * vph + geo["g_tph"] * vt
    return E, L


def kerr_null_invariant(y: np.ndarray, m: float, a: float) -> float:
    """Return g_{μν} v^μ v^ν (should be 0 for a null geodesic)."""
    _, r, th, _, vt, vr, vth, vph = (float(x) for x in y)
    geo = _kerr_geometry(r, th, m, a)
    if not geo:
        return float("nan")
    return (
        geo["g_tt"] * vt * vt +
        2.0 * geo["g_tph"] * vt * vph +
        geo["g_rr"] * vr * vr +
        geo["g_thth"] * vth * vth +
        geo["g_phph"] * vph * vph
    )


def renormalize_vr_kerr(y: np.ndarray, m: float, a: float) -> np.ndarray:
    r"""Project v^r onto the Kerr null cone; preserves sign of v^r."""
    _, r, th, _, vt, vr, vth, vph = (float(x) for x in y)
    geo = _kerr_geometry(r, th, m, a)
    if not geo:
        return y
    g_tt   = geo["g_tt"]
    g_tph  = geo["g_tph"]
    g_rr   = geo["g_rr"]
    g_thth = geo["g_thth"]
    g_phph = geo["g_phph"]

    if g_rr <= 0.0 or not math.isfinite(g_rr):
        return y
    # g_rr (v^r)² = −(g_tt v^t² + 2g_tφ v^t v^φ + g_θθ (v^θ)² + g_φφ (v^φ)²)
    inner = -(g_tt * vt * vt + 2.0 * g_tph * vt * vph + g_thth * vth * vth + g_phph * vph * vph)
    if inner < 0.0 and inner > -1e-6:
        inner = 0.0
    if inner < 0.0 or not math.isfinite(inner):
        return y
    sgn = 1.0 if vr >= 0.0 else -1.0
    y2 = y.copy()
    y2[5] = sgn * math.sqrt(inner / g_rr)
    return y2
