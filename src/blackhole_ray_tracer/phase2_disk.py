"""Thin Keplerian accretion disk: equatorial plane intersection + Doppler redshift.

Physics
-------
The disk lies in the equatorial plane (θ = π/2) and extends from the ISCO at
r_ISCO = 6M (for Schwarzschild) to an outer edge r_disk_outer.

Disk hit detection
------------------
A ray crosses the equatorial plane whenever its θ coordinate passes through π/2.
We detect sign changes in (θ - π/2) between successive integration steps and
linearly interpolate the crossing radius r_cross.

Doppler redshift (simple approximation)
-----------------------------------------
For a circular Keplerian orbit at radius r in Schwarzschild geometry, the
angular velocity is Ω = √(M/r³) and the azimuthal four-velocity of the emitter
is  u^μ = (u^t, 0, 0, u^φ) where:
    u^t = 1 / √(f − r² Ω² sin²θ),   θ = π/2
    u^φ = Ω u^t
The photon four-momentum p_μ is conserved along the geodesic; the observed
energy ratio (redshift factor 1+z) is:
    E_obs / E_emit = (−p_μ u^μ_obs) / (−p_μ u^μ_emit)
We approximate the observer as a static observer (u^μ_obs ∝ (1,0,0,0)) and
compute the emitter contraction analytically from the conserved Killing energy
E = −p_t f and the impact parameter L = p_φ.

Since we don't accumulate samples in the batch path, the module provides a
post-processor that works from the final ray state and crossing information
passed through from the Python render loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True, slots=True)
class DiskConfig:
    """Parameters for the thin equatorial disk."""

    r_inner: float = 6.0    # ISCO = 6M for Schwarzschild
    r_outer: float = 20.0
    # Simple two-color gradient: inner ring is hotter (bluer)
    inner_color: tuple[float, float, float] = (1.0, 0.6, 0.2)   # warm orange
    outer_color: tuple[float, float, float] = (0.8, 0.3, 0.05)  # dim red
    redshift_scale: float = 1.0   # multiply (1+z) effect on brightness


class DiskHit(NamedTuple):
    hit: bool
    r_cross: float             # r at the equatorial crossing (may be nan)
    z_factor: float            # (1+z)^-4 intensity scaling; 1.0 if no doppler


def disk_hit_from_equatorial_crossing(
    r_cross: float,
    disk: DiskConfig,
    m: float,
    vt: float,     # v^t at crossing (coordinate, not unit)
    vph: float,    # v^phi at crossing
) -> DiskHit:
    """Determine if the crossing is inside the disk and compute the redshift factor.

    Parameters
    ----------
    r_cross : float
        Interpolated equatorial crossing radius.
    disk : DiskConfig
    m : float
        Black hole mass.
    vt, vph : float
        Null ray coordinate four-velocity components at crossing.
    """
    if math.isnan(r_cross) or r_cross < disk.r_inner or r_cross > disk.r_outer:
        return DiskHit(hit=False, r_cross=r_cross, z_factor=1.0)

    # Doppler redshift: observer is static at large r; emitter is Keplerian.
    z_factor = _intensity_factor(r_cross, m, vt, vph, disk.redshift_scale)
    return DiskHit(hit=True, r_cross=r_cross, z_factor=z_factor)


def _intensity_factor(
    r: float, m: float, vt: float, vph: float, scale: float
) -> float:
    """Return intensity scaling = [(1+z)^-4] due to Doppler effect.

    We use a single-crossing approximation.  The energy ratio E_obs/E_emit is:
        ratio = (f * vt) / (f * vt - r² Ω vph)
    where Ω = √(M/r³) is the Keplerian angular velocity.
    Higher ratio = blueshift (approaching), lower = redshift.
    """
    if r <= 2.0 * m or not math.isfinite(r) or not math.isfinite(vt) or not math.isfinite(vph):
        return 1.0
    f = 1.0 - 2.0 * m / r
    omega = math.sqrt(m / (r ** 3))  # Keplerian angular velocity
    denom = f * vt - r ** 2 * omega * vph
    if not math.isfinite(denom) or abs(denom) < 1e-30:
        return 1.0
    ratio = (f * vt) / denom
    if ratio <= 0.0:
        return 0.0
    # Specific intensity scales as ν^4 (radiative transfer); use ratio^4
    return min(float(ratio ** 4) * scale, 10.0)


def disk_color_at_r(
    r_cross: float, z_factor: float, disk: DiskConfig
) -> tuple[float, float, float]:
    """Return RGB [0,1] for a disk hit, blending inner/outer colors by radius."""
    t = (r_cross - disk.r_inner) / max(disk.r_outer - disk.r_inner, 1e-6)
    t = max(0.0, min(1.0, t))
    ic, oc = disk.inner_color, disk.outer_color
    base_r = ic[0] * (1.0 - t) + oc[0] * t
    base_g = ic[1] * (1.0 - t) + oc[1] * t
    base_b = ic[2] * (1.0 - t) + oc[2] * t
    # Apply Doppler factor to brightness (clamp to [0,1])
    br = min(base_r * z_factor, 1.0)
    bg = min(base_g * z_factor, 1.0)
    bb = min(base_b * z_factor, 1.0)
    return (br, bg, bb)


def detect_equatorial_crossing(
    samples_theta: list[float],
    samples_r: list[float],
) -> tuple[bool, float]:
    """Search stored theta/r samples for the first equatorial crossing.

    Returns (found, r_cross).  Only the first crossing is returned.
    """
    half_pi = math.pi / 2.0
    for i in range(1, len(samples_theta)):
        th0 = samples_theta[i - 1] - half_pi
        th1 = samples_theta[i] - half_pi
        if th0 * th1 < 0.0:
            # Linear interpolation
            alpha = abs(th0) / (abs(th0) + abs(th1) + 1e-30)
            r_cross = samples_r[i - 1] * (1.0 - alpha) + samples_r[i] * alpha
            return True, r_cross
    return False, float("nan")
