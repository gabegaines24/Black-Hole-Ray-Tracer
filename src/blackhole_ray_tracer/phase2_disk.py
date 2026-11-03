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

The Python path interpolates aligned pre-step and post-step samples. The CPU and
CUDA batch paths emit the same first-crossing observables directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True, slots=True)
class DiskConfig:
    """Parameters for the thin equatorial disk."""

    r_inner: float = 6.0
    r_outer: float = 20.0
    inner_color: tuple[float, float, float] = (1.0, 0.6, 0.2)
    outer_color: tuple[float, float, float] = (0.8, 0.3, 0.05)
    redshift_scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.r_inner) or self.r_inner <= 0.0:
            raise ValueError("Disk inner radius must be positive and finite.")
        if not math.isfinite(self.r_outer) or self.r_outer <= self.r_inner:
            raise ValueError("Disk outer radius must exceed inner radius.")
        for name, color in (
            ("inner_color", self.inner_color),
            ("outer_color", self.outer_color),
        ):
            if len(color) != 3 or any(not math.isfinite(channel) or not 0.0 <= channel <= 1.0 for channel in color):
                raise ValueError(f"{name} must contain three finite channels in [0, 1].")
        if not math.isfinite(self.redshift_scale) or self.redshift_scale < 0.0:
            raise ValueError("Disk redshift scale must be finite and non-negative.")

def kerr_equatorial_isco_r(m: float, spin: float) -> float:
    """Prograde equatorial ISCO radius for Kerr (spin = dimensionless a/M).

    Reduces to 6M when spin=0 (Schwarzschild limit).
    """
    if abs(spin) < 1e-12:
        return 6.0 * m
    chi = spin
    z1 = 1.0 + (1.0 - chi * chi) ** (1.0 / 3.0) * (
        (1.0 + chi) ** (1.0 / 3.0) + (1.0 - chi) ** (1.0 / 3.0)
    )
    z2 = math.sqrt(3.0 * chi * chi + z1 * z1)
    inner = (3.0 - z1) * (3.0 + z1 - 2.0 * z2)
    if inner < 0.0 and inner > -1e-9:
        inner = 0.0
    if chi >= 0.0:
        r_over_m = 3.0 + z2 - math.sqrt(max(inner, 0.0))
    else:
        r_over_m = 3.0 + z2 + math.sqrt(max(inner, 0.0))
    return r_over_m * m


def disk_config_for_mass_spin(
    m: float = 1.0,
    spin: float = 0.0,
    *,
    r_outer: float | None = None,
) -> DiskConfig:
    """Build a disk config with spin-dependent inner edge at the equatorial ISCO."""
    r_inner = kerr_equatorial_isco_r(m, spin)
    outer = r_outer if r_outer is not None else max(20.0 * m, r_inner + 2.0 * m)
    return DiskConfig(r_inner=r_inner, r_outer=outer)


class DiskHit(NamedTuple):
    hit: bool
    r_cross: float             # r at the equatorial crossing (may be nan)
    z_factor: float            # (1+z)^-4 intensity scaling; 1.0 if no doppler


class EquatorialCrossing(NamedTuple):
    """Interpolated state at first strict equatorial crossing."""

    r_cross: float
    vt_cross: float
    vph_cross: float


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
    if not math.isfinite(ratio):
        return 1.0
    if ratio <= 0.0:
        return 0.0
    # Specific intensity scales as ν^4 (radiative transfer); use ratio^4.
    if scale == 0.0:
        return 0.0
    try:
        intensity = float(ratio**4) * scale
    except OverflowError:
        intensity = 10.0
    if not math.isfinite(intensity):
        return 10.0 if intensity > 0.0 else 0.0
    return max(0.0, min(intensity, 10.0))


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
    # Apply Doppler factor to brightness (clamp to [0,1]).
    br = max(0.0, min(base_r * z_factor, 1.0))
    bg = max(0.0, min(base_g * z_factor, 1.0))
    bb = max(0.0, min(base_b * z_factor, 1.0))
    return (br, bg, bb)


def _interpolate_equatorial_crossing(
    samples_theta: list[float],
    samples_r: list[float],
    samples_vt: list[float] | None = None,
    samples_vphi: list[float] | None = None,
) -> EquatorialCrossing | None:
    if len(samples_theta) != len(samples_r):
        raise ValueError("theta and radius samples must have equal lengths.")
    if (samples_vt is None) != (samples_vphi is None):
        raise ValueError("velocity samples must be provided together.")
    if samples_vt is not None and samples_vphi is not None:
        if (
            len(samples_vt) != len(samples_theta)
            or len(samples_vphi) != len(samples_theta)
        ):
            raise ValueError("velocity samples must match theta sample length.")

    half_pi = math.pi / 2.0
    for i in range(1, len(samples_theta)):
        th0 = samples_theta[i - 1] - half_pi
        th1 = samples_theta[i] - half_pi
        if th0 * th1 >= 0.0:
            continue
        alpha = abs(th0) / (abs(th0) + abs(th1) + 1e-30)
        vt_cross = float("nan")
        vph_cross = float("nan")
        if samples_vt is not None and samples_vphi is not None:
            vt_cross = samples_vt[i - 1] * (1.0 - alpha) + samples_vt[i] * alpha
            vph_cross = (
                samples_vphi[i - 1] * (1.0 - alpha) + samples_vphi[i] * alpha
            )
        return EquatorialCrossing(
            r_cross=samples_r[i - 1] * (1.0 - alpha) + samples_r[i] * alpha,
            vt_cross=vt_cross,
            vph_cross=vph_cross,
        )
    return None


def detect_equatorial_crossing(
    samples_theta: list[float],
    samples_r: list[float],
) -> tuple[bool, float]:
    """Return whether first strict crossing exists and its interpolated radius."""
    crossing = _interpolate_equatorial_crossing(samples_theta, samples_r)
    if crossing is None:
        return False, float("nan")
    return True, crossing.r_cross


def detect_equatorial_crossing_state(
    samples_theta: list[float],
    samples_r: list[float],
    samples_vt: list[float],
    samples_vphi: list[float],
) -> EquatorialCrossing | None:
    """Return first crossing radius and interpolated photon velocities."""
    return _interpolate_equatorial_crossing(
        samples_theta, samples_r, samples_vt, samples_vphi
    )
