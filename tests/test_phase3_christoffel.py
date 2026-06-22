"""Unit tests for the Phase 3 Kerr Christoffel / BL geometry helpers.

Each test checks one well-known analytic identity or limiting value from
Boyer-Lindquist coordinates.  No integration is run; these are pure-function
checks against closed-form results.
"""

from __future__ import annotations

import math

import pytest

from blackhole_ray_tracer.phase3_christoffel import (
    geodesic_acceleration_kerr,
    kerr_conserved,
    kerr_delta,
    kerr_null_geodesic_rhs,
    kerr_null_invariant,
    kerr_sigma,
    renormalize_vr_kerr,
)

M = 1.0


class TestSigmaDelta:
    """Σ = r² + a²cos²θ and Δ = r² - 2Mr + a² identities."""

    def test_sigma_equatorial(self):
        """At θ=π/2, Σ = r²."""
        for r in [3.0, 5.0, 10.0, 50.0]:
            assert kerr_sigma(r, math.pi / 2, a=0.9) == pytest.approx(r**2)

    def test_sigma_polar(self):
        """At θ=0, Σ = r² + a²."""
        a = 0.7
        for r in [3.0, 10.0]:
            assert kerr_sigma(r, 0.0, a=a) == pytest.approx(r**2 + a**2)

    def test_sigma_schwarzschild_limit(self):
        """a=0 → Σ = r² independent of θ."""
        for r in [4.0, 8.0]:
            for th in [0.0, math.pi / 4, math.pi / 2]:
                assert kerr_sigma(r, th, a=0.0) == pytest.approx(r**2)

    def test_delta_schwarzschild_limit(self):
        """a=0 → Δ = r² - 2Mr = r(r - 2M)."""
        for r in [4.0, 8.0, 20.0]:
            assert kerr_delta(r, m=M, a=0.0) == pytest.approx(r**2 - 2 * M * r)

    def test_delta_zero_at_outer_horizon(self):
        """Δ = 0 at r+ = M + √(M²-a²)."""
        a = 0.6
        r_plus = M + math.sqrt(M**2 - a**2)
        assert kerr_delta(r_plus, m=M, a=a) == pytest.approx(0.0, abs=1e-12)

    def test_delta_zero_at_inner_horizon(self):
        """Δ = 0 at r- = M - √(M²-a²)."""
        a = 0.6
        r_minus = M - math.sqrt(M**2 - a**2)
        assert kerr_delta(r_minus, m=M, a=a) == pytest.approx(0.0, abs=1e-12)

    def test_delta_negative_between_horizons(self):
        """Δ < 0 between inner and outer Kerr horizons."""
        a = 0.6
        r_mid = M  # exactly between r+ and r- for a < M
        assert kerr_delta(r_mid, m=M, a=a) < 0.0


class TestConservedQuantities:
    """E = -p_t and L = p_φ in BL coordinates."""

    def _state_equatorial(self, r: float, a: float, vt: float, vphi: float) -> list[float]:
        """Build y = [t, r, θ, φ, vt, vr, vθ, vφ] at the equatorial plane."""
        return [0.0, r, math.pi / 2, 0.0, vt, 0.0, 0.0, vphi]

    def test_conserved_return_tuple(self):
        y = self._state_equatorial(r=10.0, a=0.5, vt=1.0, vphi=0.1)
        result = kerr_conserved(y, m=M, a=0.5)
        assert len(result) == 2

    def test_conserved_positive_energy(self):
        """E > 0 for a photon moving forward in time far from the horizon."""
        y = self._state_equatorial(r=20.0, a=0.0, vt=1.1, vphi=0.05)
        E, _ = kerr_conserved(y, m=M, a=0.0)
        assert E > 0.0

    def test_conserved_schwarzschild_a0(self):
        """For a=0 at equatorial plane, E and L match Schwarzschild metric coefficients."""
        r = 10.0
        vt, vphi = 1.05, 0.08
        y = self._state_equatorial(r=r, a=0.0, vt=vt, vphi=vphi)
        E, L = kerr_conserved(y, m=M, a=0.0)
        # g_tt(Schwarzschild) = -(1 - 2M/r),  g_φφ = r²sin²θ = r² at eq
        gtt = -(1.0 - 2 * M / r)
        gphi = r**2
        E_expected = -gtt * vt
        L_expected = gphi * vphi
        assert E == pytest.approx(E_expected, rel=1e-6)
        assert L == pytest.approx(L_expected, rel=1e-6)


class TestNullInvariant:
    """g_μν v^μ v^ν = 0 for a null vector (up to normalisation)."""

    def test_invariant_finite(self):
        """kerr_null_invariant returns a finite value for any reasonable state."""
        y = [0.0, 10.0, math.pi / 2, 0.0, 1.1, -0.05, 0.01, 0.08]
        val = kerr_null_invariant(y, m=M, a=0.5)
        assert math.isfinite(val)


class TestRenormalise:
    """renormalize_vr_kerr adjusts v^r to restore the null constraint."""

    def test_renormalize_returns_array_of_8(self):
        """renormalize_vr_kerr returns a length-8 state vector."""
        import numpy as np
        y = [0.0, 10.0, math.pi / 2, 0.0, 1.1, -0.5, 0.01, 0.08]
        result = renormalize_vr_kerr(y, m=M, a=0.5)
        assert len(result) == 8
        for v in result:
            assert math.isfinite(float(v))

    def test_renormalize_only_modifies_vr(self):
        """Only y[5] (v^r) should be adjusted; other components unchanged."""
        import numpy as np
        y = [0.0, 10.0, math.pi / 2, 0.0, 1.1, -0.5, 0.01, 0.08]
        result = renormalize_vr_kerr(y, m=M, a=0.5)
        for i in [0, 1, 2, 3, 4, 6, 7]:
            assert float(result[i]) == pytest.approx(y[i])


class TestODERhs:
    """kerr_null_geodesic_rhs / geodesic_acceleration_kerr sanity checks."""

    def test_rhs_returns_8_components(self):
        y = [0.0, 10.0, math.pi / 2, 0.0, 1.1, -0.05, 0.01, 0.08]
        dy = kerr_null_geodesic_rhs(0.0, y, m=M, a=0.5)
        assert len(dy) == 8

    def test_rhs_velocities_match_input(self):
        """First four components of RHS = velocities = y[4:8]."""
        y = [0.0, 10.0, math.pi / 2, 0.0, 1.1, -0.05, 0.01, 0.08]
        dy = kerr_null_geodesic_rhs(0.0, y, m=M, a=0.5)
        for i in range(4):
            assert dy[i] == pytest.approx(y[i + 4])

    def test_rhs_all_finite(self):
        """All components of the RHS must be finite at well-separated r."""
        import numpy as np
        y = [0.0, 15.0, math.pi / 2, 0.0, 1.0, -0.1, 0.0, 0.05]
        dy = kerr_null_geodesic_rhs(0.0, y, m=M, a=0.6)
        assert all(math.isfinite(float(v)) for v in dy), f"Non-finite in RHS: {dy}"

    def test_rhs_schwarzschild_a0_consistency(self):
        """At a=0 the Kerr RHS should return finite values at different radii."""
        y1 = [0.0, 10.0, math.pi / 2, 0.0, 1.1, -0.05, 0.0, 0.08]
        y2 = [0.0, 20.0, math.pi / 2, 0.0, 1.05, -0.02, 0.0, 0.04]
        dy1 = kerr_null_geodesic_rhs(0.0, y1, m=M, a=0.0)
        dy2 = kerr_null_geodesic_rhs(0.0, y2, m=M, a=0.0)
        for v in list(dy1) + list(dy2):
            assert math.isfinite(float(v))
