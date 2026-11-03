"""Tests for Phase 3 Kerr Boyer-Lindquist geodesics and rendering."""

from __future__ import annotations

import numpy as np
import pytest

from blackhole_ray_tracer.phase1 import rk4_step
from blackhole_ray_tracer.phase2_camera import (
    initial_position_observer,
    make_camera_from_config,
    static_observer_null_direction,
)
from blackhole_ray_tracer.phase2_geodesic import trace_null_geodesic_3d
from blackhole_ray_tracer.phase3_camera import (
    make_kerr_camera_from_config,
    zamo_null_direction,
)
from blackhole_ray_tracer.phase3_christoffel import (
    kerr_outer_horizon,
    null_geodesic_kerr_first_order,
)
from blackhole_ray_tracer.phase3_geodesic import (
    kerr_axial_angular_momentum,
    kerr_carter_constant,
    kerr_killing_energy,
    metric_invariant_kerr,
    renormalize_kerr_momentum_from_null_constraint,
    trace_kerr_null_geodesic,
)
from blackhole_ray_tracer.phase3_render import render_kerr_image
from blackhole_ray_tracer.phase3_report import format_phase3_report, render_config_from_preset
from blackhole_ray_tracer.phase2_disk import disk_config_for_mass_spin, kerr_equatorial_isco_r
from blackhole_ray_tracer.phase3_types import KerrRenderConfig


@pytest.mark.parametrize("spin", [-1.01, 1.01, float("nan")])
def test_kerr_render_config_rejects_invalid_spin(spin: float) -> None:
    with pytest.raises(ValueError):
        KerrRenderConfig(width=4, height=4, spin=spin)


def test_kerr_zamo_camera_matches_static_camera_at_zero_spin() -> None:
    static_camera = make_camera_from_config(
        1.0,
        r=30.0,
        theta=np.pi / 3.0,
        phi=0.0,
        fov_deg=60.0,
        width=8,
        height=8,
    )
    kerr_camera = make_kerr_camera_from_config(
        1.0,
        0.0,
        r=30.0,
        theta=np.pi / 3.0,
        phi=0.0,
        fov_deg=60.0,
        width=8,
        height=8,
    )
    static_v = static_observer_null_direction(static_camera, 0.2, -0.3)
    kerr_v = zamo_null_direction(kerr_camera, 0.2, -0.3)
    np.testing.assert_allclose(kerr_v, static_v, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("spin", [0.0, 0.5, -0.5, 0.999999, -0.999999])
def test_kerr_zamo_null_direction_is_finite_and_null(spin: float) -> None:
    camera = make_kerr_camera_from_config(
        1.0,
        spin,
        r=30.0,
        theta=np.pi / 3.0,
        phi=0.0,
        fov_deg=60.0,
        width=8,
        height=8,
    )
    v = zamo_null_direction(camera, 0.2, -0.3)
    assert np.all(np.isfinite(v))
    y = np.concatenate([initial_position_observer(camera), v])
    assert abs(metric_invariant_kerr(*y, 1.0, spin)) < 1e-12
    result = trace_kerr_null_geodesic(
        initial_position_observer(camera),
        v,
        1.0,
        spin,
        dlambda=0.02,
        max_steps=100,
        r_escape=80.0,
        store_samples=False,
    )
    assert result.status.value == "max_steps"
    assert np.isfinite(result.r_min)


def test_kerr_render_config_rejects_near_pole_observer() -> None:
    with pytest.raises(ValueError, match="poles"):
        KerrRenderConfig(width=4, height=4, observer_theta=1e-12)






def test_kerr_trace_terminates_at_outer_horizon() -> None:
    m = 1.0
    spin = 0.9
    horizon = kerr_outer_horizon(m, spin)
    epsilon = 1e-3
    camera = make_kerr_camera_from_config(
        m,
        spin,
        r=horizon + 0.2,
        theta=np.pi / 3.0,
        phi=0.0,
        fov_deg=60.0,
        width=8,
        height=8,
    )
    result = trace_kerr_null_geodesic(
        initial_position_observer(camera),
        zamo_null_direction(camera, 0.0, 0.0),
        m,
        spin,
        dlambda=0.002,
        max_steps=20_000,
        r_escape=80.0,
        r_horizon_epsilon=epsilon,
        store_samples=False,
    )
    assert result.status.value == "captured"
    assert result.termination_r < horizon + epsilon


def test_kerr_central_ray_converges_with_smaller_step() -> None:
    camera = make_kerr_camera_from_config(
        1.0,
        0.5,
        r=30.0,
        theta=np.pi / 3.0,
        phi=0.0,
        fov_deg=60.0,
        width=8,
        height=8,
    )
    x0 = initial_position_observer(camera)
    v0 = zamo_null_direction(camera, 0.0, 0.0)
    coarse = trace_kerr_null_geodesic(
        x0, v0, 1.0, 0.5, 0.04, 12_000, 80.0, store_samples=False
    )
    fine = trace_kerr_null_geodesic(
        x0, v0, 1.0, 0.5, 0.02, 12_000, 80.0, store_samples=False
    )
    assert coarse.status == fine.status
    assert abs(coarse.r_min - fine.r_min) < 5e-3


def test_kerr_spin_zero_matches_phase2_schwarzschild() -> None:
    cam = make_camera_from_config(
        1.0, r=30.0, theta=np.pi / 2.0, phi=0.0, fov_deg=60.0, width=8, height=8
    )
    x0 = initial_position_observer(cam)
    v0 = static_observer_null_direction(cam, 0.0, 0.0)
    p2 = trace_null_geodesic_3d(
        x0,
        v0,
        m=1.0,
        dlambda=0.1,
        max_steps=2_000,
        r_escape=80.0,
        store_samples=False,
    )
    p3 = trace_kerr_null_geodesic(
        x0,
        v0,
        m=1.0,
        spin=0.0,
        dlambda=0.1,
        max_steps=2_000,
        r_escape=80.0,
        store_samples=False,
    )
    assert p3.status == p2.status
    assert p3.steps_taken == p2.steps_taken
    assert abs(p3.r_min - p2.r_min) < 1e-4
    assert abs(p3.termination_r - p2.termination_r) < 1e-4


def test_equatorial_kerr_conserves_energy_and_angular_momentum() -> None:
    m = 1.0
    spin = 0.5
    y = np.array([0.0, 20.0, np.pi / 2.0, 0.0, 1.2, -0.1, 0.0, 0.025])
    y = renormalize_kerr_momentum_from_null_constraint(y, m, spin)
    e0 = kerr_killing_energy(y, m, spin)
    l0 = kerr_axial_angular_momentum(y, m, spin)
    assert abs(metric_invariant_kerr(*y, m, spin)) < 1e-10

    lam = 0.0
    max_rel_e = 0.0
    max_rel_l = 0.0
    for step_idx in range(1_000):
        y = rk4_step(null_geodesic_kerr_first_order, lam, y, 0.02, m, spin)
        if step_idx % 4 == 0:
            y = renormalize_kerr_momentum_from_null_constraint(y, m, spin)
        lam += 0.02
        max_rel_e = max(max_rel_e, abs(kerr_killing_energy(y, m, spin) - e0) / abs(e0))
        max_rel_l = max(
            max_rel_l,
            abs(kerr_axial_angular_momentum(y, m, spin) - l0) / abs(l0),
        )
        assert abs(float(y[2]) - np.pi / 2.0) < 1e-7

    assert max_rel_e < 1e-3
    assert max_rel_l < 1e-3


def test_off_equatorial_kerr_conserves_carter_constant() -> None:
    m = 1.0
    spin = 0.5
    camera = make_kerr_camera_from_config(
        m,
        spin,
        r=30.0,
        theta=np.pi / 3.0,
        phi=0.0,
        fov_deg=60.0,
        width=8,
        height=8,
    )
    y = np.concatenate(
        [initial_position_observer(camera), zamo_null_direction(camera, 0.25, -0.4)]
    )
    y = renormalize_kerr_momentum_from_null_constraint(y, m, spin)
    e0 = kerr_killing_energy(y, m, spin)
    l0 = kerr_axial_angular_momentum(y, m, spin)
    q0 = kerr_carter_constant(y, m, spin)
    max_rel_e = 0.0
    max_rel_l = 0.0
    max_rel_q = 0.0
    lam = 0.0
    theta_values = [float(y[2])]
    for step_idx in range(400):
        y = rk4_step(null_geodesic_kerr_first_order, lam, y, 0.02, m, spin)
        if step_idx % 4 == 0:
            y = renormalize_kerr_momentum_from_null_constraint(y, m, spin)
        lam += 0.02
        max_rel_e = max(max_rel_e, abs(kerr_killing_energy(y, m, spin) - e0) / abs(e0))
        max_rel_l = max(
            max_rel_l,
            abs(kerr_axial_angular_momentum(y, m, spin) - l0) / abs(l0),
        )
        max_rel_q = max(max_rel_q, abs(kerr_carter_constant(y, m, spin) - q0) / abs(q0))
        theta_values.append(float(y[2]))
    assert max_rel_e < 1e-8
    assert max_rel_l < 1e-8
    assert max_rel_q < 1e-8
    assert max(theta_values) - min(theta_values) > 1e-3


def test_kerr_tiny_render_smoke() -> None:
    cfg = KerrRenderConfig(
        width=3,
        height=3,
        spin=0.4,
        dlambda=0.12,
        max_steps=1_500,
        r_escape=80.0,
    )
    rgb, stats = render_kerr_image(cfg)
    assert rgb.shape == (3, 3, 3)
    assert float(np.max(rgb)) <= 1.0
    assert stats["captured"] + stats["escaped"] + stats["other"] == 9
    assert stats["backend"] == "python_kerr"


def test_phase3_report_and_preset() -> None:
    cfg = render_config_from_preset("fast", spin=0.25)
    assert cfg.width == 18
    assert cfg.spin == 0.25
    report = format_phase3_report(spin=0.25)
    assert "Phase 3 presets" in report
    assert "Kerr null geodesic" in report


def test_kerr_equatorial_isco_decreases_with_prograde_spin() -> None:
    m = 1.0
    assert kerr_equatorial_isco_r(m, 0.0) == pytest.approx(6.0)
    assert kerr_equatorial_isco_r(m, 0.5) < 6.0
    assert kerr_equatorial_isco_r(m, 0.9) < kerr_equatorial_isco_r(m, 0.5)


def test_kerr_spin_changes_equatorial_angular_momentum() -> None:
    m = 1.0
    y = np.array([0.0, 20.0, np.pi / 2.0, 0.0, 1.2, -0.1, 0.0, 0.025])
    y0 = renormalize_kerr_momentum_from_null_constraint(y, m, 0.0)
    y_spin = renormalize_kerr_momentum_from_null_constraint(y, m, 0.5)
    l0 = kerr_axial_angular_momentum(y0, m, 0.0)
    l_spin = kerr_axial_angular_momentum(y_spin, m, 0.5)
    assert l_spin != pytest.approx(l0)


def test_kerr_spin_render_differs_from_schwarzschild_stats() -> None:
    base = dict(width=8, height=8, dlambda=0.12, max_steps=2000, r_escape=80.0)
    cfg_schw = KerrRenderConfig(spin=0.0, **base)
    cfg_kerr = KerrRenderConfig(spin=0.5, **base)
    _, st_schw = render_kerr_image(cfg_schw)
    _, st_kerr = render_kerr_image(cfg_kerr)
    assert st_schw["captured"] + st_schw["escaped"] + st_schw["other"] == 64
    assert st_kerr["captured"] + st_kerr["escaped"] + st_kerr["other"] == 64
    assert (
        st_schw["captured"] != st_kerr["captured"]
        or st_schw["escaped"] != st_kerr["escaped"]
    )


def test_disk_config_for_mass_spin_uses_isco() -> None:
    disk = disk_config_for_mass_spin(1.0, 0.5)
    assert disk.r_inner == pytest.approx(kerr_equatorial_isco_r(1.0, 0.5))
    assert disk.r_outer > disk.r_inner
