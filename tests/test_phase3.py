"""Tests for Phase 3 Kerr Boyer-Lindquist geodesics and rendering."""

from __future__ import annotations

import numpy as np

from blackhole_ray_tracer.phase1 import rk4_step
from blackhole_ray_tracer.phase2_camera import (
    initial_position_observer,
    make_camera_from_config,
    static_observer_null_direction,
)
from blackhole_ray_tracer.phase2_geodesic import trace_null_geodesic_3d
from blackhole_ray_tracer.phase3_christoffel import null_geodesic_kerr_first_order
from blackhole_ray_tracer.phase3_geodesic import (
    kerr_axial_angular_momentum,
    kerr_killing_energy,
    metric_invariant_kerr,
    renormalize_kerr_momentum_from_null_constraint,
    trace_kerr_null_geodesic,
)
from blackhole_ray_tracer.phase3_render import render_kerr_image
from blackhole_ray_tracer.phase3_report import format_phase3_report, render_config_from_preset
from blackhole_ray_tracer.phase3_types import KerrRenderConfig


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
