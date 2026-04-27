"""Tests for Phase 2: camera, Christoffel geodesics, render smoke, report."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from blackhole_ray_tracer.phase1 import RayStatus
from blackhole_ray_tracer.phase1_image import write_ppm_rgb
from blackhole_ray_tracer.phase2_camera import (
    make_camera_from_config,
    initial_position_observer,
    static_observer_null_direction,
)
from blackhole_ray_tracer.phase2_christoffel import christoffel_schwarzschild, f_schwarzschild
from blackhole_ray_tracer.phase2_geodesic import metric_invariant_schwarzschild, trace_null_geodesic_3d
from blackhole_ray_tracer.phase2_render import render_schwarzschild_3d_image
from blackhole_ray_tracer.phase2_report import format_phase2_report, render_config_from_preset
from blackhole_ray_tracer.phase2_types import Phase2RenderConfig


def test_f_schwarzschild_horizon() -> None:
    m = 1.0
    assert f_schwarzschild(2.0 * m + 0.1, m) > 0.0
    assert f_schwarzschild(2.0 * m, m) == 0.0


def test_christoffel_symmetry() -> None:
    m = 1.0
    r, th = 10.0, 1.1
    for mu in range(4):
        for a in range(4):
            for b in range(4):
                g1 = christoffel_schwarzschild(mu, a, b, r, th, m)
                g2 = christoffel_schwarzschild(mu, b, a, r, th, m)
                assert g1 == pytest.approx(g2)


def test_initial_null_direction_is_null() -> None:
    m = 1.0
    cam = make_camera_from_config(m, r=30.0, theta=np.pi / 2, phi=0.0, fov_deg=60.0, width=32, height=24)
    x0 = initial_position_observer(cam)
    v0 = static_observer_null_direction(cam, 0.0, 0.0)
    t, r, th, ph = float(x0[0]), float(x0[1]), float(x0[2]), float(x0[3])
    vt, vr, vth, vph = float(v0[0]), float(v0[1]), float(v0[2]), float(v0[3])
    gvv = metric_invariant_schwarzschild(t, r, th, ph, vt, vr, vth, vph, m)
    assert abs(gvv) < 1e-6


def test_trace_geodesic_center_pixel_finishes() -> None:
    m = 1.0
    cam = make_camera_from_config(m, r=30.0, theta=np.pi / 2, phi=0.0, fov_deg=60.0, width=16, height=16)
    x0 = initial_position_observer(cam)
    v0 = static_observer_null_direction(cam, 0.0, 0.0)
    res = trace_null_geodesic_3d(
        x0,
        v0,
        m=m,
        dlambda=0.1,
        max_steps=3000,
        r_escape=80.0,
        store_samples=False,
    )
    assert res.status in set(RayStatus)
    assert res.steps_taken > 0


def test_render_tiny_image_smoke() -> None:
    cfg = Phase2RenderConfig(width=4, height=4, dlambda=0.12, max_steps=1500, r_escape=80.0)
    rgb, stats = render_schwarzschild_3d_image(cfg)
    assert rgb.shape == (4, 4, 3)
    assert "captured" in stats
    assert float(np.max(rgb)) <= 1.0
    assert stats["captured"] + stats["escaped"] + stats["other"] == 16
    if stats["captured"] > 0:
        assert float(np.min(rgb)) < 0.1


def test_ppm_round_trip_tmp() -> None:
    cfg = Phase2RenderConfig(width=4, height=4, dlambda=0.12, max_steps=1500)
    rgb, _ = render_schwarzschild_3d_image(cfg)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.ppm"
        write_ppm_rgb(str(p), rgb)
        assert p.stat().st_size > 0


def test_format_phase2_report_contains_sections() -> None:
    r = format_phase2_report()
    assert "Phase 2 presets" in r
    assert "Phase 2 benchmark" in r


def test_render_config_from_preset() -> None:
    c = render_config_from_preset("fast", m=1.0)
    assert c.width == 24 and c.max_steps == 4000
