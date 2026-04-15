import numpy as np

from blackhole_ray_tracer.phase1 import (
    RayStatus,
    batch_schwarzschild_rays,
    format_step_d_table,
    step_b_trajectory_is_finite,
    trace_single_schwarzschild_ray,
)
from blackhole_ray_tracer.phase1_image import render_einstein_ring_image
from blackhole_ray_tracer.phase1_tuning import format_step_f_report


def test_batch_schwarzschild_rays_returns_expected_rows() -> None:
    b_values = [2.8, 6.0]
    rows = batch_schwarzschild_rays(b_values, dphi=0.02, phi_max=8.0, r_escape=80.0)

    assert len(rows) == len(b_values)
    assert [row.impact_b for row in rows] == b_values
    assert all(row.steps_taken > 0 for row in rows)
    assert all(row.status in set(RayStatus) for row in rows)


def test_format_step_d_table_includes_key_fields() -> None:
    rows = batch_schwarzschild_rays([2.8, 6.0], dphi=0.02, phi_max=8.0, r_escape=80.0)
    table = format_step_d_table(rows)

    assert "Step D (batch rays over impact parameter)" in table
    assert "counts: captured=" in table
    assert "weak-field lensing scale" in table


def test_render_einstein_ring_image_shape_and_ranges() -> None:
    rgb, b_map = render_einstein_ring_image(
        8,
        6,
        dphi=0.02,
        phi_max=4.0,
        r_escape=60.0,
        b_min=2.5,
        b_max=10.0,
    )

    assert rgb.shape == (6, 8, 3)
    assert b_map.shape == (6, 8)
    assert np.all((rgb >= 0.0) & (rgb <= 1.0))
    assert float(np.min(b_map)) >= 2.5
    assert float(np.max(b_map)) <= 10.0


def test_step_b_trajectory_is_finite_for_default_ray() -> None:
    result = trace_single_schwarzschild_ray()
    assert step_b_trajectory_is_finite(result) is True


def test_format_step_f_report_contains_sections() -> None:
    report = format_step_f_report()

    assert "Step F presets" in report
    assert "fast" in report
    assert "balanced" in report
    assert "quality" in report
    assert "Step F benchmark" in report
