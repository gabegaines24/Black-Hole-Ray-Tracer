"""Tests for Phase 2: camera, Christoffel geodesics, render smoke, report."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from blackhole_ray_tracer.native_phase2 import (
    batch_native_available,
    native_phase2_available,
    schwarzschild_phase2_batch_native,
)
from blackhole_ray_tracer.phase1 import RayStatus
from blackhole_ray_tracer.phase1_image import write_ppm_rgb
from blackhole_ray_tracer.phase2_camera import (
    make_camera_from_config,
    initial_position_observer,
    static_observer_null_direction,
)
from blackhole_ray_tracer.phase2_batch import prepare_phase2_ray_batch, trace_phase2_ray_batch_python
from blackhole_ray_tracer.phase2_christoffel import christoffel_schwarzschild, f_schwarzschild
from blackhole_ray_tracer.phase2_geodesic import metric_invariant_schwarzschild, trace_null_geodesic_3d
from blackhole_ray_tracer.phase2_disk import (
    detect_equatorial_crossing_state,
    disk_color_at_r,
    disk_hit_from_equatorial_crossing,
)
from blackhole_ray_tracer.phase2_render import render_schwarzschild_3d_image
from blackhole_ray_tracer.phase2_report import format_phase2_report, render_config_from_preset
from blackhole_ray_tracer.phase2_types import DiskConfig, Phase2RenderConfig


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 0},
        {"height": -1},
        {"m": 0.0},
        {"r_observer": 2.0},
        {"observer_theta": 0.0},
        {"fov_deg": 180.0},
        {"dlambda": 0.0},
        {"max_steps": 0},
        {"max_steps": -1},
    ],
)
def test_phase2_render_config_rejects_invalid_parameters(
    overrides: dict[str, object],
) -> None:
    params: dict[str, object] = {"width": 4, "height": 4}
    params.update(overrides)
    with pytest.raises(ValueError):
        Phase2RenderConfig(**params)


def test_disk_config_reexport_is_canonical() -> None:
    from blackhole_ray_tracer.phase2_disk import DiskConfig as CanonicalDiskConfig

    assert DiskConfig is CanonicalDiskConfig




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


def test_radial_connection_matches_schwarzschild_metric() -> None:
    m = 1.0
    r, th = 10.0, 1.1
    f = f_schwarzschild(r, m)
    expected = -m / (r * r * f)
    assert christoffel_schwarzschild(1, 1, 1, r, th, m) == pytest.approx(expected)

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


def test_prepare_phase2_ray_batch_shapes_and_order() -> None:
    cfg = Phase2RenderConfig(width=3, height=2)
    batch = prepare_phase2_ray_batch(cfg)
    assert batch.count == 6
    assert batch.width == 3
    assert batch.height == 2
    assert batch.t0.shape == (6,)
    assert batch.vphi0.shape == (6,)
    assert np.all(batch.r0 == pytest.approx(cfg.r_observer))
    assert batch.sx.tolist() == pytest.approx(
        [-2.0 / 3.0, 0.0, 2.0 / 3.0, -2.0 / 3.0, 0.0, 2.0 / 3.0]
    )
    assert batch.sy.tolist() == pytest.approx([0.5, 0.5, 0.5, -0.5, -0.5, -0.5])


def test_trace_phase2_ray_batch_python_reference() -> None:
    cfg = Phase2RenderConfig(width=2, height=2, dlambda=0.12, max_steps=1500, r_escape=80.0)
    batch = prepare_phase2_ray_batch(cfg)
    results = trace_phase2_ray_batch_python(batch, cfg)
    assert len(results) == 4
    assert all(r.status in set(RayStatus) for r in results)
    assert all(r.steps_taken > 0 for r in results)


def test_render_tiny_image_smoke() -> None:
    cfg = Phase2RenderConfig(width=4, height=4, dlambda=0.12, max_steps=1500, r_escape=80.0)
    rgb, stats = render_schwarzschild_3d_image(cfg)
    assert rgb.shape == (4, 4, 3)
    assert "captured" in stats
    assert float(np.max(rgb)) <= 1.0
    assert stats["captured"] + stats["escaped"] + stats["other"] == 16
    assert stats.get("backend") == "python"
    if stats["captured"] > 0:
        assert float(np.min(rgb)) < 0.1


def test_render_tiny_image_with_disk_smoke() -> None:
    cfg = Phase2RenderConfig(
        width=4,
        height=4,
        dlambda=0.12,
        max_steps=1500,
        r_escape=80.0,
        disk=DiskConfig(),
    )
    rgb, stats = render_schwarzschild_3d_image(cfg)
    assert rgb.shape == (4, 4, 3)
    assert stats["captured"] + stats["escaped"] + stats["other"] == 16
    assert stats.get("backend") == "python"


def test_render_tiny_image_with_ml_gate_fallback(tmp_path: Path) -> None:
    cfg = Phase2RenderConfig(
        width=2,
        height=2,
        dlambda=0.12,
        max_steps=1000,
        r_escape=80.0,
        ml_gate_model_path=str(tmp_path / "missing-surrogate.npz"),
    )
    rgb, stats = render_schwarzschild_3d_image(cfg)
    assert rgb.shape == (2, 2, 3)
    assert stats["captured"] + stats["escaped"] + stats["other"] == 4
    assert stats["backend"] == "ml_gate_fallback"
    assert stats["surrogate_frac"] == 0.0


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


def test_render_native_raises_when_extension_missing() -> None:
    if native_phase2_available():
        pytest.skip("native extension present")
    cfg = Phase2RenderConfig(
        width=2,
        height=2,
        dlambda=0.12,
        max_steps=500,
        r_escape=80.0,
        use_native_phase2=True,
    )
    with pytest.raises(RuntimeError, match="_native_phase2"):
        render_schwarzschild_3d_image(cfg)


def test_render_native_matches_python_when_extension_present() -> None:
    if not native_phase2_available():
        pytest.skip("native extension not built")
    base = Phase2RenderConfig(width=4, height=4, dlambda=0.12, max_steps=1500, r_escape=80.0)
    rgb_py, st_py = render_schwarzschild_3d_image(replace(base, use_native_phase2=False))
    rgb_na, st_na = render_schwarzschild_3d_image(replace(base, use_native_phase2=True))
    assert st_py["captured"] == st_na["captured"]
    assert st_py["escaped"] == st_na["escaped"]
    assert st_py["other"] == st_na["other"]
    assert st_na.get("backend") in ("native", "native_batch")
    assert np.allclose(rgb_py, rgb_na, rtol=0.0, atol=1e-5)


def test_render_native_batch_matches_python_stats() -> None:
    if not batch_native_available():
        pytest.skip("native batch extension not built")
    base = Phase2RenderConfig(width=8, height=8, dlambda=0.12, max_steps=1500, r_escape=80.0)
    _, st_py = render_schwarzschild_3d_image(replace(base, use_native_phase2=False))
    _, st_na = render_schwarzschild_3d_image(replace(base, use_native_phase2=True))
    assert st_py["captured"] == st_na["captured"]
    assert st_py["escaped"] == st_na["escaped"]
    assert st_py["other"] == st_na["other"]
    assert st_na.get("backend") == "native_batch"


def test_render_native_batch_with_disk_smoke() -> None:
    if not batch_native_available():
        pytest.skip("native batch extension not built")
    cfg = Phase2RenderConfig(
        width=16,
        height=16,
        dlambda=0.1,
        max_steps=3000,
        r_escape=80.0,
        use_native_phase2=True,
        disk=DiskConfig(),
    )
    rgb, stats = render_schwarzschild_3d_image(cfg)
    assert rgb.shape == (16, 16, 3)
    assert stats.get("backend") == "native_batch"
    assert stats["captured"] + stats["escaped"] + stats["other"] == 256
    # Disk overlay uses warm colors distinct from flat sky background.
    warm_pixels = np.sum((rgb[:, :, 0] > 0.4) & (rgb[:, :, 1] > 0.2) & (rgb[:, :, 2] < 0.5))
    assert warm_pixels >= 1


def test_equatorial_crossing_interpolates_first_velocity_state() -> None:
    half_pi = np.pi / 2.0
    crossing = detect_equatorial_crossing_state(
        [half_pi - 0.2, half_pi + 0.1, half_pi - 0.2],
        [10.0, 20.0, 30.0],
        [1.0, 3.0, 5.0],
        [0.0, 1.0, 2.0],
    )
    assert crossing is not None
    assert crossing.r_cross == pytest.approx(16.6666666667)
    assert crossing.vt_cross == pytest.approx(2.3333333333)
    assert crossing.vph_cross == pytest.approx(0.6666666667)


def test_equatorial_crossing_rejects_misaligned_velocity_samples() -> None:
    with pytest.raises(ValueError, match="velocity samples"):
        detect_equatorial_crossing_state(
            [np.pi / 2.0 - 0.1, np.pi / 2.0 + 0.1],
            [10.0, 12.0],
            [1.0],
            [0.0, 0.1],
        )


def test_disk_edges_and_colors_are_bounded() -> None:
    disk = DiskConfig()
    assert disk_hit_from_equatorial_crossing(6.0, disk, 1.0, 1.0, 0.1).hit
    assert disk_hit_from_equatorial_crossing(20.0, disk, 1.0, 1.0, 0.1).hit
    assert not disk_hit_from_equatorial_crossing(5.999, disk, 1.0, 1.0, 0.1).hit
    assert not disk_hit_from_equatorial_crossing(20.001, disk, 1.0, 1.0, 0.1).hit
    color = disk_color_at_r(6.0, -1.0, disk)
    assert all(0.0 <= channel <= 1.0 and np.isfinite(channel) for channel in color)


def test_equatorial_crossing_returns_none_without_sign_change() -> None:
    half_pi = np.pi / 2.0
    assert (
        detect_equatorial_crossing_state(
            [half_pi - 0.2, half_pi - 0.1],
            [10.0, 12.0],
            [1.0, 1.1],
            [0.0, 0.1],
        )
        is None
    )


def test_native_capture_precedes_disk_crossing() -> None:
    if not batch_native_available():
        pytest.skip("native batch extension not built")
    y0 = np.array(
        [[0.0, 2.0, np.pi / 2.0 - 0.1, 0.0, 1.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    result = schwarzschild_phase2_batch_native(y0, 1.0, 0.1, 10, 80.0)
    assert int(result["status"][0]) == 0
    assert int(result["disk_hit"][0]) == 0


def test_equatorial_crossing_in_disk_range() -> None:
    m = 1.0
    cam = make_camera_from_config(
        m, r=30.0, theta=np.pi / 2 + 0.15, phi=0.0, fov_deg=60.0, width=8, height=8
    )
    x0 = initial_position_observer(cam)
    v0 = static_observer_null_direction(cam, 0.0, 0.75)
    res = trace_null_geodesic_3d(
        x0,
        v0,
        m=m,
        dlambda=0.08,
        max_steps=4000,
        r_escape=80.0,
        store_samples=True,
    )
    crossing = detect_equatorial_crossing_state(
        res.theta_samples, res.r_samples, res.vt_samples, res.vphi_samples
    )
    assert crossing is not None
    assert 6.0 <= crossing.r_cross <= 20.0
    assert np.isfinite(crossing.vt_cross)
    assert np.isfinite(crossing.vph_cross)


def test_native_batch_disk_crossing_matches_python() -> None:
    if not batch_native_available():
        pytest.skip("native batch extension not built")
    m = 1.0
    cam = make_camera_from_config(
        m, r=30.0, theta=np.pi / 2 + 0.15, phi=0.0, fov_deg=60.0, width=1, height=1
    )
    x0 = initial_position_observer(cam)
    v0 = static_observer_null_direction(cam, 0.0, 0.75)
    y0 = np.concatenate([x0, v0]).reshape(1, 8)
    py = trace_null_geodesic_3d(
        x0, v0, m=m, dlambda=0.08, max_steps=4000, r_escape=80.0, store_samples=True
    )
    batch = schwarzschild_phase2_batch_native(y0, m, 0.08, 4000, 80.0)
    crossing = detect_equatorial_crossing_state(
        py.theta_samples, py.r_samples, py.vt_samples, py.vphi_samples
    )
    assert crossing is not None
    assert int(batch["disk_hit"][0]) == 1
    assert np.isclose(
        float(batch["disk_r_cross"][0]), crossing.r_cross, rtol=0.0, atol=1e-6
    )
    assert np.isclose(
        float(batch["disk_vt"][0]), crossing.vt_cross, rtol=0.0, atol=1e-6
    )
    assert np.isclose(
        float(batch["disk_vph"][0]), crossing.vph_cross, rtol=0.0, atol=1e-6
    )
