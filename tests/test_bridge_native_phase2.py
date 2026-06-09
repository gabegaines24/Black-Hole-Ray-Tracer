"""Optional parity: compiled `_native_phase2` vs Python `trace_null_geodesic_3d`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from blackhole_ray_tracer.native_phase2 import (
    batch_native_available,
    native_phase2_available,
    ray_status_array_from_native,
    schwarzschild_phase2_batch_native,
    schwarzschild_phase2_trace_native,
)
from blackhole_ray_tracer.phase1 import RayStatus
from blackhole_ray_tracer.phase2_camera import (
    make_camera_from_config,
    initial_position_observer,
    static_observer_null_direction,
)
from blackhole_ray_tracer.phase2_geodesic import trace_null_geodesic_3d

STATUS_NATIVE_TO_PY = {
    0: RayStatus.CAPTURED,
    1: RayStatus.ESCAPED,
    2: RayStatus.MAX_STEPS,
    3: RayStatus.NUMERICAL_ERROR,
}


def test_native_phase2_parity_optional() -> None:
    if not native_phase2_available():
        pytest.skip("Extension blackhole_ray_tracer._native_phase2 not installed")

    m = 1.0
    cam = make_camera_from_config(m, r=30.0, theta=np.pi / 2, phi=0.0, fov_deg=60.0, width=16, height=16)
    x0 = initial_position_observer(cam)
    v0 = static_observer_null_direction(cam, 0.0, 0.0)
    y0 = np.concatenate([x0, v0])

    py = trace_null_geodesic_3d(
        x0,
        v0,
        m=m,
        dlambda=0.1,
        max_steps=3000,
        r_escape=80.0,
        store_samples=False,
    )
    nt = schwarzschild_phase2_trace_native(y0, m, 0.1, 3000, 80.0)

    assert STATUS_NATIVE_TO_PY[int(nt["status"])] == py.status
    assert int(nt["steps_taken"]) == py.steps_taken
    assert math.isclose(float(nt["termination_r"]), py.termination_r, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(nt["termination_lambda"]), py.termination_lambda, rel_tol=0.0, abs_tol=1e-7)
    if math.isnan(py.r_min):
        assert math.isnan(float(nt["r_min"]))
    else:
        assert math.isclose(float(nt["r_min"]), py.r_min, rel_tol=0.0, abs_tol=1e-9)


def test_native_phase2_batch_parity_4x4() -> None:
    """Batch bridge produces same results as per-ray Python traces on a 4×4 grid."""
    if not batch_native_available():
        pytest.skip("Extension blackhole_ray_tracer._native_phase2 not installed")

    m, dlambda, max_steps, r_escape = 1.0, 0.1, 3000, 80.0
    cam = make_camera_from_config(m, r=30.0, theta=np.pi / 2, phi=0.0,
                                  fov_deg=60.0, width=4, height=4)
    x0 = initial_position_observer(cam)
    w, h = 4, 4
    y0_list = []
    for j in range(h):
        for i in range(w):
            sx = 2.0 * (i + 0.5) / w - 1.0
            sy = 1.0 - 2.0 * (j + 0.5) / h
            v0 = static_observer_null_direction(cam, sx, sy)
            y0_list.append(np.concatenate([x0, v0]))
    y0_batch = np.stack(y0_list, axis=0)

    batch_result = schwarzschild_phase2_batch_native(y0_batch, m, dlambda, max_steps, r_escape)
    c_statuses = ray_status_array_from_native(batch_result["status"])

    for idx, y0_row in enumerate(y0_list):
        py = trace_null_geodesic_3d(
            y0_row[:4], y0_row[4:],
            m=m, dlambda=dlambda, max_steps=max_steps, r_escape=r_escape,
            store_samples=False,
        )
        assert c_statuses[idx] == py.status, f"ray {idx}: status mismatch"
        assert int(batch_result["steps_taken"][idx]) == py.steps_taken, f"ray {idx}: steps mismatch"
        c_tr = float(batch_result["termination_r"][idx])
        assert math.isclose(c_tr, py.termination_r, rel_tol=0.0, abs_tol=1e-9), \
            f"ray {idx}: termination_r {c_tr} != {py.termination_r}"
