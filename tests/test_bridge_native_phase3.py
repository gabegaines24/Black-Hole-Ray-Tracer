"""Optional parity: compiled `_native_phase3` vs Python `trace_kerr_null_geodesic`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from blackhole_ray_tracer.native_phase3 import (
    batch_native_phase3_available,
    kerr_phase3_batch_native,
    kerr_phase3_trace_native,
    native_phase3_available,
    ray_status_array_from_native,
    ray_status_from_native_phase2,
)
from blackhole_ray_tracer.phase2_batch import build_camera_y0
from blackhole_ray_tracer.phase2_camera import (
    initial_position_observer,
    make_camera_from_config,
    static_observer_null_direction,
)
from blackhole_ray_tracer.phase2_types import Phase2RenderConfig
from blackhole_ray_tracer.phase3_geodesic import trace_kerr_null_geodesic


def test_native_phase3_parity_optional() -> None:
    if not native_phase3_available():
        pytest.skip("Extension blackhole_ray_tracer._native_phase3 not installed")

    m, a = 1.0, 0.5
    cam = make_camera_from_config(m, r=30.0, theta=np.pi / 2, phi=0.0, fov_deg=60.0, width=16, height=16)
    x0 = initial_position_observer(cam)
    v0 = static_observer_null_direction(cam, 0.0, 0.0)
    y0 = np.concatenate([x0, v0])

    py = trace_kerr_null_geodesic(
        x0,
        v0,
        m=m,
        a=a,
        dlambda=0.1,
        max_steps=3000,
        r_escape=80.0,
        store_samples=False,
    )
    nt = kerr_phase3_trace_native(y0, m, a, 0.1, 3000, 80.0)

    assert ray_status_from_native_phase2(nt) == py.status
    assert int(nt["steps_taken"]) == py.steps_taken
    assert math.isclose(float(nt["termination_r"]), py.termination_r, rel_tol=0.0, abs_tol=1e-7)
    assert math.isclose(float(nt["termination_lambda"]), py.termination_lambda, rel_tol=0.0, abs_tol=1e-5)
    if math.isnan(py.r_min):
        assert math.isnan(float(nt["r_min"]))
    else:
        assert math.isclose(float(nt["r_min"]), py.r_min, rel_tol=0.0, abs_tol=1e-7)


def test_native_phase3_batch_parity_4x4() -> None:
    if not batch_native_phase3_available():
        pytest.skip("Extension blackhole_ray_tracer._native_phase3 not installed")

    m, a = 1.0, 0.7
    dlambda, max_steps, r_escape = 0.1, 3000, 80.0
    cfg = Phase2RenderConfig(width=4, height=4, m=m, r_observer=30.0)
    y0 = build_camera_y0(cfg)

    batch = kerr_phase3_batch_native(y0, m, a, dlambda, max_steps, r_escape)
    statuses = ray_status_array_from_native(batch["status"])

    for idx in range(y0.shape[0]):
        py = trace_kerr_null_geodesic(
            y0[idx, :4],
            y0[idx, 4:],
            m=m,
            a=a,
            dlambda=dlambda,
            max_steps=max_steps,
            r_escape=r_escape,
            store_samples=False,
        )
        assert statuses[idx] == py.status
        assert int(batch["steps_taken"][idx]) == py.steps_taken
