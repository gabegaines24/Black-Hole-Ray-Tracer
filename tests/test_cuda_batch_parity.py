"""Optional parity: CUDA batch vs CPU native batch (skipped without GPU lib)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from blackhole_ray_tracer.native_phase2 import (
    batch_native_available,
    schwarzschild_phase2_batch_native,
)
from blackhole_ray_tracer.native_phase2_cuda import (
    MAX_CUDA_BATCH_RAYS,
    cuda_batch_available,
    schwarzschild_phase2_batch_cuda,
)
from blackhole_ray_tracer.phase2_camera import (
    initial_position_observer,
    make_camera_from_config,
    static_observer_null_direction,
)


def _build_grid_y0(width: int, height: int) -> np.ndarray:
    cam = make_camera_from_config(
        1.0, r=30.0, theta=np.pi / 2, phi=0.0, fov_deg=60.0, width=width, height=height
    )
    x0 = initial_position_observer(cam)
    rays = []
    for j in range(height):
        for i in range(width):
            sx = 2.0 * (i + 0.5) / width - 1.0
            sy = 1.0 - 2.0 * (j + 0.5) / height
            v0 = static_observer_null_direction(cam, sx, sy)
            rays.append(np.concatenate([x0, v0]))
    return np.stack(rays, axis=0).astype(np.float64)


def test_cuda_batch_boundaries_are_defined_without_cuda_library() -> None:
    with pytest.raises(ValueError, match=r"\(N, 8\)"):
        schwarzschild_phase2_batch_cuda(
            np.zeros((1, 7), dtype=np.float32), 1.0, 0.1, 10, 80.0
        )
    with pytest.raises(ValueError, match="finite"):
        schwarzschild_phase2_batch_cuda(
            np.full((1, 8), np.nan), 1.0, 0.1, 10, 80.0
        )
    with pytest.raises(ValueError, match="max_steps"):
        schwarzschild_phase2_batch_cuda(
            np.zeros((1, 8), dtype=np.float32), 1.0, 0.1, 0, 80.0
        )
    with pytest.raises(ValueError, match="limit"):
        schwarzschild_phase2_batch_cuda(
            np.empty((MAX_CUDA_BATCH_RAYS + 1, 8), dtype=np.float64),
            1.0,
            0.1,
            10,
            80.0,
        )
    result = schwarzschild_phase2_batch_cuda(
        np.empty((0, 8), dtype=np.float64), 1.0, 0.1, 10, 80.0
    )
    assert result["status"].dtype == np.int32
    assert result["status"].shape == (0,)


def test_cuda_batch_parity_vs_cpu_native() -> None:
    if not cuda_batch_available():
        pytest.skip("CUDA batch library not available (BLACKHOLE_CUDA_LIB)")
    if not batch_native_available():
        pytest.skip("CPU native batch extension not built")

    y0 = _build_grid_y0(4, 4)
    m, dlambda, max_steps, r_escape = 1.0, 0.1, 3000, 80.0
    cpu = schwarzschild_phase2_batch_native(
        y0, m, dlambda, max_steps, r_escape, r_horizon_epsilon=1e-3
    )
    gpu = schwarzschild_phase2_batch_cuda(
        y0, m, dlambda, max_steps, r_escape, r_horizon_epsilon=1e-3
    )

    assert np.array_equal(cpu["status"], gpu["status"])
    assert np.array_equal(cpu["steps_taken"], gpu["steps_taken"])
    for key in ("termination_r", "r_min", "disk_r_cross", "disk_vt", "disk_vph"):
        for a, b in zip(cpu[key], gpu[key], strict=True):
            if math.isnan(float(a)) and math.isnan(float(b)):
                continue
            assert math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6)
    assert np.array_equal(cpu["disk_hit"], gpu["disk_hit"])
