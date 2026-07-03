"""CUDA batch backend wiring for Phase 2 render."""

from __future__ import annotations

import pytest

from blackhole_ray_tracer.native_phase2_cuda import cuda_batch_available
from blackhole_ray_tracer.phase2_render import render_schwarzschild_3d_image
from blackhole_ray_tracer.phase2_types import Phase2RenderConfig


def test_cuda_batch_unavailable_raises_when_requested() -> None:
    if cuda_batch_available():
        pytest.skip("CUDA library present — cannot test missing-lib error path")
    cfg = Phase2RenderConfig(width=4, height=4, use_cuda_batch=True)
    with pytest.raises(RuntimeError, match="libbh_rt_phase2_cuda"):
        render_schwarzschild_3d_image(cfg)


def test_python_backend_when_cuda_not_requested() -> None:
    cfg = Phase2RenderConfig(width=4, height=4, dlambda=0.12, max_steps=1500, r_escape=80.0)
    _, stats = render_schwarzschild_3d_image(cfg)
    assert stats["backend"] == "python"
