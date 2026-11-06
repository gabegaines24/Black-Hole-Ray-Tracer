"""Phase 2 render path with CUDA batch backend (skip without lib/GPU)."""

from __future__ import annotations

import pytest

from blackhole_ray_tracer.native_phase2_cuda import cuda_batch_available
from blackhole_ray_tracer.phase2_render import render_schwarzschild_3d_image
from blackhole_ray_tracer.phase2_types import Phase2RenderConfig


@pytest.mark.skipif(not cuda_batch_available(), reason="CUDA batch library not available")
def test_render_cuda_batch_backend() -> None:
    cfg = Phase2RenderConfig(
        width=8,
        height=8,
        dlambda=0.1,
        max_steps=400,
        use_cuda_phase2=True,
    )
    rgb, stats = render_schwarzschild_3d_image(cfg)

    assert rgb.shape == (8, 8, 3)
    assert stats["backend"] == "cuda_batch"
    assert 0.0 <= float(rgb.max()) <= 1.0


def test_render_cuda_flag_raises_without_lib() -> None:
    if cuda_batch_available():
        pytest.skip("CUDA lib present; cannot test missing-lib path")
    cfg = Phase2RenderConfig(width=4, height=4, use_cuda_phase2=True)
    with pytest.raises(RuntimeError, match="CUDA batch library"):
        render_schwarzschild_3d_image(cfg)


def test_render_cuda_request_does_not_fallback_to_native(monkeypatch) -> None:
    monkeypatch.setattr(
        "blackhole_ray_tracer.phase2_render.cuda_batch_available", lambda: False
    )
    monkeypatch.setattr(
        "blackhole_ray_tracer.phase2_render.native_phase2_available", lambda: True
    )
    cfg = Phase2RenderConfig(
        width=4,
        height=4,
        use_native_phase2=True,
        use_cuda_batch=True,
    )
    with pytest.raises(RuntimeError, match="CUDA batch library"):
        render_schwarzschild_3d_image(cfg)
