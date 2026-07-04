"""Smoke tests for CUDA-backed Phase 2 render path."""

from __future__ import annotations

import numpy as np

from blackhole_ray_tracer.native_phase2_cuda import cuda_batch_available
from blackhole_ray_tracer.phase2_render import render_schwarzschild_3d_image
from blackhole_ray_tracer.phase2_types import Phase2RenderConfig


def test_render_cuda_flag_smoke() -> None:
    """use_cuda=True completes; uses cuda_batch when the library is loadable."""
    cfg = Phase2RenderConfig(
        width=8,
        height=8,
        dlambda=0.1,
        max_steps=2000,
        use_cuda=True,
        use_native_phase2=True,
    )
    rgb, stats = render_schwarzschild_3d_image(cfg)
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.float32
    assert stats["captured"] + stats["escaped"] + stats["other"] == 64
    if cuda_batch_available():
        assert stats["backend"] == "cuda_batch"
    else:
        assert stats["backend"] in ("native_batch", "python")
