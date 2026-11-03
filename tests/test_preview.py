"""Headless preview routing tests."""

from __future__ import annotations

import numpy as np

from blackhole_ray_tracer.native_phase2 import batch_native_available
from blackhole_ray_tracer.preview import (
    _build_config,
    preview_backend_label,
    render_preview_frame,
)


def test_preview_flags_propagate_to_headless_frame() -> None:
    cfg = _build_config(
        preset=None,
        width=4,
        height=4,
        use_native=True,
        disk=True,
        use_cuda=True,
    )
    assert cfg.use_native_phase2
    assert cfg.use_cuda_batch
    assert cfg.disk is not None
    expected_backend = "native_batch" if batch_native_available() else "python"
    assert preview_backend_label(cfg) == expected_backend
    rgb, stats = render_preview_frame(cfg)
    assert rgb.shape == (4, 4, 3)
    assert np.all(np.isfinite(rgb))
    assert stats.get("backend") == expected_backend
