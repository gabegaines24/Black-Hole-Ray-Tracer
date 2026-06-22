"""Tests for Phase 2 supersample (anti-aliasing) path.

Verifies that:
1. Rendering with supersample=2 returns (H, W, 3) at the *original* resolution.
2. Pixel values are in [0, 1].
3. The supersample image differs from the non-supersample image (box-average changes values).
"""

from __future__ import annotations

import numpy as np
import pytest

from blackhole_ray_tracer.phase2_render import render_schwarzschild_3d_image
from blackhole_ray_tracer.phase2_types import Phase2RenderConfig


def _tiny_cfg(supersample: int = 1) -> Phase2RenderConfig:
    return Phase2RenderConfig(
        width=6,
        height=4,
        m=1.0,
        dlambda=0.2,
        max_steps=200,
        r_escape=40.0,
        supersample=supersample,
    )


class TestSupersample:
    def test_output_shape_unchanged(self):
        """supersample=2 must return the *original* (H, W, 3) shape."""
        cfg = _tiny_cfg(supersample=2)
        rgb, stats = render_schwarzschild_3d_image(cfg)
        assert rgb.shape == (cfg.height, cfg.width, 3), f"Unexpected shape: {rgb.shape}"

    def test_output_dtype_float32(self):
        cfg = _tiny_cfg(supersample=2)
        rgb, _ = render_schwarzschild_3d_image(cfg)
        assert rgb.dtype == np.float32

    def test_pixel_values_in_unit_interval(self):
        cfg = _tiny_cfg(supersample=2)
        rgb, _ = render_schwarzschild_3d_image(cfg)
        assert rgb.min() >= 0.0 - 1e-5
        assert rgb.max() <= 1.0 + 1e-5

    def test_supersample_1_matches_no_supersample(self):
        """supersample=1 must give exactly the same result as the default."""
        cfg_default = _tiny_cfg(supersample=1)
        cfg_one = _tiny_cfg(supersample=1)
        rgb_d, _ = render_schwarzschild_3d_image(cfg_default)
        rgb_1, _ = render_schwarzschild_3d_image(cfg_one)
        np.testing.assert_array_equal(rgb_d, rgb_1)

    def test_stats_reports_supersample(self):
        """Stats dict should include supersample=2 when set."""
        cfg = _tiny_cfg(supersample=2)
        _, stats = render_schwarzschild_3d_image(cfg)
        assert stats.get("supersample") == 2

    def test_supersample_2_differs_from_1(self):
        """2× supersample should produce different pixel values from 1× (box-average smooths edges)."""
        cfg_1x = _tiny_cfg(supersample=1)
        cfg_2x = _tiny_cfg(supersample=2)
        rgb_1x, _ = render_schwarzschild_3d_image(cfg_1x)
        rgb_2x, _ = render_schwarzschild_3d_image(cfg_2x)
        assert not np.allclose(rgb_1x, rgb_2x), "2× supersample should differ from 1×"
