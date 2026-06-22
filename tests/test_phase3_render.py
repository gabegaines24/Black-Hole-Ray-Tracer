"""Tests for phase3_render.render_kerr_3d_image dispatch and output contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from blackhole_ray_tracer.phase3_render import render_kerr_3d_image
from blackhole_ray_tracer.phase3_types import KerrRenderConfig


def _cfg(a: float = 0.0, width: int = 6, height: int = 4, **kw) -> KerrRenderConfig:
    return KerrRenderConfig(
        width=width,
        height=height,
        m=1.0,
        a=a,
        dlambda=0.2,
        max_steps=200,
        r_escape=40.0,
        **kw,
    )


class TestOutputContract:
    """Basic shape / dtype / range guarantees."""

    def test_shape(self):
        cfg = _cfg()
        rgb, _ = render_kerr_3d_image(cfg)
        assert rgb.shape == (cfg.height, cfg.width, 3)

    def test_dtype(self):
        cfg = _cfg()
        rgb, _ = render_kerr_3d_image(cfg)
        assert rgb.dtype == np.float32

    def test_pixel_range(self):
        cfg = _cfg()
        rgb, _ = render_kerr_3d_image(cfg)
        assert rgb.min() >= 0.0 - 1e-5
        assert rgb.max() <= 1.0 + 1e-5

    def test_stats_keys_present(self):
        cfg = _cfg()
        _, stats = render_kerr_3d_image(cfg)
        for k in ("frac_captured", "backend"):
            assert k in stats, f"stats missing '{k}'"


class TestSpinVariants:
    """Ensure different spin values complete without error."""

    @pytest.mark.parametrize("a", [0.0, 0.5, 0.9, -0.5])
    def test_various_spins(self, a: float):
        cfg = _cfg(a=a)
        rgb, stats = render_kerr_3d_image(cfg)
        assert rgb.shape == (cfg.height, cfg.width, 3)
        assert 0.0 <= stats["frac_captured"] <= 1.0

    def test_extreme_spin_near_limit(self):
        """a = 0.99 M should not crash."""
        cfg = _cfg(a=0.99)
        rgb, _ = render_kerr_3d_image(cfg)
        assert rgb.shape == (cfg.height, cfg.width, 3)


class TestDispatch:
    """Dispatch-to-Schwarzschild path when a=0 + use_native_phase2=False."""

    def test_a0_backend_is_kerr_python(self):
        """When a=0 but native is unavailable, backend must be 'kerr_python'."""
        cfg = _cfg(a=0.0, use_native_phase2=False)
        _, stats = render_kerr_3d_image(cfg)
        assert stats["backend"] == "kerr_python"

    def test_captured_fraction_non_negative(self):
        """frac_captured must be a valid fraction in [0, 1]."""
        cfg = _cfg(a=0.5)
        _, stats = render_kerr_3d_image(cfg)
        assert 0.0 <= stats["frac_captured"] <= 1.0


class TestDiskOverlay:
    """Disk overlay does not break output contract."""

    def test_disk_enabled_shape(self):
        from blackhole_ray_tracer.phase2_disk import DiskConfig

        cfg = _cfg(a=0.5, disk=DiskConfig())
        rgb, _ = render_kerr_3d_image(cfg)
        assert rgb.shape == (cfg.height, cfg.width, 3)
