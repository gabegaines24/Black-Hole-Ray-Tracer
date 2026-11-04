"""ML surrogate integration in Phase 2 render."""

from __future__ import annotations

import numpy as np
import pytest

from blackhole_ray_tracer.phase2_ml import ml_surrogate_available
from blackhole_ray_tracer.phase2_render import render_schwarzschild_3d_image
from blackhole_ray_tracer.phase2_types import Phase2RenderConfig
from ml.surrogate import init_random_weights, save_surrogate


def test_ml_render_missing_model_raises() -> None:
    cfg = Phase2RenderConfig(
        width=4,
        height=4,
        use_ml_surrogate=True,
        ml_model_path="/nonexistent/surrogate.npz",
    )
    with pytest.raises(RuntimeError, match="surrogate model"):
        render_schwarzschild_3d_image(cfg)


def test_ml_render_backend_with_saved_weights(tmp_path) -> None:
    model_path = tmp_path / "surrogate.npz"
    save_surrogate(init_random_weights(np.random.default_rng(0)), model_path)

    cfg = Phase2RenderConfig(
        width=4,
        height=4,
        dlambda=0.12,
        max_steps=1500,
        r_escape=80.0,
        use_ml_surrogate=True,
        ml_model_path=str(model_path),
    )
    assert ml_surrogate_available(cfg)
    rgb, stats = render_schwarzschild_3d_image(cfg)
    assert rgb.shape == (4, 4, 3)
    assert stats["backend"] == "ml_surrogate"
    assert stats["n_surrogate"] == 16.0
    assert stats["n_integrated"] == 0.0
