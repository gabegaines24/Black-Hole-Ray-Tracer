"""Schwarzschild 3D pinhole image: trace null geodesics per pixel, color by outcome + sky."""

from __future__ import annotations

import math

import numpy as np

from .phase1 import RayStatus
from .phase2_batch import prepare_phase2_ray_batch, trace_phase2_ray_batch_python
from .phase2_types import Phase2RenderConfig


def _sky_rgb_from_direction(
    sx: float, sy: float, mode: str
) -> tuple[float, float, float]:
    """Map screen space to a simple synthetic background color (escaped rays)."""
    if mode == "flat":
        return (0.15, 0.18, 0.28)
    th = math.atan2(sy, sx)
    g = 0.5 + 0.5 * math.sin(th)
    b = 0.5 + 0.5 * math.cos(th + 0.7)
    r = 0.35 + 0.25 * g
    return (min(r, 1.0), min(g, 1.0), min(b, 1.0))


def render_schwarzschild_3d_image(
    cfg: Phase2RenderConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return RGB float32 (H, W, 3) in [0,1] and simple stats dict."""
    h, w = cfg.height, cfg.width
    batch = prepare_phase2_ray_batch(cfg)
    results = trace_phase2_ray_batch_python(batch, cfg)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    n_cap = n_esc = n_other = 0

    for idx, result in enumerate(results):
        j, i = divmod(idx, w)
        sx = float(batch.sx[idx])
        sy = float(batch.sy[idx])
        if result.status == RayStatus.CAPTURED:
            rgb[j, i, :] = 0.0
            n_cap += 1
        elif result.status == RayStatus.ESCAPED:
            br, bg, bb = _sky_rgb_from_direction(sx, sy, cfg.sky_mode)
            rgb[j, i, 0] = br
            rgb[j, i, 1] = bg
            rgb[j, i, 2] = bb
            n_esc += 1
        else:
            rgb[j, i, :] = (0.12, 0.1, 0.15)
            n_other += 1

    total = max(h * w, 1)
    stats = {
        "captured": n_cap,
        "escaped": n_esc,
        "other": n_other,
        "frac_captured": n_cap / total,
    }
    return rgb, stats
