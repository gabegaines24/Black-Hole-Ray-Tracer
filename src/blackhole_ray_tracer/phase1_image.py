"""Step E: simple equatorial camera image — shadow + sky (Einstein-ring prototype).

Each pixel maps to an impact parameter b proportional to radius from the image center
(circular camera / axial symmetry). Escaped rays sample a synthetic background by
sky angle; captured rays are black.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .phase1 import RayStatus, trace_single_schwarzschild_ray


def render_einstein_ring_image(
    width: int,
    height: int,
    *,
    m: float = 1.0,
    b_min: float = 2.0,
    b_max: float = 12.0,
    **trace_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB float32 array (H, W, 3) in [0, 1] and per-pixel impact parameter b.

    ``trace_kwargs`` are passed to :func:`trace_single_schwarzschild_ray` (e.g. ``dphi``,
    ``phi_max``, ``r_escape``) except ``b`` and ``m``, which are set per pixel.
    """
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    r_max = float(np.hypot(cx, cy))
    if r_max <= 0.0:
        r_max = 1.0

    ii = np.arange(width, dtype=float)
    jj = np.arange(height, dtype=float)
    ix, jy = np.meshgrid(ii, jj, indexing="xy")
    dx = ix - cx
    dy = jy - cy
    r_px = np.sqrt(dx * dx + dy * dy)
    b_map = b_min + (b_max - b_min) * (r_px / r_max)
    theta = np.arctan2(dy, dx)

    # Synthetic background: angle + mild vertical gradient (readable ring vs sky).
    bg_r = 0.5 + 0.5 * np.cos(theta)
    bg_g = 0.5 + 0.5 * np.sin(theta)
    denom = max(height - 1, 1)
    bg_b = np.clip(jy / denom, 0.0, 1.0)

    rgb = np.zeros((height, width, 3), dtype=np.float32)
    fallback = np.array([0.12, 0.12, 0.14], dtype=np.float32)

    for j in range(height):
        for i in range(width):
            b = float(b_map[j, i])
            result = trace_single_schwarzschild_ray(b=b, m=m, **trace_kwargs)
            if result.status == RayStatus.ESCAPED:
                rgb[j, i, 0] = bg_r[j, i]
                rgb[j, i, 1] = bg_g[j, i]
                rgb[j, i, 2] = bg_b[j, i]
            elif result.status == RayStatus.CAPTURED:
                rgb[j, i, :] = 0.0
            else:
                rgb[j, i, :] = fallback

    return rgb, b_map.astype(np.float32)


def write_ppm_rgb(path: str, rgb: np.ndarray) -> None:
    """Write a binary P6 PPM (RGB8) from float array (H, W, 3) in [0, 1]."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    h, w, _ = rgb.shape
    u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(u8.tobytes())
