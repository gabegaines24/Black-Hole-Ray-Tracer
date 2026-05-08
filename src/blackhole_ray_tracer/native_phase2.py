"""Optional native Phase 2 single-ray trace via `blackhole_ray_tracer._native_phase2`."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from blackhole_ray_tracer._native_phase2 import schwarzschild_phase2_trace as _raw_trace
except ImportError:  # pragma: no cover - extension missing on some installs
    _raw_trace = None


def native_phase2_available() -> bool:
    return _raw_trace is not None


def schwarzschild_phase2_trace_native(
    y0: np.ndarray,
    m: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    *,
    r_horizon_epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Call C kernel `bh_rt_schwarzschild_phase2_trace` through PyBind11.

    `y0` must be float64, shape (8,), order (t, r, θ, φ, v^t, v^r, v^θ, v^φ).
    """
    if _raw_trace is None:
        raise RuntimeError(
            "Native Phase 2 extension not built (_native_phase2). "
            "Reinstall package with a C++/C toolchain (pip/uv install -e .)."
        )
    y = np.asarray(y0, dtype=np.float64).reshape((8,))
    return _raw_trace(
        y, float(m), float(dlambda), int(max_steps), float(r_escape), float(r_horizon_epsilon)
    )
