"""Optional native Phase 3 Kerr trace functions via `blackhole_ray_tracer._native_phase3`."""

from __future__ import annotations

from typing import Any

import numpy as np

from .native_phase2 import ray_status_array_from_native, ray_status_from_native_phase2
from .phase1 import RayStatus

try:
    from blackhole_ray_tracer._native_phase3 import (
        kerr_phase3_batch_trace as _raw_batch_trace,
        kerr_phase3_trace as _raw_trace,
    )
except ImportError:  # pragma: no cover - extension missing on some installs
    _raw_trace = None
    _raw_batch_trace = None


def native_phase3_available() -> bool:
    return _raw_trace is not None


def batch_native_phase3_available() -> bool:
    return _raw_batch_trace is not None


def kerr_phase3_trace_native(
    y0: np.ndarray,
    m: float,
    a: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    *,
    r_horizon_epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Call C kernel `bh_rt_kerr_phase3_trace` through PyBind11."""
    if _raw_trace is None:
        raise RuntimeError(
            "Native Phase 3 extension not built (_native_phase3). "
            "Reinstall package with a C++/C toolchain (pip/uv install -e .)."
        )
    y = np.asarray(y0, dtype=np.float64).reshape((8,))
    return _raw_trace(
        y,
        float(m),
        float(a),
        float(dlambda),
        int(max_steps),
        float(r_escape),
        float(r_horizon_epsilon),
    )


def kerr_phase3_batch_native(
    y0_batch: np.ndarray,
    m: float,
    a: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    *,
    r_horizon_epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Call the batched Kerr C kernel via PyBind11."""
    if _raw_batch_trace is None:
        raise RuntimeError(
            "Native Phase 3 extension not built (_native_phase3). "
            "Reinstall package with a C++/C toolchain (pip/uv install -e .)."
        )
    batch = np.asarray(y0_batch, dtype=np.float64)
    if batch.ndim != 2 or batch.shape[1] != 8:
        raise ValueError(f"y0_batch must be shape (N, 8), got {batch.shape}")
    batch = np.ascontiguousarray(batch)
    return _raw_batch_trace(
        batch,
        float(m),
        float(a),
        float(dlambda),
        int(max_steps),
        float(r_escape),
        float(r_horizon_epsilon),
    )


__all__ = [
    "RayStatus",
    "batch_native_phase3_available",
    "kerr_phase3_batch_native",
    "kerr_phase3_trace_native",
    "native_phase3_available",
    "ray_status_array_from_native",
    "ray_status_from_native_phase2",
]
