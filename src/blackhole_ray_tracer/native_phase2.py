"""Optional native Phase 2 trace functions via `blackhole_ray_tracer._native_phase2`."""

from __future__ import annotations

from typing import Any

import numpy as np

from .phase1 import RayStatus

try:
    from blackhole_ray_tracer._native_phase2 import (
        schwarzschild_phase2_trace as _raw_trace,
        schwarzschild_phase2_batch_trace as _raw_batch_trace,
    )
except ImportError:  # pragma: no cover - extension missing on some installs
    _raw_trace = None
    _raw_batch_trace = None


def native_phase2_available() -> bool:
    return _raw_trace is not None


_NATIVE_STATUS_TO_RAY: dict[int, RayStatus] = {
    0: RayStatus.CAPTURED,
    1: RayStatus.ESCAPED,
    2: RayStatus.MAX_STEPS,
    3: RayStatus.NUMERICAL_ERROR,
}


def ray_status_from_native_phase2(result: dict[str, Any]) -> RayStatus:
    """Map C `bh_rt_phase2_trace_result.status` to Python `RayStatus`."""
    code = int(result["status"])
    if code not in _NATIVE_STATUS_TO_RAY:
        return RayStatus.NUMERICAL_ERROR
    return _NATIVE_STATUS_TO_RAY[code]


def ray_status_array_from_native(status_array: np.ndarray) -> list[RayStatus]:
    """Map an int32 status array (batch result) to a list of `RayStatus`."""
    return [_NATIVE_STATUS_TO_RAY.get(int(s), RayStatus.NUMERICAL_ERROR) for s in status_array]


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


def schwarzschild_phase2_batch_native(
    y0_batch: np.ndarray,
    m: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    *,
    r_horizon_epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Call the batched C kernel via PyBind11.

    Parameters
    ----------
    y0_batch : float64 array, shape (N, 8)
        Initial states for N rays, each row (t, r, θ, φ, v^t, v^r, v^θ, v^φ).

    Returns
    -------
    dict with numpy arrays:
        status          (N,) int32   — BH_RT_STATUS_* codes
        steps_taken     (N,) int32
        termination_r   (N,) float64
        r_min           (N,) float64
    """
    if _raw_batch_trace is None:
        raise RuntimeError(
            "Native Phase 2 extension not built (_native_phase2). "
            "Reinstall package with a C++/C toolchain (pip/uv install -e .)."
        )
    batch = np.asarray(y0_batch, dtype=np.float64)
    if batch.ndim != 2 or batch.shape[1] != 8:
        raise ValueError(f"y0_batch must be shape (N, 8), got {batch.shape}")
    batch = np.ascontiguousarray(batch)
    return _raw_batch_trace(
        batch, float(m), float(dlambda), int(max_steps), float(r_escape), float(r_horizon_epsilon)
    )


def batch_native_available() -> bool:
    """Return True if the native batch extension is importable."""
    return _raw_batch_trace is not None
