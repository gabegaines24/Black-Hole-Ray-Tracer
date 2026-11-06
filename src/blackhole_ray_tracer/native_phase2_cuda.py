"""Optional CUDA batch backend for Phase 2 null geodesic tracing.

The renderer selects CPU batch when CUDA is unavailable. Direct CUDA calls raise
explicit errors for missing libraries or failed host/device operations.

The CUDA library is NOT built by default (requires nvcc + GPU). Build with:

    make -C kernel/cuda cuda_batch

Then set BLACKHOLE_CUDA_LIB=/path/to/libbh_rt_phase2_cuda.so, or place the
file in build/libbh_rt_phase2_cuda.so at the repo root.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np

MAX_CUDA_BATCH_RAYS = 1_000_000

_LIB: ctypes.CDLL | None = None
_TRIED = False


def _find_cuda_lib() -> Path | None:
    env = os.environ.get("BLACKHOLE_CUDA_LIB", "").strip()
    if env:
        p = Path(env)
        return p if p.is_file() else None

    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parents[1]
    names = (
        "libbh_rt_phase2_cuda.so",
        "libbh_rt_phase2_cuda.dylib",
        "libbh_rt_phase2_cuda.dll",
    )
    roots = (package_dir, repo_root / "build", repo_root)
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None


def _load_cuda_lib() -> ctypes.CDLL | None:
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    lib_path = _find_cuda_lib()
    if lib_path is None:
        return None
    try:
        lib = ctypes.CDLL(str(lib_path))
        fn = lib.bh_rt_schwarzschild_phase2_batch_trace_cuda
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        fn.restype = ctypes.c_int32
        _LIB = lib
    except (OSError, AttributeError):
        _LIB = None
    return _LIB


def cuda_batch_available() -> bool:
    """Return True if the CUDA batch library is loadable."""
    return _load_cuda_lib() is not None


def _empty_cuda_result() -> dict[str, Any]:
    empty_i = np.empty(0, dtype=np.int32)
    empty_f = np.empty(0, dtype=np.float64)
    return {
        "status": empty_i,
        "steps_taken": empty_i,
        "termination_r": empty_f,
        "r_min": empty_f,
        "disk_hit": empty_i,
        "disk_r_cross": empty_f,
        "disk_vt": empty_f,
        "disk_vph": empty_f,
        "eq_r_cross": empty_f,
    }


def schwarzschild_phase2_batch_cuda(
    y0_batch: np.ndarray,
    m: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    *,
    r_horizon_epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Trace N null geodesics on the GPU via the CUDA batch kernel."""
    batch = np.asarray(y0_batch, dtype=np.float64)
    if batch.ndim != 2 or batch.shape[1] != 8:
        raise ValueError(f"y0_batch must be (N, 8), got {batch.shape}")
    n = int(batch.shape[0])
    if n > MAX_CUDA_BATCH_RAYS:
        raise ValueError(f"y0_batch exceeds CUDA batch limit {MAX_CUDA_BATCH_RAYS}.")
    if not np.all(np.isfinite(batch)):
        raise ValueError("y0_batch must contain only finite values.")
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError("m must be finite and positive.")
    if not np.isfinite(dlambda) or dlambda <= 0.0:
        raise ValueError("dlambda must be finite and positive.")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer.")
    if max_steps > np.iinfo(np.int32).max:
        raise ValueError("max_steps exceeds int32 ABI range.")
    if not np.isfinite(r_escape) or r_escape <= 0.0:
        raise ValueError("r_escape must be finite and positive.")
    if not np.isfinite(r_horizon_epsilon) or r_horizon_epsilon < 0.0:
        raise ValueError("r_horizon_epsilon must be finite and non-negative.")
    if n == 0:
        return _empty_cuda_result()
    lib = _load_cuda_lib()
    if lib is None:
        raise RuntimeError(
            "CUDA batch library not found. Build with `make -C kernel/cuda cuda_batch` "
            "and set BLACKHOLE_CUDA_LIB=<path>."
        )
    batch = np.ascontiguousarray(batch)
    flat = batch.ravel()
    c_y0 = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_status = np.empty(n, dtype=np.int32)
    out_steps = np.empty(n, dtype=np.int32)

    out_term_r = np.empty(n, dtype=np.float64)
    out_r_min = np.empty(n, dtype=np.float64)
    out_disk_hit = np.empty(n, dtype=np.int32)
    out_disk_r = np.empty(n, dtype=np.float64)
    out_disk_vt = np.empty(n, dtype=np.float64)
    out_disk_vph = np.empty(n, dtype=np.float64)

    error_code = lib.bh_rt_schwarzschild_phase2_batch_trace_cuda(
        c_y0,
        ctypes.c_int32(n),
        ctypes.c_double(m),
        ctypes.c_double(dlambda),
        ctypes.c_int32(max_steps),
        ctypes.c_double(r_escape),
        ctypes.c_double(r_horizon_epsilon),
        out_status.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_steps.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_term_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_r_min.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_disk_hit.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_disk_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_disk_vt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_disk_vph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if error_code != 0:
        raise RuntimeError(f"CUDA batch execution failed with error code {error_code}.")
    return {
        "status": out_status,
        "steps_taken": out_steps,
        "termination_r": out_term_r,
        "r_min": out_r_min,
        "disk_hit": out_disk_hit,
        "disk_r_cross": out_disk_r,
        "disk_vt": out_disk_vt,
        "disk_vph": out_disk_vph,
        "eq_r_cross": out_disk_r,
    }
