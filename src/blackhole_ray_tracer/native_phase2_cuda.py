"""Optional CUDA batch backend for Phase 2 null geodesic tracing.

Falls back to CPU batch automatically if the CUDA shared library is not found.

The CUDA library is NOT built by default (requires nvcc + GPU).  Build with:

    nvcc -O2 -arch=sm_75 -Ikernel/include --compiler-options -fPIC -shared \\
         -o build/libbh_rt_phase2_cuda.so \\
         kernel/cuda/bh_rt_schwarzschild_phase2_batch.cu

Then set BLACKHOLE_CUDA_LIB=/path/to/libbh_rt_phase2_cuda.so, or place the
file in the package root.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np

_LIB: ctypes.CDLL | None = None
_TRIED = False


def _find_cuda_lib() -> Path | None:
    env = os.environ.get("BLACKHOLE_CUDA_LIB", "").strip()
    if env:
        p = Path(env)
        return p if p.exists() else None
    # Search standard locations relative to the package
    candidates = [
        Path(__file__).resolve().parents[2] / "build" / "libbh_rt_phase2_cuda.so",
        Path(__file__).resolve().parents[2] / "build" / "libbh_rt_phase2_cuda.dll",
        Path(__file__).resolve().parents[2] / "libbh_rt_phase2_cuda.so",
    ]
    for c in candidates:
        if c.exists():
            return c
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
            ctypes.POINTER(ctypes.c_double),  # y0
            ctypes.c_int,                     # n
            ctypes.c_double,                  # m
            ctypes.c_double,                  # dlambda
            ctypes.c_int,                     # max_steps
            ctypes.c_double,                  # r_escape
            ctypes.c_double,                  # r_horizon_epsilon
            ctypes.POINTER(ctypes.c_int),     # out_status
            ctypes.POINTER(ctypes.c_int),     # out_steps_taken
            ctypes.POINTER(ctypes.c_double),  # out_termination_r
            ctypes.POINTER(ctypes.c_double),  # out_r_min
        ]
        fn.restype = None
        _LIB = lib
    except (OSError, AttributeError):
        _LIB = None
    return _LIB


def cuda_batch_available() -> bool:
    """Return True if the CUDA batch library is loadable."""
    return _load_cuda_lib() is not None


def schwarzschild_phase2_batch_cuda(
    y0_batch: np.ndarray,
    m: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    *,
    r_horizon_epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Trace N null geodesics on the GPU via the CUDA batch kernel.

    Parameters
    ----------
    y0_batch : float64 (N, 8) C-contiguous

    Returns
    -------
    dict with status(N,int32), steps_taken(N,int32),
         termination_r(N,float64), r_min(N,float64)
    """
    lib = _load_cuda_lib()
    if lib is None:
        raise RuntimeError(
            "CUDA batch library not found. Build kernel/cuda/bh_rt_schwarzschild_phase2_batch.cu "
            "with nvcc and set BLACKHOLE_CUDA_LIB=<path>."
        )
    batch = np.ascontiguousarray(y0_batch, dtype=np.float64)
    if batch.ndim != 2 or batch.shape[1] != 8:
        raise ValueError(f"y0_batch must be (N, 8), got {batch.shape}")
    n = batch.shape[0]
    flat = batch.ravel()
    c_y0 = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    out_status  = np.empty(n, dtype=np.int32)
    out_steps   = np.empty(n, dtype=np.int32)
    out_term_r  = np.empty(n, dtype=np.float64)
    out_r_min   = np.empty(n, dtype=np.float64)

    lib.bh_rt_schwarzschild_phase2_batch_trace_cuda(
        c_y0, ctypes.c_int(n),
        ctypes.c_double(m), ctypes.c_double(dlambda),
        ctypes.c_int(max_steps), ctypes.c_double(r_escape),
        ctypes.c_double(r_horizon_epsilon),
        out_status.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_steps.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_term_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_r_min.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return {
        "status":        out_status,
        "steps_taken":   out_steps,
        "termination_r": out_term_r,
        "r_min":         out_r_min,
    }
