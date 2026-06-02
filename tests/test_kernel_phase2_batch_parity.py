"""Optional parity: C `bh_rt_schwarzschild_phase2_batch_trace` vs N Python traces."""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from blackhole_ray_tracer.phase1 import RayStatus
from blackhole_ray_tracer.phase2_camera import (
    make_camera_from_config,
    initial_position_observer,
    static_observer_null_direction,
)
from blackhole_ray_tracer.phase2_geodesic import trace_null_geodesic_3d

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernel"
INCLUDE = KERNEL / "include"

STATUS_PY_TO_C = {
    RayStatus.CAPTURED: 0,
    RayStatus.ESCAPED: 1,
    RayStatus.MAX_STEPS: 2,
    RayStatus.NUMERICAL_ERROR: 3,
}


def _compiler() -> str | None:
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def _shared_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt" or sys.platform.startswith("win"):
        return ".dll"
    return ".so"


def _build_batch_shlib(dest: Path) -> bool:
    cc = _compiler()
    if cc is None:
        return False
    cmd = [
        cc,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-O2",
        "-shared",
        f"-I{INCLUDE}",
        str(KERNEL / "src" / "bh_rt_rk4.c"),
        str(KERNEL / "src" / "bh_rt_schwarzschild_phase2.c"),
        str(KERNEL / "src" / "bh_rt_schwarzschild_phase2_batch.c"),
        "-o",
        str(dest),
        "-lm",
    ]
    if sys.platform not in {"win32", "cygwin"}:
        cmd.insert(cmd.index("-o"), "-fPIC")
    proc = subprocess.run(cmd, cwd=str(KERNEL), capture_output=True, text=True)
    return proc.returncode == 0


@pytest.fixture(scope="module")
def batch_trace_fn(tmp_path_factory: pytest.TempPathFactory):
    if os.environ.get("SKIP_KERNEL_TESTS", "").strip().lower() in ("1", "true", "yes"):
        pytest.skip("SKIP_KERNEL_TESTS set")
    if _compiler() is None:
        pytest.skip("No C toolchain")
    build_dir = tmp_path_factory.mktemp("phase2_batch_shlib")
    lib_path = build_dir / ("libbh_rt_phase2_batch" + _shared_suffix())
    if not _build_batch_shlib(lib_path):
        pytest.skip("Kernel phase2 batch compile failed")

    lib = ctypes.CDLL(str(lib_path))
    fn = lib.bh_rt_schwarzschild_phase2_batch_trace
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
    return fn


def _build_grid_y0(width: int, height: int, m: float = 1.0, r_obs: float = 30.0,
                   fov_deg: float = 60.0):
    """Return y0 (N,8) array and per-pixel Python trace results."""
    cam = make_camera_from_config(m, r=r_obs, theta=np.pi / 2, phi=0.0,
                                  fov_deg=fov_deg, width=width, height=height)
    x0 = initial_position_observer(cam)
    rays = []
    for j in range(height):
        for i in range(width):
            sx = 2.0 * (i + 0.5) / width - 1.0
            sy = 1.0 - 2.0 * (j + 0.5) / height
            v0 = static_observer_null_direction(cam, sx, sy)
            rays.append(np.concatenate([x0, v0]))
    return np.stack(rays, axis=0).astype(np.float64)  # (N, 8)


def _run_python_batch(y0: np.ndarray, m: float, dlambda: float, max_steps: int,
                      r_escape: float, r_horizon_epsilon: float):
    results = []
    for row in y0:
        r = trace_null_geodesic_3d(
            row[:4], row[4:],
            m=m, dlambda=dlambda, max_steps=max_steps,
            r_escape=r_escape, r_horizon_epsilon=r_horizon_epsilon,
            store_samples=False,
        )
        results.append(r)
    return results


def _call_c_batch(fn, y0: np.ndarray, m: float, dlambda: float, max_steps: int,
                  r_escape: float, r_horizon_epsilon: float):
    n = len(y0)
    flat = np.ascontiguousarray(y0, dtype=np.float64).ravel()
    c_y0 = (ctypes.c_double * len(flat))(*flat.tolist())
    c_status  = (ctypes.c_int * n)()
    c_steps   = (ctypes.c_int * n)()
    c_term_r  = (ctypes.c_double * n)()
    c_r_min   = (ctypes.c_double * n)()
    fn(c_y0, n, m, dlambda, max_steps, r_escape, r_horizon_epsilon,
       c_status, c_steps, c_term_r, c_r_min)
    return (
        np.array(list(c_status), dtype=np.int32),
        np.array(list(c_steps),  dtype=np.int32),
        np.array(list(c_term_r), dtype=np.float64),
        np.array(list(c_r_min),  dtype=np.float64),
    )


def _close(a: float, b: float, atol: float) -> None:
    if math.isnan(a) and math.isnan(b):
        return
    assert math.isclose(a, b, rel_tol=0.0, abs_tol=atol), f"{a} != {b} (atol={atol})"


@pytest.mark.parametrize("width,height,dlambda,max_steps", [
    (4, 4, 0.1,  3000),
    (2, 3, 0.12,  500),
])
def test_phase2_batch_parity_grid(batch_trace_fn, width, height, dlambda, max_steps):
    m, r_escape, r_hep = 1.0, 80.0, 1e-3
    y0 = _build_grid_y0(width, height, m=m)

    py_results = _run_python_batch(y0, m, dlambda, max_steps, r_escape, r_hep)
    c_status, c_steps, c_term_r, c_r_min = _call_c_batch(
        batch_trace_fn, y0, m, dlambda, max_steps, r_escape, r_hep)

    for i, py in enumerate(py_results):
        assert STATUS_PY_TO_C[py.status] == int(c_status[i]), \
            f"ray {i}: status py={py.status.value} c={c_status[i]}"
        assert py.steps_taken == int(c_steps[i]), \
            f"ray {i}: steps py={py.steps_taken} c={c_steps[i]}"
        _close(py.termination_r, float(c_term_r[i]), atol=1e-8)
        _close(py.r_min, float(c_r_min[i]), atol=1e-8)
