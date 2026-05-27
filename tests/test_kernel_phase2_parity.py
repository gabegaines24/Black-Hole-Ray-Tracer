"""Optional parity: C `bh_rt_schwarzschild_phase2_trace` vs Python `trace_null_geodesic_3d`."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
import math
import os
import shutil
import subprocess
import sys
from ctypes import Structure, byref, c_double, c_int
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


class Phase2TraceResult(Structure):
    _fields_ = (
        ("status", c_int),
        ("steps_taken", c_int),
        ("max_steps", c_int),
        ("termination_r", c_double),
        ("termination_lambda", c_double),
        ("r_min", c_double),
    )


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


def _build_phase2_shlib(dest: Path) -> bool:
    cc = _compiler()
    if cc is None:
        return False
    cmd: list[str] = [
        cc,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-O2",
        "-shared",
        f"-I{INCLUDE}",
        str(KERNEL / "src" / "bh_rt_rk4.c"),
        str(KERNEL / "src" / "bh_rt_schwarzschild_phase2.c"),
        "-o",
        str(dest),
        "-lm",
    ]
    if sys.platform not in {"win32", "cygwin"}:
        cmd.insert(cmd.index("-o"), "-fPIC")
    proc = subprocess.run(cmd, cwd=str(KERNEL), capture_output=True, text=True)
    return proc.returncode == 0


def _close(a: float, b: float, *, atol: float) -> None:
    if math.isnan(a) and math.isnan(b):
        return
    assert pytest.approx(a, abs=atol) == b


@pytest.fixture(scope="module")
def phase2_trace_fn(tmp_path_factory: pytest.TempPathFactory):
    if os.environ.get("SKIP_KERNEL_TESTS", "").strip().lower() in ("1", "true", "yes"):
        pytest.skip("SKIP_KERNEL_TESTS set")
    if _compiler() is None:
        pytest.skip("No C toolchain")
    build_dir = tmp_path_factory.mktemp("phase2_shlib")
    lib_path = build_dir / ("libbh_rt_schwarzschild_phase2" + _shared_suffix())
    if not _build_phase2_shlib(lib_path):
        pytest.skip("Kernel Phase 2 compile failed")
    assert lib_path.is_file()

    lib = ctypes.CDLL(str(lib_path))
    fn = lib.bh_rt_schwarzschild_phase2_trace
    fn.argtypes = [
        ctypes.POINTER(c_double),
        c_double,
        c_double,
        c_int,
        c_double,
        c_double,
        ctypes.POINTER(Phase2TraceResult),
    ]
    fn.restype = None
    return fn


def _run_both(
    phase2_trace_fn: Callable[..., None],
    *,
    m: float,
    y0: np.ndarray,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    r_horizon_epsilon: float = 1e-3,
) -> tuple[object, Phase2TraceResult]:
    py = trace_null_geodesic_3d(
        y0[:4],
        y0[4:],
        m=m,
        dlambda=dlambda,
        max_steps=max_steps,
        r_escape=r_escape,
        r_horizon_epsilon=r_horizon_epsilon,
        store_samples=False,
    )
    arr = (c_double * 8)(*y0.astype(float).tolist())
    out = Phase2TraceResult()
    phase2_trace_fn(
        arr,
        c_double(m),
        c_double(dlambda),
        c_int(max_steps),
        c_double(r_escape),
        c_double(r_horizon_epsilon),
        byref(out),
    )
    return py, out


def test_phase2_parity_center_pixel(phase2_trace_fn: Callable[..., None]) -> None:
    m = 1.0
    cam = make_camera_from_config(m, r=30.0, theta=np.pi / 2, phi=0.0, fov_deg=60.0, width=16, height=16)
    x0 = initial_position_observer(cam)
    v0 = static_observer_null_direction(cam, 0.0, 0.0)
    y0 = np.concatenate([x0, v0])
    py, c = _run_both(
        phase2_trace_fn,
        m=m,
        y0=y0,
        dlambda=0.1,
        max_steps=3000,
        r_escape=80.0,
    )
    assert STATUS_PY_TO_C[py.status] == c.status
    assert py.steps_taken == c.steps_taken
    _close(py.termination_r, c.termination_r, atol=1e-9)
    _close(py.termination_lambda, c.termination_lambda, atol=1e-7)
    _close(py.r_min, c.r_min, atol=1e-9)


@pytest.mark.parametrize(
    "dlambda,max_steps",
    [(0.12, 500), (0.06, 1200), (0.2, 400)],
)
def test_phase2_parity_param(
    phase2_trace_fn: Callable[..., None],
    dlambda: float,
    max_steps: int,
) -> None:
    m = 1.0
    cam = make_camera_from_config(m, r=25.0, theta=np.pi / 2, phi=0.1, fov_deg=50.0, width=8, height=8)
    x0 = initial_position_observer(cam)
    v0 = static_observer_null_direction(cam, 2.0, -1.0)
    y0 = np.concatenate([x0, v0])
    py, c = _run_both(
        phase2_trace_fn,
        m=m,
        y0=y0,
        dlambda=dlambda,
        max_steps=max_steps,
        r_escape=100.0,
        r_horizon_epsilon=5e-4,
    )
    assert STATUS_PY_TO_C[py.status] == c.status
    assert py.steps_taken == c.steps_taken
    _close(py.termination_r, c.termination_r, atol=1e-8)
    _close(py.termination_lambda, c.termination_lambda, atol=1e-6)
    _close(py.r_min, c.r_min, atol=1e-8)
