"""Optional parity: C 3D Schwarzschild trace vs `phase2_geodesic`."""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from ctypes import Structure, byref, c_double, c_int
from pathlib import Path

import numpy as np
import pytest

from blackhole_ray_tracer.phase2_camera import (
    initial_position_observer,
    make_camera_from_config,
    static_observer_null_direction,
)
from blackhole_ray_tracer.phase2_geodesic import trace_null_geodesic_3d

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernel"
INCLUDE = KERNEL / "include"

STATUS_FROM_PY: dict[str, int] = {
    "captured": 0,
    "escaped": 1,
    "max_steps": 2,
    "numerical_error": 3,
}


class Schw3DResult(Structure):
    _fields_ = (
        ("status", c_int),
        ("steps_taken", c_int),
        ("max_steps", c_int),
        ("termination_r", c_double),
        ("termination_lambda", c_double),
        ("r_min", c_double),
        ("final_state", c_double * 8),
    )


def _compiler() -> str | None:
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def _shared_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt" or sys.platform.startswith("win"):
        return ".dll"
    return ".so"


def _build_schwarzschild_3d_shlib(dest: Path) -> bool:
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
        str(KERNEL / "src" / "bh_rt_schwarzschild_3d.c"),
        "-o",
        str(dest),
        "-lm",
    ]
    if sys.platform not in {"win32", "cygwin"}:
        cmd.insert(cmd.index("-o"), "-fPIC")
    proc = subprocess.run(cmd, cwd=str(KERNEL), capture_output=True, text=True)
    return proc.returncode == 0


@pytest.fixture(scope="module")
def schwarzschild_3d_trace_fn(tmp_path_factory: pytest.TempPathFactory):
    if os.environ.get("SKIP_KERNEL_TESTS", "").strip().lower() in ("1", "true", "yes"):
        pytest.skip("SKIP_KERNEL_TESTS set")
    if _compiler() is None:
        pytest.skip("No C toolchain")
    build_dir = tmp_path_factory.mktemp("schwarzschild_3d_shlib")
    lib_path = build_dir / ("libbh_rt_schwarzschild_3d" + _shared_suffix())
    if not _build_schwarzschild_3d_shlib(lib_path):
        pytest.skip("Kernel Schwarzschild 3D compile failed")
    assert lib_path.is_file()

    lib = ctypes.CDLL(str(lib_path))
    fn = lib.bh_rt_schwarzschild_3d_trace
    fn.argtypes = [
        ctypes.POINTER(c_double),
        ctypes.POINTER(c_double),
        c_double,
        c_double,
        c_int,
        c_double,
        c_double,
        ctypes.POINTER(Schw3DResult),
    ]
    fn.restype = None
    return fn


def _assert_float_close_or_same_special(py_value: float, c_value: float, *, abs_tol: float) -> None:
    py_value = float(py_value)
    c_value = float(c_value)
    if math.isnan(py_value):
        assert math.isnan(c_value)
    elif math.isinf(py_value):
        assert math.isinf(c_value) and math.copysign(1.0, py_value) == math.copysign(1.0, c_value)
    else:
        assert math.isclose(py_value, c_value, rel_tol=0.0, abs_tol=abs_tol)


@pytest.mark.parametrize(
    "screen_xy,dlambda,max_steps,r_escape",
    [
        ((0.0, 0.0), 0.1, 3000, 80.0),
        ((0.35, 0.0), 0.08, 2500, 80.0),
        ((0.0, 0.45), 0.08, 2500, 80.0),
        ((0.6, 0.6), 0.12, 1200, 60.0),
    ],
)
def test_schwarzschild_3d_trace_parity(
    schwarzschild_3d_trace_fn: Callable[..., None],
    screen_xy: tuple[float, float],
    dlambda: float,
    max_steps: int,
    r_escape: float,
) -> None:
    m = 1.0
    eps = 1e-3
    cam = make_camera_from_config(
        m=m,
        r=30.0,
        theta=float(np.pi / 2),
        phi=0.0,
        fov_deg=60.0,
        width=16,
        height=16,
    )
    x0 = initial_position_observer(cam)
    v0 = static_observer_null_direction(cam, screen_xy[0], screen_xy[1])

    py = trace_null_geodesic_3d(
        x0,
        v0,
        m=m,
        dlambda=dlambda,
        max_steps=max_steps,
        r_escape=r_escape,
        r_horizon_epsilon=eps,
        store_samples=False,
    )

    x_arr = (c_double * 4)(*map(float, x0))
    v_arr = (c_double * 4)(*map(float, v0))
    out = Schw3DResult()
    schwarzschild_3d_trace_fn(
        x_arr,
        v_arr,
        m,
        dlambda,
        max_steps,
        r_escape,
        eps,
        byref(out),
    )

    assert STATUS_FROM_PY[py.status.value] == out.status
    assert py.steps_taken == out.steps_taken
    assert py.max_steps == out.max_steps
    _assert_float_close_or_same_special(py.termination_r, out.termination_r, abs_tol=1e-8)
    _assert_float_close_or_same_special(
        py.termination_lambda, out.termination_lambda, abs_tol=1e-10
    )
    _assert_float_close_or_same_special(py.r_min, out.r_min, abs_tol=1e-8)
