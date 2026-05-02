"""Optional parity: C `bh_rt_schwarzschild_u_trace` vs `phase1.trace_single_schwarzschild_ray`."""

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

import pytest

from blackhole_ray_tracer.phase1 import trace_single_schwarzschild_ray

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernel"
INCLUDE = KERNEL / "include"

STATUS_FROM_PY: dict[str, int] = {
    "captured": 0,
    "escaped": 1,
    "max_steps": 2,
    "numerical_error": 3,
}


class SchwUResult(Structure):
    _fields_ = (
        ("status", c_int),
        ("termination_phi", c_double),
        ("termination_r", c_double),
        ("steps_taken", c_int),
        ("r_min", c_double),
    )


def _compiler() -> str | None:
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def _shared_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt" or sys.platform.startswith("win"):
        return ".dll"
    return ".so"


def _build_schwarzschild_u_shlib(dest: Path) -> bool:
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
        str(KERNEL / "src" / "bh_rt_schwarzschild_u.c"),
        "-o",
        str(dest),
        "-lm",
    ]
    if sys.platform not in {"win32", "cygwin"}:
        cmd.insert(cmd.index("-o"), "-fPIC")
    proc = subprocess.run(cmd, cwd=str(KERNEL), capture_output=True, text=True)
    return proc.returncode == 0


def _assert_compatible_r(py_r: float, c_r: float) -> None:
    py_r = float(py_r)
    c_r = float(c_r)
    if math.isinf(py_r):
        assert math.isinf(c_r) and math.copysign(1.0, py_r) == math.copysign(1.0, c_r)
    elif math.isnan(py_r):
        assert math.isnan(c_r)
    else:
        assert math.isclose(py_r, c_r, rel_tol=0.0, abs_tol=1e-11)


def _assert_r_min(py_rs, c_r_min: float) -> None:
    if py_rs.size == 0:
        assert math.isnan(float(c_r_min))
    else:
        assert math.isclose(float(py_rs.min()), float(c_r_min), rel_tol=0.0, abs_tol=1e-11)


@pytest.fixture(scope="module")
def schwarzschild_u_trace_fn(tmp_path_factory: pytest.TempPathFactory):
    if os.environ.get("SKIP_KERNEL_TESTS", "").strip().lower() in ("1", "true", "yes"):
        pytest.skip("SKIP_KERNEL_TESTS set")
    if _compiler() is None:
        pytest.skip("No C toolchain")
    build_dir = tmp_path_factory.mktemp("schwarzschild_u_shlib")
    lib_path = build_dir / ("libbh_rt_schwarzschild_u" + _shared_suffix())
    if not _build_schwarzschild_u_shlib(lib_path):
        pytest.skip("Kernel Schwarzschild compile failed")
    assert lib_path.is_file()

    lib = ctypes.CDLL(str(lib_path))
    fn = lib.bh_rt_schwarzschild_u_trace
    fn.argtypes = [
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        ctypes.POINTER(SchwUResult),
    ]
    fn.restype = None
    return fn


_DEFAULT_KW = {
    "m": 1.0,
    "b": 6.0,
    "phi_start": 0.2,
    "phi_max": 8.0,
    "dphi": 0.002,
    "r_escape": 80.0,
}


@pytest.mark.parametrize(
    "override",
    [
        {},
        {"b": 2.8, "dphi": 0.05, "phi_max": 4.0},
        {"dphi": 0.012},
        {"phi_max": 0.2},
    ],
)
def test_schwarzschild_u_parity_discrete(
    schwarzschild_u_trace_fn: Callable[..., None],
    override: dict,
) -> None:
    kw = {**_DEFAULT_KW, **override}
    py = trace_single_schwarzschild_ray(**kw)
    m = kw["m"]
    b = kw["b"]
    phi_start = kw["phi_start"]
    phi_max = kw["phi_max"]
    dphi = kw["dphi"]
    r_escape = kw["r_escape"]
    r_capture = kw.get("r_capture")
    if r_capture is None:
        r_capture = 2.0 * m + 1e-3

    out = SchwUResult()
    schwarzschild_u_trace_fn(
        m,
        b,
        phi_start,
        phi_max,
        dphi,
        r_capture,
        r_escape,
        byref(out),
    )

    assert STATUS_FROM_PY[py.status.value] == out.status
    assert py.steps_taken == out.steps_taken
    assert abs(py.termination_phi - out.termination_phi) <= 1e-11
    _assert_compatible_r(py.termination_r, out.termination_r)
    _assert_r_min(py.rs, out.r_min)
