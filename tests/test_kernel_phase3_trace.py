"""C/Python parity for the bounded Kerr Phase 3 single-ray tracer."""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from ctypes import Structure, byref, c_double, c_int32
from pathlib import Path

import numpy as np
import pytest

from blackhole_ray_tracer.phase1 import RayStatus
from blackhole_ray_tracer.phase3_camera import (
    initial_position_observer,
    make_kerr_camera_from_config,
    zamo_null_direction,
)
from blackhole_ray_tracer.phase3_geodesic import trace_kerr_null_geodesic
from blackhole_ray_tracer.phase3_christoffel import kerr_outer_horizon

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernel"
INCLUDE = KERNEL / "include"


class KerrTraceResult(Structure):
    _fields_ = (
        ("status", c_int32),
        ("steps_taken", c_int32),
        ("max_steps", c_int32),
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


class KerrError:
    OK = 0
    INVALID_ARGUMENT = 1
    DOMAIN_ERROR = 2


def _compiler() -> str | None:
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def _shared_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt" or sys.platform.startswith("win"):
        return ".dll"
    return ".so"


@pytest.fixture(scope="module")
def kerr_trace_fn(tmp_path_factory: pytest.TempPathFactory):
    if os.environ.get("SKIP_KERNEL_TESTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip("SKIP_KERNEL_TESTS set")
    compiler = _compiler()
    if compiler is None:
        pytest.skip("No C toolchain")
    build_dir = tmp_path_factory.mktemp("kerr_trace")
    lib_path = build_dir / ("libbh_rt_kerr_phase3" + _shared_suffix())
    cmd = [
        compiler,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-O2",
        "-shared",
        f"-I{INCLUDE}",
        str(KERNEL / "src" / "bh_rt_kerr_phase3.c"),
        "-o",
        str(lib_path),
        "-lm",
    ]
    if sys.platform not in {"win32", "cygwin"}:
        cmd.insert(cmd.index("-o"), "-fPIC")
    proc = subprocess.run(cmd, cwd=str(KERNEL), capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"Kerr tracer compile failed:\n{proc.stdout}\n{proc.stderr}")

    lib = ctypes.CDLL(str(lib_path))
    fn = lib.bh_rt_kerr_phase3_trace
    fn.argtypes = [
        ctypes.POINTER(c_double),
        c_double,
        c_double,
        c_double,
        c_int32,
        c_double,
        c_double,
        ctypes.POINTER(KerrTraceResult),
    ]
    fn.restype = c_int32
    return fn


def _close(a: float, b: float, *, atol: float) -> None:
    if math.isnan(a) and math.isnan(b):
        return
    assert pytest.approx(a, abs=atol) == b


def _run_both(
    trace_fn: Callable[..., int],
    *,
    y0: np.ndarray,
    m: float,
    spin: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    r_horizon_epsilon: float = 1e-3,
):
    py = trace_kerr_null_geodesic(
        y0[:4],
        y0[4:],
        m=m,
        spin=spin,
        dlambda=dlambda,
        max_steps=max_steps,
        r_escape=r_escape,
        r_horizon_epsilon=r_horizon_epsilon,
        store_samples=False,
    )
    arr = (c_double * 8)(*y0.astype(float).tolist())
    out = KerrTraceResult()
    code = trace_fn(
        arr,
        c_double(m),
        c_double(spin),
        c_double(dlambda),
        c_int32(max_steps),
        c_double(r_escape),
        c_double(r_horizon_epsilon),
        byref(out),
    )
    assert code == KerrError.OK
    return py, out


def test_kerr_trace_terminal_statuses_match(kerr_trace_fn: Callable[..., int]) -> None:
    cases = (
        (
            np.array([0.0, 1.5, 1.1, 0.0, 1.0, 0.0, 0.0, 0.0]),
            10,
            RayStatus.CAPTURED,
        ),
        (
            np.array([0.0, 81.0, 1.1, 0.0, 1.0, 0.0, 0.0, 0.0]),
            10,
            RayStatus.ESCAPED,
        ),
        (
            np.array([0.0, 30.0, 1.1, 0.0, 1.0, 0.0, 0.0, 0.0]),
            0,
            RayStatus.MAX_STEPS,
        ),
    )
    for y0, max_steps, expected in cases:
        py, c = _run_both(
            kerr_trace_fn,
            y0=y0,
            m=1.0,
            spin=0.5,
            dlambda=0.1,
            max_steps=max_steps,
            r_escape=80.0,
        )
        assert py.status == expected
        assert c.status == STATUS_PY_TO_C[expected]
        assert py.steps_taken == c.steps_taken
        _close(py.termination_r, c.termination_r, atol=1e-10)
        _close(py.termination_lambda, c.termination_lambda, atol=1e-10)
        _close(py.r_min, c.r_min, atol=1e-10)


def test_kerr_trace_matches_python_reference(
    kerr_trace_fn: Callable[..., int],
) -> None:
    for spin in (0.0, 0.5, -0.5, 0.999999):
        camera = make_kerr_camera_from_config(
            1.0,
            spin,
            r=30.0,
            theta=np.pi / 3.0,
            phi=0.0,
            fov_deg=60.0,
            width=8,
            height=8,
        )
        y0 = np.concatenate(
            [
                initial_position_observer(camera),
                zamo_null_direction(camera, 0.2, -0.3),
            ]
        )
        py, c = _run_both(
            kerr_trace_fn,
            y0=y0,
            m=1.0,
            spin=spin,
            dlambda=0.02,
            max_steps=250,
            r_escape=80.0,
        )
        assert c.status == STATUS_PY_TO_C[py.status]
        assert c.steps_taken == py.steps_taken
        _close(py.termination_r, c.termination_r, atol=1e-8)
        _close(py.termination_lambda, c.termination_lambda, atol=1e-10)
        _close(py.r_min, c.r_min, atol=1e-8)


def test_kerr_trace_capture_boundary_matches_python(
    kerr_trace_fn: Callable[..., int],
) -> None:
    m = 1.0
    spin = 0.9
    horizon = kerr_outer_horizon(m, spin)
    camera = make_kerr_camera_from_config(
        m,
        spin,
        r=horizon + 0.2,
        theta=np.pi / 3.0,
        phi=0.0,
        fov_deg=60.0,
        width=8,
        height=8,
    )
    y0 = np.concatenate(
        [
            initial_position_observer(camera),
            zamo_null_direction(camera, 0.0, 0.0),
        ]
    )
    py, c = _run_both(
        kerr_trace_fn,
        y0=y0,
        m=m,
        spin=spin,
        dlambda=0.002,
        max_steps=20_000,
        r_escape=80.0,
    )
    assert py.status == RayStatus.CAPTURED
    assert c.status == STATUS_PY_TO_C[py.status]
    assert c.steps_taken == py.steps_taken
    _close(py.termination_r, c.termination_r, atol=2e-7)
    _close(py.termination_lambda, c.termination_lambda, atol=2e-7)
    _close(py.r_min, c.r_min, atol=2e-7)


def test_kerr_trace_rejects_invalid_parameters(kerr_trace_fn: Callable[..., int]) -> None:
    y0 = (c_double * 8)(0.0, 30.0, 1.1, 0.0, 1.0, 0.0, 0.0, 0.0)
    out = KerrTraceResult()
    assert (
        kerr_trace_fn(
            y0,
            1.0,
            0.5,
            0.1,
            -1,
            80.0,
            1e-3,
            byref(out),
        )
        == KerrError.DOMAIN_ERROR
    )
    assert (
        kerr_trace_fn(
            y0,
            1.0,
            0.5,
            0.1,
            10,
            1.0,
            1e-3,
            byref(out),
        )
        == KerrError.DOMAIN_ERROR
    )
    y0[4] = math.nan
    assert (
        kerr_trace_fn(
            y0,
            1.0,
            0.5,
            0.1,
            10,
            80.0,
            1e-3,
            byref(out),
        )
        == KerrError.OK
    )
    assert out.status == STATUS_PY_TO_C[RayStatus.NUMERICAL_ERROR]
    assert out.steps_taken == 0
