"""C/Python parity for Kerr equatorial crossing observables."""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
from ctypes import Structure, byref, c_double, c_int32
from pathlib import Path

import numpy as np
import pytest

from blackhole_ray_tracer.phase1 import RayStatus
from blackhole_ray_tracer.phase2_disk import detect_equatorial_crossing_state
from blackhole_ray_tracer.phase3_camera import (
    initial_position_observer,
    make_kerr_camera_from_config,
    zamo_null_direction,
)
from blackhole_ray_tracer.phase3_geodesic import trace_kerr_null_geodesic

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


class KerrCrossing(Structure):
    _fields_ = (
        ("hit", c_int32),
        ("r_cross", c_double),
        ("vt_cross", c_double),
        ("vphi_cross", c_double),
    )


class KerrError:
    OK = 0
    INVALID_ARGUMENT = 1


@pytest.fixture(scope="module")
def kerr_crossing_fn(tmp_path_factory: pytest.TempPathFactory):
    if os.environ.get("SKIP_KERNEL_TESTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip("SKIP_KERNEL_TESTS set")
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        pytest.skip("No C toolchain")
    build_dir = tmp_path_factory.mktemp("kerr_crossing")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    lib_path = build_dir / ("libbh_rt_kerr_phase3" + suffix)
    cmd = [
        compiler,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-O2",
        "-shared",
        "-fPIC",
        f"-I{INCLUDE}",
        str(KERNEL / "src" / "bh_rt_kerr_phase3.c"),
        "-o",
        str(lib_path),
        "-lm",
    ]
    proc = subprocess.run(cmd, cwd=str(KERNEL), capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"Kerr crossing compile failed:\n{proc.stdout}\n{proc.stderr}")

    lib = ctypes.CDLL(str(lib_path))
    fn = lib.bh_rt_kerr_phase3_trace_crossing
    fn.argtypes = [
        ctypes.POINTER(c_double),
        c_double,
        c_double,
        c_double,
        c_int32,
        c_double,
        c_double,
        ctypes.POINTER(KerrTraceResult),
        ctypes.POINTER(KerrCrossing),
    ]
    fn.restype = c_int32
    return fn


def _run_both(fn, spin: float, max_steps: int):
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
            zamo_null_direction(camera, 1.0, 0.0),
        ]
    )
    py = trace_kerr_null_geodesic(
        y0[:4],
        y0[4:],
        m=1.0,
        spin=spin,
        dlambda=0.05,
        max_steps=max_steps,
        r_escape=80.0,
        store_samples=True,
    )
    expected = detect_equatorial_crossing_state(
        py.theta_samples, py.r_samples, py.vt_samples, py.vphi_samples
    )
    state = (c_double * 8)(*y0.tolist())
    trace = KerrTraceResult()
    crossing = KerrCrossing()
    code = fn(
        state,
        1.0,
        spin,
        0.05,
        max_steps,
        80.0,
        1e-3,
        byref(trace),
        byref(crossing),
    )
    assert code == KerrError.OK
    return py, expected, trace, crossing


@pytest.mark.parametrize("spin", [0.0, 0.5, -0.5, 0.999999])
def test_kerr_first_equatorial_crossing_matches_python(
    kerr_crossing_fn, spin: float
) -> None:
    py, expected, trace, crossing = _run_both(kerr_crossing_fn, spin, 400)
    assert py.status == RayStatus.MAX_STEPS
    assert trace.status == 2
    assert expected is not None
    assert crossing.hit == 1
    assert crossing.r_cross == pytest.approx(expected.r_cross, abs=2e-10)
    assert crossing.vt_cross == pytest.approx(expected.vt_cross, abs=2e-10)
    assert crossing.vphi_cross == pytest.approx(expected.vph_cross, abs=2e-10)


def test_kerr_crossing_stays_empty_without_steps(kerr_crossing_fn) -> None:
    camera = make_kerr_camera_from_config(
        1.0,
        0.5,
        r=30.0,
        theta=np.pi / 3.0,
        phi=0.0,
        fov_deg=60.0,
        width=8,
        height=8,
    )
    y0 = np.concatenate(
        [initial_position_observer(camera), zamo_null_direction(camera, 1.0, 0.0)]
    )
    state = (c_double * 8)(*y0.tolist())
    trace = KerrTraceResult()
    crossing = KerrCrossing()
    assert (
        kerr_crossing_fn(
            state,
            1.0,
            0.5,
            0.05,
            0,
            80.0,
            1e-3,
            byref(trace),
            byref(crossing),
        )
        == KerrError.OK
    )
    assert trace.status == 2
    assert crossing.hit == 0
    assert math.isnan(crossing.r_cross)
    assert math.isnan(crossing.vt_cross)
    assert math.isnan(crossing.vphi_cross)
    assert (
        kerr_crossing_fn(
            state,
            1.0,
            0.5,
            0.05,
            0,
            80.0,
            1e-3,
            byref(trace),
            None,
        )
        == KerrError.INVALID_ARGUMENT
    )
