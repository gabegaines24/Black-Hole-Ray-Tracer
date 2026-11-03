"""C/Python parity for the bounded Kerr ZAMO camera initializer."""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
from ctypes import c_double, c_int32
from pathlib import Path

import numpy as np
import pytest

from blackhole_ray_tracer.phase3_camera import (
    initial_position_observer,
    make_kerr_camera_from_config,
    zamo_null_direction,
)
from blackhole_ray_tracer.phase3_christoffel import kerr_outer_horizon
from blackhole_ray_tracer.phase3_geodesic import metric_invariant_kerr

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernel"
INCLUDE = KERNEL / "include"


class KerrError:
    OK = 0
    INVALID_ARGUMENT = 1
    DOMAIN_ERROR = 2
    NUMERICAL_ERROR = 3


def _compiler() -> str | None:
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def _shared_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt" or sys.platform.startswith("win"):
        return ".dll"
    return ".so"


@pytest.fixture(scope="module")
def kerr_camera_fn(tmp_path_factory: pytest.TempPathFactory):
    if os.environ.get("SKIP_KERNEL_TESTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip("SKIP_KERNEL_TESTS set")
    compiler = _compiler()
    if compiler is None:
        pytest.skip("No C toolchain")
    build_dir = tmp_path_factory.mktemp("kerr_camera")
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
        pytest.fail(f"Kerr camera compile failed:\n{proc.stdout}\n{proc.stderr}")

    lib = ctypes.CDLL(str(lib_path))
    fn = lib.bh_rt_kerr_zamo_initial_state
    fn.argtypes = [
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_int32,
        c_int32,
        c_double,
        c_double,
        ctypes.POINTER(c_double),
    ]
    fn.restype = c_int32
    return fn


def test_kerr_zamo_initial_state_matches_python(
    kerr_camera_fn,
) -> None:
    for spin in (0.0, 0.5, -0.5, 0.999999, -0.999999):
        m = 1.0
        r = 30.0
        theta = np.pi / 3.0
        phi = -0.4
        fov_deg = 60.0
        width = 16
        height = 9
        sx = 0.2
        sy = -0.3
        camera = make_kerr_camera_from_config(
            m,
            spin,
            r=r,
            theta=theta,
            phi=phi,
            fov_deg=fov_deg,
            width=width,
            height=height,
        )
        expected = np.concatenate(
            [
                initial_position_observer(camera),
                zamo_null_direction(camera, sx, sy),
            ]
        )
        out = (c_double * 8)()
        code = kerr_camera_fn(
            m,
            spin,
            r,
            theta,
            phi,
            fov_deg,
            width,
            height,
            sx,
            sy,
            out,
        )
        assert code == KerrError.OK
        actual = np.ctypeslib.as_array(out).copy()
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
        assert abs(metric_invariant_kerr(*actual, m, spin)) < 2e-13


def test_kerr_zamo_initial_state_rejects_invalid_camera_domain(
    kerr_camera_fn,
) -> None:
    out = (c_double * 8)()
    horizon = kerr_outer_horizon(1.0, 0.5)
    cases = (
        (1.0, 0.5, horizon, 1.1, 0.0, 60.0, 8, 8, 0.0, 0.0),
        (1.0, 0.5, 30.0, 1e-12, 0.0, 60.0, 8, 8, 0.0, 0.0),
        (1.0, 0.5, 30.0, 1.1, 0.0, 180.0, 8, 8, 0.0, 0.0),
        (1.0, 0.5, 30.0, 1.1, 0.0, 60.0, 0, 8, 0.0, 0.0),
    )
    for args in cases:
        assert kerr_camera_fn(*args, out) == KerrError.DOMAIN_ERROR

    assert (
        kerr_camera_fn(
            1.0,
            0.5,
            30.0,
            1.1,
            0.0,
            60.0,
            8,
            8,
            math.nan,
            0.0,
            out,
        )
        == KerrError.INVALID_ARGUMENT
    )
