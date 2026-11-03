"""C/Python parity for Kerr metric and first-order geodesic algebra."""

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

from blackhole_ray_tracer.phase3_christoffel import kerr_metric_matrix
from blackhole_ray_tracer.phase3_geodesic import null_geodesic_kerr_first_order

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernel"
INCLUDE = KERNEL / "include"


class KerrProbeError:
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
def kerr_probe(tmp_path_factory: pytest.TempPathFactory):
    if os.environ.get("SKIP_KERNEL_TESTS", "").strip().lower() in ("1", "true", "yes"):
        pytest.skip("SKIP_KERNEL_TESTS set")
    compiler = _compiler()
    if compiler is None:
        pytest.skip("No C toolchain")
    build_dir = tmp_path_factory.mktemp("kerr_algebra")
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
        pytest.fail(f"Kerr algebra compile failed:\n{proc.stdout}\n{proc.stderr}")

    lib = ctypes.CDLL(str(lib_path))
    metric = lib.bh_rt_kerr_metric
    metric.argtypes = [c_double, c_double, c_double, c_double, ctypes.POINTER(c_double)]
    metric.restype = c_int32
    rhs = lib.bh_rt_kerr_geodesic_rhs
    rhs.argtypes = [ctypes.POINTER(c_double), c_double, c_double, ctypes.POINTER(c_double)]
    rhs.restype = c_int32
    return metric, rhs


def test_kerr_metric_matches_python(kerr_probe) -> None:
    metric, _ = kerr_probe
    for spin in (0.0, 0.5, -0.5, 0.999999):
        r = 12.0
        theta = 1.1
        m = 1.0
        out = (c_double * 16)()
        code = metric(r, theta, m, spin, out)
        assert code == KerrProbeError.OK
        actual = np.ctypeslib.as_array(out).reshape(4, 4).copy()
        expected = kerr_metric_matrix(r, theta, m, spin)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_kerr_rhs_matches_python(kerr_probe) -> None:
    _, rhs = kerr_probe
    for spin in (0.0, 0.5, -0.5, 0.999999):
        y = np.array(
            [0.2, 12.0, 1.1, -0.3, 1.0, -0.15, 0.04, 0.02],
            dtype=np.float64,
        )
        expected = null_geodesic_kerr_first_order(0.0, y, 1.0, spin)
        y_arg = (c_double * 8)(*y.tolist())
        out = (c_double * 8)()
        code = rhs(y_arg, 1.0, spin, out)
        assert code == KerrProbeError.OK
        actual = np.ctypeslib.as_array(out).copy()
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


def test_kerr_probe_rejects_invalid_domain(kerr_probe) -> None:
    metric, rhs = kerr_probe
    out_metric = (c_double * 16)()
    assert metric(1.0, 1.1, 1.0, 0.5, out_metric) == KerrProbeError.DOMAIN_ERROR
    y = np.array([0.0, 12.0, 1.1, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    y[1] = math.nan
    y_arg = (c_double * 8)(*y.tolist())
    out_rhs = (c_double * 8)()
    assert rhs(y_arg, 1.0, 0.5, out_rhs) == KerrProbeError.INVALID_ARGUMENT
