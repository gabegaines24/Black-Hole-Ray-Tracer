"""C/Python parity for Kerr disk crossing policy and Doppler intensity."""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
from ctypes import Structure, byref, c_double, c_int32
from pathlib import Path

import pytest

from blackhole_ray_tracer.phase2_disk import (
    DiskConfig,
    disk_config_for_mass_spin,
    disk_hit_from_equatorial_crossing,
)

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernel"
INCLUDE = KERNEL / "include"


class KerrDiskResult(Structure):
    _fields_ = (("hit", c_int32), ("z_factor", c_double))


class KerrError:
    OK = 0
    INVALID_ARGUMENT = 1
    DOMAIN_ERROR = 2


def _shared_suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


@pytest.fixture(scope="module")
def disk_fn(tmp_path_factory: pytest.TempPathFactory):
    if os.environ.get("SKIP_KERNEL_TESTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip("SKIP_KERNEL_TESTS set")
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        pytest.skip("No C toolchain")
    build_dir = tmp_path_factory.mktemp("kerr_disk")
    lib_path = build_dir / ("libbh_rt_kerr_phase3" + _shared_suffix())
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
        pytest.fail(f"Kerr disk compile failed:\n{proc.stdout}\n{proc.stderr}")
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.bh_rt_kerr_disk_observable
    fn.argtypes = [
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        ctypes.POINTER(KerrDiskResult),
    ]
    fn.restype = c_int32
    return fn


def test_kerr_disk_policy_matches_python(disk_fn) -> None:
    disk = disk_config_for_mass_spin(1.0, 0.5)
    cases = (
        math.nan,
        math.inf,
        -math.inf,
        disk.r_inner - 1e-3,
        disk.r_inner,
        0.5 * (disk.r_inner + disk.r_outer),
        disk.r_outer,
        disk.r_outer + 1e-3,
    )
    for r_cross in cases:
        expected = disk_hit_from_equatorial_crossing(
            r_cross, disk, 1.0, vt=1.2, vph=0.04
        )
        out = KerrDiskResult()
        code = disk_fn(
            r_cross,
            disk.r_inner,
            disk.r_outer,
            1.0,
            1.2,
            0.04,
            disk.redshift_scale,
            byref(out),
        )
        assert code == KerrError.OK
        assert out.hit == int(expected.hit)
        assert out.z_factor == pytest.approx(expected.z_factor, abs=1e-12)

    out = KerrDiskResult()
    assert (
        disk_fn(
            disk.r_inner + 1.0,
            disk.r_inner,
            disk.r_outer,
            1.0,
            math.nan,
            math.nan,
            disk.redshift_scale,
            byref(out),
        )
        == KerrError.OK
    )
    assert out.hit == 1
    assert out.z_factor == 1.0

    zero_scale_disk = DiskConfig(
        r_inner=disk.r_inner, r_outer=disk.r_outer, redshift_scale=0.0
    )
    expected = disk_hit_from_equatorial_crossing(
        disk.r_inner + 1.0, zero_scale_disk, 1.0, vt=1.2, vph=0.04
    )
    out = KerrDiskResult()
    assert (
        disk_fn(
            disk.r_inner + 1.0,
            disk.r_inner,
            disk.r_outer,
            1.0,
            1.2,
            0.04,
            0.0,
            byref(out),
        )
        == KerrError.OK
    )
    assert out.hit == int(expected.hit)
    assert out.z_factor == pytest.approx(expected.z_factor, abs=1e-12)

    expected = disk_hit_from_equatorial_crossing(
        disk.r_inner + 1.0, disk, 1.0, vt=1.2, vph=10.0
    )
    out = KerrDiskResult()
    assert (
        disk_fn(
            disk.r_inner + 1.0,
            disk.r_inner,
            disk.r_outer,
            1.0,
            1.2,
            10.0,
            disk.redshift_scale,
            byref(out),
        )
        == KerrError.OK
    )
    assert out.hit == int(expected.hit)
    assert out.z_factor == pytest.approx(expected.z_factor, abs=1e-12)

    horizon_disk = DiskConfig(r_inner=1.0, r_outer=3.0)
    expected = disk_hit_from_equatorial_crossing(
        2.0, horizon_disk, 1.0, vt=1.0, vph=0.0
    )
    out = KerrDiskResult()
    assert (
        disk_fn(2.0, 1.0, 3.0, 1.0, 1.0, 0.0, 1.0, byref(out))
        == KerrError.OK
    )
    assert out.hit == int(expected.hit)
    assert out.z_factor == pytest.approx(expected.z_factor, abs=1e-12)


def test_kerr_disk_policy_rejects_invalid_controls(disk_fn) -> None:
    out = KerrDiskResult()
    assert (
        disk_fn(10.0, 0.0, 20.0, 1.0, 1.0, 0.1, 1.0, byref(out))
        == KerrError.DOMAIN_ERROR
    )
    assert (
        disk_fn(10.0, 6.0, 5.0, 1.0, 1.0, 0.1, 1.0, byref(out))
        == KerrError.DOMAIN_ERROR
    )
    assert (
        disk_fn(10.0, 6.0, 20.0, 1.0, 1.0, 0.1, -1.0, byref(out))
        == KerrError.DOMAIN_ERROR
    )
    assert (
        disk_fn(10.0, 6.0, 20.0, 1.0, 1.0, 0.1, 1.0, None)
        == KerrError.INVALID_ARGUMENT
    )
