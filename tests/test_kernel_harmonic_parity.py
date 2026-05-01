"""Optional parity test: kernel C harmonic RK4 demo vs Phase A Python reference."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from blackhole_ray_tracer.phase1 import run_rk4_sanity

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernel"
EXE_NAME = os.name == "nt" and "harmonic_demo.exe" or "harmonic_demo"
EXE = KERNEL / EXE_NAME


def _compiler() -> str | None:
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def _build_harmonic_demo() -> bool:
    cc = _compiler()
    if cc is None:
        return False
    include = KERNEL / "include"
    cmd = [
        cc,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-O2",
        f"-I{include}",
        str(KERNEL / "src" / "bh_rt_rk4.c"),
        str(KERNEL / "src" / "demo_harmonic.c"),
        "-o",
        str(EXE),
        "-lm",
    ]
    r = subprocess.run(cmd, cwd=str(KERNEL), capture_output=True, text=True)
    return r.returncode == 0


@pytest.fixture(scope="module")
def harmonic_demo_path() -> Path:
    if os.environ.get("SKIP_KERNEL_TESTS", "").strip().lower() in ("1", "true", "yes"):
        pytest.skip("SKIP_KERNEL_TESTS set")
    if not _build_harmonic_demo():
        pytest.skip("No C toolchain or kernel compile failed")
    assert EXE.is_file()
    return EXE


def test_harmonic_rk4_parity_python_phase_a(harmonic_demo_path: Path) -> None:
    dt = 0.02
    t_total = 8.0
    omega = 1.0
    py_max, py_mean = run_rk4_sanity(dt=dt, total_time=t_total, omega=omega)

    proc = subprocess.run(
        [str(harmonic_demo_path), str(dt), str(t_total), str(omega)],
        cwd=str(KERNEL),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    line = proc.stdout.strip().splitlines()[-1]
    m = re.search(
        r"max_abs_err_pos=([\d.eE+-]+)\s+mean_abs_err_pos=([\d.eE+-]+)",
        line,
    )
    assert m is not None, line
    c_max = float(m.group(1))
    c_mean = float(m.group(2))

    tol = 1e-13
    assert abs(c_max - py_max) < tol
    assert abs(c_mean - py_mean) < tol
