"""Optional parity: CUDA batch vs CPU batch (same physics, docs/STATE_API.md layout)."""

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

from blackhole_ray_tracer.phase2_camera import (
    initial_position_observer,
    make_camera_from_config,
    static_observer_null_direction,
)

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kernel"
INCLUDE = KERNEL / "include"
CUDA_DIR = KERNEL / "cuda"
BUILD_DIR = ROOT / "build"


def _compiler() -> str | None:
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def _nvcc() -> str | None:
    return shutil.which("nvcc")


def _shared_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt" or sys.platform.startswith("win"):
        return ".dll"
    return ".so"


def _cuda_device_available() -> bool:
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            libcudart = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        return False
    count = ctypes.c_int(0)
    err = libcudart.cudaGetDeviceCount(ctypes.byref(count))
    return err == 0 and count.value > 0


def _build_cpu_batch_shlib(dest: Path) -> bool:
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


def _build_cuda_shlib(dest: Path) -> bool:
    nvcc = _nvcc()
    if nvcc is None:
        return False
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        nvcc,
        "-O2",
        "-arch=sm_75",
        f"-I{INCLUDE}",
        "--compiler-options",
        "-fPIC",
        "-shared",
        "-o",
        str(dest),
        str(CUDA_DIR / "bh_rt_schwarzschild_phase2_batch.cu"),
        "-lm",
    ]
    proc = subprocess.run(cmd, cwd=str(CUDA_DIR), capture_output=True, text=True)
    return proc.returncode == 0


def _build_grid_y0(width: int, height: int, m: float = 1.0, r_obs: float = 30.0,
                   fov_deg: float = 60.0) -> np.ndarray:
    cam = make_camera_from_config(
        m, r=r_obs, theta=np.pi / 2, phi=0.0,
        fov_deg=fov_deg, width=width, height=height,
    )
    x0 = initial_position_observer(cam)
    rays = []
    for j in range(height):
        for i in range(width):
            sx = 2.0 * (i + 0.5) / width - 1.0
            sy = 1.0 - 2.0 * (j + 0.5) / height
            v0 = static_observer_null_direction(cam, sx, sy)
            rays.append(np.concatenate([x0, v0]))
    return np.stack(rays, axis=0).astype(np.float64)


def _call_cpu_batch(fn, y0: np.ndarray, m: float, dlambda: float, max_steps: int,
                    r_escape: float, r_horizon_epsilon: float):
    n = len(y0)
    flat = np.ascontiguousarray(y0, dtype=np.float64).ravel()
    c_y0 = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_status = np.empty(n, dtype=np.int32)
    c_steps = np.empty(n, dtype=np.int32)
    c_term_r = np.empty(n, dtype=np.float64)
    c_r_min = np.empty(n, dtype=np.float64)
    c_eq_r_cross = np.empty(n, dtype=np.float64)
    fn(
        c_y0, n, m, dlambda, max_steps, r_escape, r_horizon_epsilon,
        c_status.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        c_steps.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        c_term_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c_r_min.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c_eq_r_cross.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return c_status, c_steps, c_term_r, c_r_min, c_eq_r_cross


def _call_cuda_batch(fn, y0: np.ndarray, m: float, dlambda: float, max_steps: int,
                     r_escape: float, r_horizon_epsilon: float):
    n = len(y0)
    flat = np.ascontiguousarray(y0, dtype=np.float64).ravel()
    c_y0 = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_status = np.empty(n, dtype=np.int32)
    c_steps = np.empty(n, dtype=np.int32)
    c_term_r = np.empty(n, dtype=np.float64)
    c_r_min = np.empty(n, dtype=np.float64)
    c_eq_r_cross = np.empty(n, dtype=np.float64)
    fn(
        c_y0, ctypes.c_int(n),
        ctypes.c_double(m), ctypes.c_double(dlambda),
        ctypes.c_int(max_steps), ctypes.c_double(r_escape),
        ctypes.c_double(r_horizon_epsilon),
        c_status.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        c_steps.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        c_term_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c_r_min.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c_eq_r_cross.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return c_status, c_steps, c_term_r, c_r_min, c_eq_r_cross


def _close(a: float, b: float, atol: float) -> None:
    if math.isnan(a) and math.isnan(b):
        return
    assert math.isclose(a, b, rel_tol=0.0, abs_tol=atol), f"{a} != {b} (atol={atol})"


@pytest.fixture(scope="module")
def batch_backends(tmp_path_factory: pytest.TempPathFactory):
    if os.environ.get("SKIP_KERNEL_TESTS", "").strip().lower() in ("1", "true", "yes"):
        pytest.skip("SKIP_KERNEL_TESTS set")
    if _compiler() is None:
        pytest.skip("No C toolchain for CPU batch")
    if _nvcc() is None:
        pytest.skip("nvcc not found")
    if not _cuda_device_available():
        pytest.skip("No CUDA GPU device available")

    build_dir = tmp_path_factory.mktemp("cuda_batch_parity")
    cpu_lib = build_dir / ("libbh_rt_phase2_batch" + _shared_suffix())
    cuda_lib = build_dir / ("libbh_rt_phase2_cuda" + _shared_suffix())

    if not _build_cpu_batch_shlib(cpu_lib):
        pytest.skip("CPU batch compile failed")
    if not _build_cuda_shlib(cuda_lib):
        pytest.skip("CUDA batch compile failed")

    cpu = ctypes.CDLL(str(cpu_lib))
    cpu_fn = cpu.bh_rt_schwarzschild_phase2_batch_trace
    cpu_fn.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    cpu_fn.restype = None

    cuda = ctypes.CDLL(str(cuda_lib))
    cuda_fn = cuda.bh_rt_schwarzschild_phase2_batch_trace_cuda
    cuda_fn.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    cuda_fn.restype = None

    return cpu_fn, cuda_fn


@pytest.mark.parametrize("width,height,dlambda,max_steps", [
    (4, 4, 0.1, 3000),
    (2, 3, 0.12, 500),
])
def test_cuda_batch_parity_vs_cpu(batch_backends, width, height, dlambda, max_steps):
    cpu_fn, cuda_fn = batch_backends
    m, r_escape, r_hep = 1.0, 80.0, 1e-3
    y0 = _build_grid_y0(width, height, m=m)

    cpu_out = _call_cpu_batch(cpu_fn, y0, m, dlambda, max_steps, r_escape, r_hep)
    cuda_out = _call_cuda_batch(cuda_fn, y0, m, dlambda, max_steps, r_escape, r_hep)

    c_status, c_steps, c_term_r, c_r_min, c_eq = cpu_out
    g_status, g_steps, g_term_r, g_r_min, g_eq = cuda_out

    for i in range(len(y0)):
        assert int(c_status[i]) == int(g_status[i]), f"ray {i}: status"
        assert int(c_steps[i]) == int(g_steps[i]), f"ray {i}: steps"
        _close(float(c_term_r[i]), float(g_term_r[i]), atol=1e-8)
        _close(float(c_r_min[i]), float(g_r_min[i]), atol=1e-8)
        _close(float(c_eq[i]), float(g_eq[i]), atol=1e-8)
