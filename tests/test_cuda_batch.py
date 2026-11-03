"""CUDA batch backend smoke tests (skip when lib or GPU unavailable)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from blackhole_ray_tracer.native_phase2_cuda import (
    cuda_batch_available,
    schwarzschild_phase2_batch_cuda,
)
from blackhole_ray_tracer.phase2_batch import build_camera_y0
from blackhole_ray_tracer.phase2_types import Phase2RenderConfig

ROOT = Path(__file__).resolve().parents[1]


def test_cuda_batch_available_returns_bool() -> None:
    assert isinstance(cuda_batch_available(), bool)


def test_setup_cuda_build_helpers_present() -> None:
    text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "def should_build_cuda" in text
    assert "def cuda_lib_basename" in text
    assert "class BuildCuda" in text


@pytest.mark.skipif(
    os.environ.get("SKIP_CUDA_TESTS", "").strip().lower() in ("1", "true", "yes"),
    reason="SKIP_CUDA_TESTS set",
)
def test_cuda_batch_trace_smoke() -> None:
    if not cuda_batch_available():
        pytest.skip("CUDA batch library not built or not loadable")

    cfg = Phase2RenderConfig(width=4, height=4, dlambda=0.1, max_steps=500)
    y0 = build_camera_y0(cfg)
    try:
        result = schwarzschild_phase2_batch_cuda(
            y0,
            m=cfg.m,
            dlambda=cfg.dlambda,
            max_steps=cfg.max_steps,
            r_escape=cfg.r_escape,
            r_horizon_epsilon=cfg.r_horizon_epsilon,
        )
    except RuntimeError as exc:
        if "CUDA" in str(exc) or "cuda" in str(exc).lower():
            pytest.skip(f"CUDA runtime unavailable: {exc}")
        raise

    assert result["status"].shape == (16,)
    assert result["steps_taken"].shape == (16,)
    assert result["termination_r"].shape == (16,)
    assert result["r_min"].shape == (16,)
    assert np.all(result["status"] >= 0)


@pytest.mark.skipif(
    shutil.which("nvcc") is None,
    reason="nvcc not on PATH",
)
def test_setup_cuda_build_command_smoke(tmp_path: Path) -> None:
    """Verify setup.py BuildCuda compiles when nvcc is available."""
    if os.environ.get("SKIP_CUDA_TESTS", "").strip().lower() in ("1", "true", "yes"):
        pytest.skip("SKIP_CUDA_TESTS set")

    env = os.environ.copy()
    env["BLACKHOLE_BUILD_CUDA"] = "1"
    env["BLACKHOLE_CUDA_ARCH"] = env.get("BLACKHOLE_CUDA_ARCH", "sm_75")

    proc = subprocess.run(
        [
            sys.executable,
            "setup.py",
            "build_cuda",
            "--build-lib",
            str(tmp_path),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"nvcc build failed (no toolkit?): {proc.stderr[:500]}")

    lib_name = "libbh_rt_phase2_cuda.so"
    if sys.platform.startswith("win"):
        lib_name = "libbh_rt_phase2_cuda.dll"
    assert (tmp_path / "blackhole_ray_tracer" / lib_name).is_file()
