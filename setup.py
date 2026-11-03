#!/usr/bin/env python3
"""Setuptools glue for PyBind11 extension `blackhole_ray_tracer._native_phase2`.

On Windows we **skip** compiling the native module unless `BLACKHOLE_BUILD_NATIVE`
is truthy (`1`) so plain `uv sync` works without Visual C++. Set
`BLACKHOLE_SKIP_NATIVE=1` on Unix to force-disable.

Optional CUDA batch library (Phase 4): set ``BLACKHOLE_BUILD_CUDA=1`` when ``nvcc``
is on ``PATH``. The shared library is installed next to the Python package as
``libbh_rt_phase2_cuda.so`` (``.dll`` on Windows) and loaded by
``native_phase2_cuda.py``. Override GPU arch with ``BLACKHOLE_CUDA_ARCH=sm_89``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import Command, Extension, setup
from setuptools.command.build import build as _build
from setuptools.command.build_py import build_py as _build_py

ROOT = Path(__file__).resolve().parent
CUDA_SRC = ROOT / "kernel" / "cuda" / "bh_rt_schwarzschild_phase2_batch.cu"
CUDA_INCLUDE = ROOT / "kernel" / "include"


def src(*parts: str) -> str:
    return "/".join(parts)


def _truthy(env: str) -> bool:
    return os.environ.get(env, "").strip().lower() in ("1", "true", "yes")


def _can_compile_cpp_stdlib() -> bool:
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        return False
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "probe.cpp"
        o = Path(td) / "probe.o"
        p.write_text("#include <cstddef>\nint main() { return 0; }\n", encoding="utf-8")
        proc = subprocess.run(
            [cxx, "-c", str(p), "-o", str(o)],
            capture_output=True,
            text=True,
            check=False,
        )
    return proc.returncode == 0


def should_build_native() -> bool:
    if _truthy("BLACKHOLE_SKIP_NATIVE"):
        return False
    if _truthy("BLACKHOLE_BUILD_NATIVE"):
        return True
    if sys.platform.startswith("win"):
        return False
    return _can_compile_cpp_stdlib()


def _nvcc_available() -> bool:
    return shutil.which("nvcc") is not None


def should_build_cuda() -> bool:
    """CUDA build is opt-in: requires BLACKHOLE_BUILD_CUDA=1 and nvcc on PATH."""
    if _truthy("BLACKHOLE_SKIP_CUDA"):
        return False
    if not _truthy("BLACKHOLE_BUILD_CUDA"):
        return False
    return _nvcc_available()


def cuda_lib_basename() -> str:
    if sys.platform.startswith("win"):
        return "libbh_rt_phase2_cuda.dll"
    if sys.platform == "darwin":
        return "libbh_rt_phase2_cuda.dylib"
    return "libbh_rt_phase2_cuda.so"


class BuildCuda(Command):
    """Compile the CUDA batch kernel with nvcc into the package build tree."""

    description = "Build CUDA batch shared library (requires nvcc)"
    user_options = [
        ("cuda-arch=", None, "CUDA architecture flag, e.g. sm_75"),
    ]

    def initialize_options(self) -> None:
        self.cuda_arch = os.environ.get("BLACKHOLE_CUDA_ARCH", "sm_75")
        self.build_lib: str | None = None
        self.force = False

    def finalize_options(self) -> None:
        self.set_undefined_options("build", ("build_lib", "build_lib"))

    def run(self) -> None:
        if not should_build_cuda():
            return
        if not CUDA_SRC.is_file():
            raise RuntimeError(f"CUDA source not found: {CUDA_SRC}")

        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise RuntimeError("BLACKHOLE_BUILD_CUDA=1 but nvcc not found on PATH")

        if self.build_lib is None:
            raise RuntimeError("build_lib not set; run via `build` command")

        out_dir = Path(self.build_lib) / "blackhole_ray_tracer"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / cuda_lib_basename()

        cmd: list[str] = [
            nvcc,
            "-O2",
            f"-arch={self.cuda_arch}",
            f"-I{CUDA_INCLUDE}",
        ]
        if sys.platform.startswith("win"):
            cmd.extend(["--compiler-options", "/LD"])
        else:
            cmd.extend(["--compiler-options", "-fPIC"])
        cmd.extend(["-shared", "-o", str(out_path), str(CUDA_SRC), "-lm"])

        print(f"Building CUDA batch library: {out_path} (arch={self.cuda_arch})")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                "nvcc failed:\n"
                + (proc.stdout or "")
                + (proc.stderr or "")
            )
        print(f"Built {out_path}")

        # Also place in repo build/ for editable installs and Makefile parity.
        repo_out = ROOT / "build" / cuda_lib_basename()
        repo_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, repo_out)
        print(f"Copied CUDA library to {repo_out}")


class build_py(_build_py):
    """Run CUDA build after Python sources are staged into build_lib."""

    def run(self) -> None:
        super().run()
        if should_build_cuda():
            self.run_command("build_cuda")


class build(_build):
    sub_commands = _build.sub_commands + [("build_cuda", lambda self: should_build_cuda())]


ext_modules = []
cmdclass: dict[str, type] = {
    "build_cuda": BuildCuda,
    "build_py": build_py,
    "build": build,
}

if should_build_native():
    from pybind11 import get_include
    from setuptools.command.build_ext import build_ext

    ext_modules = [
        Extension(
            "blackhole_ray_tracer._native_phase2",
            [
                src("bridge", "module_phase2.cpp"),
                src("kernel", "src", "bh_rt_rk4.c"),
                src("kernel", "src", "bh_rt_schwarzschild_phase2.c"),
                src("kernel", "src", "bh_rt_schwarzschild_phase2_batch.c"),
            ],
            include_dirs=[
                src("kernel", "include"),
                get_include(),
            ],
            libraries=(
                ["c++", "c++abi"] if sys.platform == "darwin" else ["stdc++"]
            ),
            language="c++",
        ),
    ]
    cmdclass["build_ext"] = build_ext


setup(ext_modules=ext_modules, cmdclass=cmdclass)
