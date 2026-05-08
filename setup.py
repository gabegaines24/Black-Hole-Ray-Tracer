#!/usr/bin/env python3
"""Setuptools glue for PyBind11 extension `blackhole_ray_tracer._native_phase2`.

On Windows we **skip** compiling the native module unless `BLACKHOLE_BUILD_NATIVE`
is truthy (`1`) so plain `uv sync` works without Visual C++. Set
`BLACKHOLE_SKIP_NATIVE=1` on Unix to force-disable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent


def src(*parts: str) -> str:
    return str(ROOT.joinpath(*parts))


def _truthy(env: str) -> bool:
    return os.environ.get(env, "").strip().lower() in ("1", "true", "yes")


def should_build_native() -> bool:
    if _truthy("BLACKHOLE_SKIP_NATIVE"):
        return False
    if _truthy("BLACKHOLE_BUILD_NATIVE"):
        return True
    if sys.platform.startswith("win"):
        return False
    return True


ext_modules = []
cmdclass = {}
if should_build_native():
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext_modules = [
        Pybind11Extension(
            "blackhole_ray_tracer._native_phase2",
            [
                src("bridge", "module_phase2.cpp"),
                src("kernel", "src", "bh_rt_rk4.c"),
                src("kernel", "src", "bh_rt_schwarzschild_phase2.c"),
            ],
            include_dirs=[
                src("kernel", "include"),
            ],
            cxx_std=17,
        ),
    ]
    cmdclass = {"build_ext": build_ext}


setup(ext_modules=ext_modules, cmdclass=cmdclass)
