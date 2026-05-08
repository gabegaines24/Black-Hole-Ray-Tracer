#!/usr/bin/env python3
"""Setuptools glue for PyBind11 extension `blackhole_ray_tracer._native_phase2`."""

from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ROOT = Path(__file__).resolve().parent


def src(*parts: str) -> str:
  return str(ROOT.joinpath(*parts))


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

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
