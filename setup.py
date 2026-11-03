#!/usr/bin/env python3
"""Setuptools glue for PyBind11 extension `blackhole_ray_tracer._native_phase2`.

On Windows we **skip** compiling the native module unless `BLACKHOLE_BUILD_NATIVE`
is truthy (`1`) so plain `uv sync` works without Visual C++. Set
`BLACKHOLE_SKIP_NATIVE=1` on Unix to force-disable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import setup


def src(*parts: str) -> str:
    """Return a project-relative POSIX path for setuptools."""
    return Path(*parts).as_posix()


def _truthy(env: str) -> bool:
    return os.environ.get(env, "").strip().lower() in ("1", "true", "yes")


def _cxx_compiler() -> str | None:
    configured = os.environ.get("CXX")
    if configured:
        return configured
    if sys.platform == "darwin":
        xcrun = shutil.which("xcrun")
        if xcrun is not None:
            proc = subprocess.run(
                [xcrun, "--find", "clang++"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
    return shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")


def _darwin_cxx_include_flags() -> list[str]:
    if sys.platform != "darwin":
        return []
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return []
    proc = subprocess.run(
        [xcrun, "--show-sdk-path"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    sdk_dir = Path(proc.stdout.strip())
    include_dir = sdk_dir / "usr/include/c++/v1"
    if not include_dir.is_dir():
        return []
    return [
        "-isysroot",
        sdk_dir.as_posix(),
        "-isystem",
        include_dir.as_posix(),
    ]


def _darwin_cxx_link_flags() -> list[str]:
    return _darwin_cxx_include_flags()[:2]


def _can_compile_cpp_stdlib() -> bool:
    cxx = _cxx_compiler()
    if cxx is None:
        return False
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "probe.cpp"
        o = Path(td) / "probe.o"
        p.write_text("#include <cstddef>\nint main() { return 0; }\n", encoding="utf-8")
        proc = subprocess.run(
            [cxx, *_darwin_cxx_include_flags(), "-c", str(p), "-o", str(o)],
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


ext_modules = []
cmdclass = {}
if should_build_native():
    native_cxx = _cxx_compiler()
    if native_cxx is not None:
        os.environ.setdefault("CXX", native_cxx)
    from pybind11.setup_helpers import Pybind11Extension, build_ext as pybind11_build_ext

    def _remove_cxx_standard_flags(flags: list[str]) -> list[str]:
        return [
            flag
            for flag in flags
            if not (
                (flag.startswith("-std=") and "++" in flag)
                or flag.startswith("/std:c++")
            )
        ]

    class build_ext(pybind11_build_ext):
        """Compile C kernel sources without C++-only standard flags."""

        def build_extensions(self) -> None:
            compiler = self.compiler
            original_compile = getattr(compiler, "_compile", None)
            if original_compile is None:
                super().build_extensions()
                return

            def compile_source(
                obj: str,
                source: str,
                extension: str,
                cc_args: list[str],
                extra_postargs: list[str] | None,
                pp_opts: list[str],
            ) -> object:
                postargs = list(extra_postargs or [])
                source_suffix = Path(source).suffix.lower()
                if source_suffix == ".c":
                    postargs = _remove_cxx_standard_flags(postargs)
                elif source_suffix in {".cc", ".cpp", ".cxx"}:
                    postargs.extend(_darwin_cxx_include_flags())
                return original_compile(
                    obj, source, extension, cc_args, postargs, pp_opts
                )

            compiler._compile = compile_source
            try:
                super().build_extensions()
            finally:
                compiler._compile = original_compile

    ext_modules = [
        Pybind11Extension(
            "blackhole_ray_tracer._native_phase2",
            [
                src("bridge", "module_phase2.cpp"),
                src("kernel", "src", "bh_rt_rk4.c"),
                src("kernel", "src", "bh_rt_schwarzschild_phase2.c"),
                src("kernel", "src", "bh_rt_schwarzschild_phase2_batch.c"),
            ],
            include_dirs=[
                src("kernel", "include"),
            ],
            extra_link_args=_darwin_cxx_link_flags(),
            cxx_std=17,
        ),
    ]
    cmdclass = {"build_ext": build_ext}


setup(ext_modules=ext_modules, cmdclass=cmdclass)
