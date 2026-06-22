"""Smoke tests for the Phase 3 Kerr CLI driver."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from blackhole_ray_tracer.phase3_driver import main


class TestPhase3DriverCLI:
    def test_help_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["kerr-render", "--help"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0

    def test_no_args_prints_help(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without --render the driver prints help and returns (no SystemExit)."""
        monkeypatch.setattr(sys, "argv", ["kerr-render"])
        main()  # should not raise
        # help text is printed to stdout via argparse
        out = capsys.readouterr().out
        assert "--spin" in out or "--render" in out

    def test_render_writes_ppm(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--render with tiny image should create a PPM file."""
        out_ppm = str(tmp_path / "kerr_test.ppm")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "kerr-render",
                "--render",
                "--width", "6",
                "--height", "4",
                "--spin", "0.5",
                "--max-steps", "100",
                "--dlambda", "0.3",
                "--out", out_ppm,
            ],
        )
        main()
        assert Path(out_ppm).exists(), "PPM output not created"
        assert Path(out_ppm).stat().st_size > 0, "PPM output is empty"

    def test_render_schwarzschild_a0(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """a=0 (Schwarzschild) must also complete without error."""
        out_ppm = str(tmp_path / "schwarzschild.ppm")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "kerr-render",
                "--render",
                "--width", "6",
                "--height", "4",
                "--spin", "0.0",
                "--max-steps", "100",
                "--dlambda", "0.3",
                "--out", out_ppm,
            ],
        )
        main()
        assert Path(out_ppm).exists()

    def test_ppm_starts_with_magic(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Output file should begin with PPM magic bytes 'P6'."""
        out_ppm = str(tmp_path / "magic_check.ppm")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "kerr-render",
                "--render",
                "--width", "4",
                "--height", "4",
                "--spin", "0.3",
                "--max-steps", "80",
                "--dlambda", "0.4",
                "--out", out_ppm,
            ],
        )
        main()
        header = Path(out_ppm).read_bytes()[:2]
        assert header == b"P6"
