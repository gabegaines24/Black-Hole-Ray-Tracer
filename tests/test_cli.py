"""CLI surface smoke tests."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from blackhole_ray_tracer.main import main


def test_main_help_has_single_phase2_out_and_disk_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["blackhole-ray-tracer", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    phase2_out_rows = [
        line for line in out.splitlines() if line.strip().startswith("--phase2-out ")
    ]
    assert len(phase2_out_rows) == 1
    assert "--phase2-disk" in out
    assert "--phase2-ml-gate" in out


def test_cli_phase2_render_tiny_exits_zero() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "cli_smoke.ppm"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "blackhole_ray_tracer.main",
                "--phase2-render",
                "--img-width",
                "4",
                "--img-height",
                "4",
                "--phase2-out",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert out.exists()
        assert out.stat().st_size > 0
