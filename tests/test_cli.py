"""CLI surface smoke tests."""

from __future__ import annotations

import sys

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
