"""Tests for preview / dashboard configuration helpers."""

from __future__ import annotations

import pytest

from blackhole_ray_tracer.preview_config import (
    PreviewBuildArgs,
    backend_from_phase2_flags,
    build_preview_config,
    resolve_preview_backend_flags,
)
from blackhole_ray_tracer.phase2_types import DiskConfig


def test_backend_from_phase2_flags_priority() -> None:
    assert backend_from_phase2_flags(cuda=True, native=True) == "cuda"
    assert backend_from_phase2_flags(native=True, ml=True) == "native"
    assert backend_from_phase2_flags(ml=True) == "ml"
    assert backend_from_phase2_flags() == "auto"


def test_resolve_preview_backend_python_native_cuda_ml() -> None:
    assert resolve_preview_backend_flags("python") == (False, False, False)
    assert resolve_preview_backend_flags("native") == (False, True, False)
    assert resolve_preview_backend_flags("cuda") == (True, False, False)
    assert resolve_preview_backend_flags("ml") == (False, False, True)


def test_build_preview_config_custom_size_no_disk() -> None:
    cfg, warnings = build_preview_config(PreviewBuildArgs(width=32, height=24, backend="python"))
    assert cfg.width == 32
    assert cfg.height == 24
    assert cfg.disk is None
    assert not cfg.use_native_phase2
    assert not cfg.use_cuda_batch
    assert warnings == []


def test_build_preview_config_preset_with_disk() -> None:
    cfg, _ = build_preview_config(
        PreviewBuildArgs(preset="fast", disk=True, backend="python", sky_mode="flat")
    )
    assert cfg.width == 24
    assert cfg.height == 24
    assert cfg.sky_mode == "flat"
    assert isinstance(cfg.disk, DiskConfig)


def test_build_preview_config_camera_overrides() -> None:
    cfg, _ = build_preview_config(
        PreviewBuildArgs(
            width=16,
            height=16,
            backend="python",
            r_observer=40.0,
            fov_deg=45.0,
            observer_phi=1.0,
        )
    )
    assert cfg.r_observer == 40.0
    assert cfg.fov_deg == 45.0
    assert cfg.observer_phi == 1.0


def test_build_preview_config_native_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "blackhole_ray_tracer.preview_config.batch_native_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "blackhole_ray_tracer.preview_config.cuda_batch_available",
        lambda: False,
    )
    cfg, warnings = build_preview_config(PreviewBuildArgs(backend="native", width=8, height=8))
    assert not cfg.use_native_phase2
    assert any("Native batch" in w for w in warnings)


def test_build_preview_config_auto_prefers_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "blackhole_ray_tracer.preview_config.cuda_batch_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "blackhole_ray_tracer.preview_config.batch_native_available",
        lambda: True,
    )
    cfg, warnings = build_preview_config(PreviewBuildArgs(backend="auto", width=8, height=8))
    assert cfg.use_native_phase2
    assert warnings == []


def test_preview_cli_help_includes_backend_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import sys

    from blackhole_ray_tracer.preview import main as preview_main

    monkeypatch.setattr(sys, "argv", ["preview", "--help"])
    with pytest.raises(SystemExit) as exc:
        preview_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--backend" in out
    assert "--disk" in out
