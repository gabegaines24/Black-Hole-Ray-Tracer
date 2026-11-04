"""Native extension availability checks for requested builds."""

from __future__ import annotations

import os

import pytest

from blackhole_ray_tracer.native_phase2 import native_phase2_available
from blackhole_ray_tracer.native_phase3 import native_phase3_available


_NATIVE_REQUIRED = os.environ.get("BLACKHOLE_REQUIRE_NATIVE", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


@pytest.mark.skipif(
    not _NATIVE_REQUIRED,
    reason="BLACKHOLE_REQUIRE_NATIVE is not enabled",
)
def test_requested_native_extensions_load() -> None:
    """Requested native builds must expose both compiled extension modules."""
    assert native_phase2_available(), "_native_phase2 failed to load"
    assert native_phase3_available(), "_native_phase3 failed to load"
