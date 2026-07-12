"""Streamlit dashboard for single-frame Phase 2 Schwarzschild renders.

Run::

    uv sync --group viz
    uv run streamlit run src/blackhole_ray_tracer/streamlit_app.py

Or::

    uv run blackhole-ray-tracer-dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path

from .phase2_render import render_schwarzschild_3d_image
from .phase2_report import PRESETS
from .preview_config import (
    PREVIEW_BACKENDS,
    PreviewBuildArgs,
    build_preview_config,
    describe_requested_backend,
)


def _render_once(cfg):
    rgb, stats = render_schwarzschild_3d_image(cfg)
    return rgb, stats


def run_dashboard() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "streamlit is required for the dashboard. Install with: uv sync --group viz"
        ) from exc

    st.set_page_config(page_title="Black hole ray tracer", layout="wide")
    st.title("Schwarzschild 3D render")
    st.caption(
        "Accuracy-oriented single-frame preview. Pure-Python paths are slow at high resolution."
    )

    with st.sidebar:
        st.header("Scene")
        use_preset = st.checkbox("Use preset", value=True)
        preset = st.selectbox("Preset", list(PRESETS.keys()), index=0, disabled=not use_preset)
        width = st.number_input("Width", min_value=8, max_value=256, value=48, step=8, disabled=use_preset)
        height = st.number_input("Height", min_value=8, max_value=256, value=48, step=8, disabled=use_preset)

        st.header("Camera")
        r_observer = st.slider("Observer r", min_value=6.0, max_value=80.0, value=30.0, step=1.0)
        fov_deg = st.slider("FOV (deg)", min_value=20.0, max_value=120.0, value=60.0, step=5.0)
        observer_phi = st.slider("Observer φ (rad)", min_value=0.0, max_value=6.28, value=0.0, step=0.1)

        st.header("Appearance")
        disk = st.checkbox("Accretion disk", value=False)
        sky_mode = st.selectbox("Sky mode", ("gradient", "flat"))
        supersample = st.selectbox("Anti-alias (supersample)", (1, 2), index=0)

        st.header("Backend")
        backend = st.selectbox("Backend", PREVIEW_BACKENDS, index=0)
        ml_model_path: str | None = None
        if backend == "ml":
            default_model = Path("ml/models/surrogate.npz")
            ml_model_path = st.text_input(
                "ML model path",
                value=str(default_model) if default_model.is_file() else "",
            )
            if not ml_model_path.strip():
                ml_model_path = None

    build_args = PreviewBuildArgs(
        preset=preset if use_preset else None,
        width=int(width),
        height=int(height),
        backend=backend,  # type: ignore[arg-type]
        disk=disk,
        sky_mode=sky_mode,
        r_observer=r_observer,
        fov_deg=fov_deg,
        observer_phi=observer_phi,
        ml_model_path=ml_model_path,
        supersample=int(supersample),
    )
    cfg, warnings = build_preview_config(build_args)
    for msg in warnings:
        st.sidebar.warning(msg)

    st.sidebar.markdown(f"**Effective backend:** `{describe_requested_backend(cfg)}`")

    if st.button("Render frame", type="primary"):
        with st.spinner("Tracing geodesics…"):
            rgb, stats = _render_once(cfg)
        st.image(rgb, caption=f"{cfg.width}×{cfg.height}  backend={stats['backend']}", clamp=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Captured", int(stats["captured"]))
        c2.metric("Escaped", int(stats["escaped"]))
        c3.metric("Other", int(stats["other"]))
        c4.metric("Backend", str(stats["backend"]))


def main() -> None:
    """Console entry: launch Streamlit on this module."""
    import subprocess

    app_path = Path(__file__).resolve()
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    run_dashboard()
