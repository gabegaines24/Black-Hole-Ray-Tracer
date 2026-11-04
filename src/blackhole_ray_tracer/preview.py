"""Interactive preview loop: renders the black hole scene in a matplotlib window.

Uses the best available batch backend by default (CUDA → native → Python).
Designed for a quick visual sanity check during development — not a finished GUI.

Usage
-----
    uv run python -m blackhole_ray_tracer.preview
    uv run python -m blackhole_ray_tracer.preview --preset balanced --fps 4
    uv run python -m blackhole_ray_tracer.preview --width 48 --height 48 --disk
    uv run streamlit run src/blackhole_ray_tracer/streamlit_app.py
"""

from __future__ import annotations

import argparse
import time

from .phase2_render import render_schwarzschild_3d_image
from .preview_config import (
    add_preview_cli_args,
    build_preview_config,
    describe_requested_backend,
    preview_args_from_namespace,
)


def run_preview(
    cfg,
    target_fps: float = 2.0,
    max_frames: int | None = None,
) -> None:
    """Run the interactive matplotlib preview loop.

    Press Ctrl-C or close the window to exit.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for the preview. "
            "Install it with: uv sync --group viz"
        ) from exc

    backend_label = describe_requested_backend(cfg)
    print(f"[preview] {cfg.width}x{cfg.height}  backend={backend_label}  target_fps={target_fps}")

    fig, ax = plt.subplots(figsize=(5, 5))
    title = f"Phase 2 Schwarzschild preview — {cfg.width}x{cfg.height}"
    if cfg.disk is not None:
        title += "  [disk]"
    ax.set_title(title)
    ax.axis("off")

    rgb, stats = render_schwarzschild_3d_image(cfg)
    im = ax.imshow(rgb, origin="upper", interpolation="nearest", vmin=0.0, vmax=1.0)
    info_text = ax.text(
        0.01, 0.99, "", transform=ax.transAxes,
        fontsize=7, color="white", verticalalignment="top",
        bbox=dict(facecolor="black", alpha=0.5, pad=2),
    )

    frame_count = [0]
    start_time = [time.perf_counter()]

    def update(_frame: int) -> tuple:
        t0 = time.perf_counter()
        rgb_new, st = render_schwarzschild_3d_image(cfg)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_s = time.perf_counter() - start_time[0]
        frame_count[0] += 1
        fps = frame_count[0] / total_s if total_s > 0 else 0.0
        im.set_data(rgb_new)
        info_text.set_text(
            f"frame {frame_count[0]}  {elapsed_ms:.0f} ms/frame  {fps:.2f} fps\n"
            f"captured={st['captured']}  escaped={st['escaped']}  other={st['other']}\n"
            f"backend={st['backend']}"
        )
        if max_frames is not None and frame_count[0] >= max_frames:
            ani.event_source.stop()
        return (im, info_text)

    interval_ms = max(1, int(1000.0 / target_fps))
    ani = animation.FuncAnimation(
        fig, update, interval=interval_ms, blit=False, cache_frame_data=False
    )

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Schwarzschild 3D preview (matplotlib FuncAnimation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_preview_cli_args(parser)
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Target animation frames per second.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many frames (useful for benchmarks).")
    args = parser.parse_args(argv)

    build_args = preview_args_from_namespace(args)
    cfg, warnings = build_preview_config(build_args)
    for msg in warnings:
        print(f"[preview] {msg}")

    run_preview(cfg, target_fps=args.fps, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
