"""Interactive preview loop: renders the black hole scene in a matplotlib window.

Uses the native batch kernel when the extension is available, falling back to
the Python integrator automatically.  Designed for a quick visual sanity check
during development — not a finished GUI.

Usage
-----
    uv run python -m blackhole_ray_tracer.preview
    uv run python -m blackhole_ray_tracer.preview --preset balanced --fps 4
    uv run python -m blackhole_ray_tracer.preview --width 48 --height 48
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .native_phase2 import batch_native_available
from .phase2_render import render_schwarzschild_3d_image
from .phase2_report import render_config_from_preset, PRESETS
from .phase2_types import Phase2RenderConfig

_DEFAULT_WIDTH = 48
_DEFAULT_HEIGHT = 48
_DEFAULT_PRESET = "fast"


def _build_config(
    preset: str | None,
    width: int,
    height: int,
    use_native: bool,
) -> Phase2RenderConfig:
    if preset is not None:
        return render_config_from_preset(preset, use_native_phase2=use_native)
    return Phase2RenderConfig(
        width=width,
        height=height,
        m=1.0,
        dlambda=0.1,
        max_steps=4000,
        use_native_phase2=use_native,
    )


def run_preview(
    cfg: Phase2RenderConfig,
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
            "Install it with: uv add matplotlib"
        ) from exc

    backend_label = "native_batch" if (cfg.use_native_phase2 and batch_native_available()) else "python"
    print(f"[preview] {cfg.width}x{cfg.height}  backend={backend_label}  target_fps={target_fps}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(f"Phase 2 Schwarzschild preview — {cfg.width}x{cfg.height}")
    ax.axis("off")

    # First frame so we can create the AxesImage
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
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                        help="Use a named render preset (overrides --width/--height).")
    parser.add_argument("--width",  type=int, default=_DEFAULT_WIDTH,
                        help="Image width in pixels.")
    parser.add_argument("--height", type=int, default=_DEFAULT_HEIGHT,
                        help="Image height in pixels.")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Target animation frames per second.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many frames (useful for benchmarks).")
    parser.add_argument("--no-native", action="store_true",
                        help="Force Python integrator even if the native extension is present.")
    args = parser.parse_args(argv)

    use_native = not args.no_native
    if use_native and not batch_native_available():
        print(
            "[preview] Native extension not available — using Python integrator. "
            "Build with BLACKHOLE_BUILD_NATIVE=1 for faster renders."
        )
        use_native = False

    cfg = _build_config(args.preset, args.width, args.height, use_native)
    run_preview(cfg, target_fps=args.fps, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
