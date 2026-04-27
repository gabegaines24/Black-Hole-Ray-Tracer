"""Phase 2 CLI driver: 3D Schwarzschild render and benchmark report.

Run with:
    PYTHONPATH=src python -m blackhole_ray_tracer.phase2_driver --render --out phase2_shadow.ppm
    PYTHONPATH=src python -m blackhole_ray_tracer.phase2_driver --report
    PYTHONPATH=src python -m blackhole_ray_tracer.phase2_driver --render --preset fast --out out.ppm
"""

from __future__ import annotations

import argparse

from .phase1_image import write_ppm_rgb
from .phase2_render import render_schwarzschild_3d_image
from .phase2_report import format_phase2_report, render_config_from_preset
from .phase2_types import Phase2RenderConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog="phase2-driver")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render a PPM using the 3D Schwarzschild pinhole tracer",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print presets and single-ray timing benchmark",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="phase2_schwarzschild_3d.ppm",
        help="Output PPM path for --render",
    )
    parser.add_argument(
        "--preset",
        choices=("fast", "balanced", "quality"),
        default=None,
        help="Use a quality/speed preset (overrides width/height/dlambda/max_steps/r_escape/fov/r_observer)",
    )
    parser.add_argument("--width", type=int, default=48, help="Image width")
    parser.add_argument("--height", type=int, default=48, help="Image height")
    parser.add_argument("--m", type=float, default=1.0, help="Black hole mass M (geometric units)")
    parser.add_argument("--r-observer", type=float, default=30.0, help="Observer Schwarzschild r")
    parser.add_argument("--fov-deg", type=float, default=60.0, help="Vertical field of view (degrees)")
    parser.add_argument("--dlambda", type=float, default=0.06, help="Affine parameter step for RK4")
    parser.add_argument("--max-steps", type=int, default=8000, help="Max RK4 steps per ray")
    parser.add_argument("--r-escape", type=float, default=80.0, help="Escape radius")
    parser.add_argument(
        "--sky",
        choices=("gradient", "flat"),
        default="gradient",
        help="Background for escaped rays",
    )
    args = parser.parse_args()

    if args.report:
        print(format_phase2_report())
        return

    if not args.render:
        parser.print_help()
        return

    if args.preset is not None:
        cfg = render_config_from_preset(args.preset, m=args.m, sky_mode=args.sky)
    else:
        cfg = Phase2RenderConfig(
            width=args.width,
            height=args.height,
            m=args.m,
            r_observer=args.r_observer,
            fov_deg=args.fov_deg,
            dlambda=args.dlambda,
            max_steps=args.max_steps,
            r_escape=args.r_escape,
            sky_mode=args.sky,
        )
    rgb, stats = render_schwarzschild_3d_image(cfg)
    write_ppm_rgb(args.out, rgb)
    print("Phase 2 (3D Schwarzschild pinhole)")
    print(f"- wrote PPM: {args.out} ({cfg.width}x{cfg.height})")
    if args.preset:
        print(f"- preset: {args.preset}")
    print(
        f"- per-ray: dlambda={cfg.dlambda}, max_steps={cfg.max_steps}, r_escape={cfg.r_escape}, "
        f"r_obs={cfg.r_observer}, fov={cfg.fov_deg}°"
    )
    print(f"- stats: {stats}")


if __name__ == "__main__":
    main()
