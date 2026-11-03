"""Phase 3 CLI driver: Kerr Boyer-Lindquist render and report."""

from __future__ import annotations

import argparse
from dataclasses import replace

from .phase1_image import write_ppm_rgb
from .phase2_disk import disk_config_for_mass_spin
from .phase3_render import render_kerr_image
from .phase3_report import format_phase3_report, render_config_from_preset
from .phase3_types import KerrRenderConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog="phase3-driver")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render a PPM using the Kerr Boyer-Lindquist pinhole tracer",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print presets and single-ray Kerr timing benchmark",
    )
    parser.add_argument("--out", type=str, default="phase3_kerr.ppm")
    parser.add_argument("--preset", choices=("fast", "balanced", "quality"), default=None)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--spin", type=float, default=0.5, help="Dimensionless Kerr spin a/M")
    parser.add_argument("--r-observer", type=float, default=30.0)
    parser.add_argument("--observer-theta", type=float, default=1.0471975511965976)
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--dlambda", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=8_000)
    parser.add_argument("--r-escape", type=float, default=80.0)
    parser.add_argument(
        "--sky",
        choices=("gradient", "flat"),
        default="gradient",
        help="Background for escaped rays",
    )
    parser.add_argument(
        "--disk",
        action="store_true",
        help="Enable the thin equatorial accretion disk overlay.",
    )
    args = parser.parse_args()

    if args.report:
        print(format_phase3_report(m=args.m, spin=args.spin))
        return

    if not args.render:
        parser.print_help()
        return

    if args.preset is not None:
        cfg = render_config_from_preset(
            args.preset, m=args.m, spin=args.spin, sky_mode=args.sky
        )
        if args.disk:
            cfg = replace(cfg, disk=disk_config_for_mass_spin(args.m, args.spin))
    else:
        cfg = KerrRenderConfig(
            width=args.width,
            height=args.height,
            m=args.m,
            spin=args.spin,
            r_observer=args.r_observer,
            observer_theta=args.observer_theta,
            fov_deg=args.fov_deg,
            dlambda=args.dlambda,
            max_steps=args.max_steps,
            r_escape=args.r_escape,
            sky_mode=args.sky,
            disk=disk_config_for_mass_spin(args.m, args.spin) if args.disk else None,
        )

    rgb, stats = render_kerr_image(cfg)
    write_ppm_rgb(args.out, rgb)
    print("Phase 3 (Kerr Boyer-Lindquist pinhole)")
    print(f"- wrote PPM: {args.out} ({cfg.width}x{cfg.height})")
    if args.preset:
        print(f"- preset: {args.preset}")
    if args.disk:
        print("- disk: enabled")
    print(
        f"- per-ray: spin={cfg.spin}, dlambda={cfg.dlambda}, max_steps={cfg.max_steps}, "
        f"r_escape={cfg.r_escape}, r_obs={cfg.r_observer}, fov={cfg.fov_deg}°"
    )
    print(f"- stats: {stats}")


if __name__ == "__main__":
    main()
