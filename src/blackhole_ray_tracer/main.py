"""CLI entrypoint for bootstrap and Phase 1 experiments."""

import argparse
from rich.console import Console

import numpy as np

from .phase1 import (
    batch_schwarzschild_rays,
    format_step_b_log,
    format_step_d_table,
    summarize_phase1_a_b,
    trace_single_schwarzschild_ray,
)
from .phase1_image import render_einstein_ring_image, write_ppm_rgb
from .phase1_tuning import PRESETS, format_step_f_report
from .phase2_render import render_schwarzschild_3d_image
from .phase2_report import format_phase2_report, render_config_from_preset
from .phase2_types import Phase2RenderConfig


def main() -> None:
    """Run command-line tasks for the project."""
    parser = argparse.ArgumentParser(prog="blackhole-ray-tracer")
    parser.add_argument(
        "--phase1-ab",
        action="store_true",
        help="Run Step A (RK4 sanity) and Step B (single Schwarzschild ray).",
    )
    parser.add_argument(
        "--phase1-step-b",
        action="store_true",
        help="Step B only: log first 50 (r, phi) samples for one Schwarzschild ray.",
    )
    parser.add_argument(
        "--phase1-step-d",
        action="store_true",
        help="Step D: batch rays over impact parameter (see also phase1_driver --step d).",
    )
    parser.add_argument(
        "--phase1-step-e",
        action="store_true",
        help="Step E: write einstein_ring.ppm (simple shadow + sky; see phase1_driver --step e).",
    )
    parser.add_argument(
        "--phase1-step-f",
        action="store_true",
        help="Step F: print tuning presets and single-ray benchmark (dphi vs speed / r_min).",
    )
    parser.add_argument(
        "--phase2-render",
        action="store_true",
        help="Phase 2: 3D Schwarzschild pinhole PPM (shadow + sky; slow at high res).",
    )
    parser.add_argument(
        "--phase2-report",
        action="store_true",
        help="Phase 2: print presets and single-ray dlambda benchmark.",
    )
    parser.add_argument(
        "--ppm-out",
        type=str,
        default="einstein_ring.ppm",
        help="Output path for --phase1-step-e (PPM RGB)",
    )
    parser.add_argument("--img-width", type=int, default=72, help="Step E image width")
    parser.add_argument("--img-height", type=int, default=72, help="Step E image height")
    parser.add_argument(
        "--preset",
        choices=("fast", "balanced", "quality"),
        default=None,
        help="Step E: use fast/balanced/quality preset (overrides img size and integration settings)",
    )
    parser.add_argument(
        "--phase2-preset",
        choices=("fast", "balanced", "quality"),
        default=None,
        help="Phase 2: use fast/balanced/quality (overrides phase2 size and integration; use with --phase2-render)",
    )
    parser.add_argument(
        "--phase2-native",
        action="store_true",
        help="With --phase2-render: trace each ray via C extension (_native_phase2); requires build.",
    )
    args = parser.parse_args()

    console = Console()
    if args.phase2_report:
        console.print(format_phase2_report())
        return
    if args.phase2_render:
        if args.phase2_preset is not None:
            cfg = render_config_from_preset(
                args.phase2_preset,
                m=1.0,
                sky_mode="gradient",
                use_native_phase2=args.phase2_native,
            )
        else:
            cfg = Phase2RenderConfig(
                width=args.img_width,
                height=args.img_height,
                m=1.0,
                dlambda=0.06,
                max_steps=8000,
                use_native_phase2=args.phase2_native,
            )
        rgb, stats = render_schwarzschild_3d_image(cfg)
        write_ppm_rgb(args.phase2_out, rgb)
        preset_note = f", preset [bold]{args.phase2_preset}[/bold]" if args.phase2_preset else ""
        console.print(
            f"Phase 2: wrote [bold]{args.phase2_out}[/bold] ({cfg.width}x{cfg.height} PPM){preset_note}; stats {stats}"
        )
        return
    if args.phase1_step_f:
        console.print(format_step_f_report())
        return
    if args.phase1_step_e:
        w, h = args.img_width, args.img_height
        b_min, b_max = 2.5, 10.0
        phi_max, dphi, r_escape = 8.0, 0.012, 80.0
        if args.preset is not None:
            p = PRESETS[args.preset]
            w, h = p["width"], p["height"]
            dphi = p["dphi"]
            phi_max = p["phi_max"]
            r_escape = p["r_escape"]
            b_min, b_max = p["b_min"], p["b_max"]
        rgb, _ = render_einstein_ring_image(
            w,
            h,
            m=1.0,
            b_min=b_min,
            b_max=b_max,
            phi_max=phi_max,
            dphi=dphi,
            r_escape=r_escape,
        )
        write_ppm_rgb(args.ppm_out, rgb)
        preset_note = f", preset [bold]{args.preset}[/bold]" if args.preset else ""
        console.print(
            f"Step E: wrote [bold]{args.ppm_out}[/bold] ({w}x{h} PPM){preset_note}"
        )
        return
    if args.phase1_step_d:
        b_vals = np.linspace(2.5, 10.0, 24)
        rows = batch_schwarzschild_rays(b_vals, m=1.0)
        console.print(format_step_d_table(rows, m=1.0))
        return
    if args.phase1_step_b:
        console.print(format_step_b_log(trace_single_schwarzschild_ray()))
        return
    if args.phase1_ab:
        console.print(summarize_phase1_a_b())
        return
    console.print(
        "blackhole-ray-tracer bootstrap ready. Use --phase1-ab … --phase1-step-f, or Phase 2: --phase2-report / --phase2-render [--phase2-preset]."
    )


if __name__ == "__main__":
    main()
