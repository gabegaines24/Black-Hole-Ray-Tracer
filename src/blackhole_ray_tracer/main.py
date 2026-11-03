"""CLI entrypoint for bootstrap and Phase 1 experiments."""

import argparse
from dataclasses import replace

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
from .phase2_types import DiskConfig, Phase2RenderConfig
from .phase2_disk import disk_config_for_mass_spin
from .native_phase2 import batch_native_available


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
        "--phase2-out",
        type=str,
        default="phase2_schwarzschild_3d.ppm",
        help="Output PPM for --phase2-render",
    )
    parser.add_argument(
        "--phase2-native",
        action="store_true",
        help="With --phase2-render: trace each ray via C extension (_native_phase2); requires build.",
    )
    parser.add_argument(
        "--phase2-cuda",
        action="store_true",
        help="With --phase2-render and --phase2-native: prefer CUDA batch when lib is available.",
    )
    parser.add_argument(
        "--phase2-disk",
        action="store_true",
        help="With --phase2-render: enable the thin equatorial accretion disk overlay.",
    )
    parser.add_argument(
        "--phase2-ml-gate",
        type=str,
        default=None,
        help="With --phase2-render: use ML RuntimeGate model path; missing model falls back to RK4.",
    )
    parser.add_argument(
        "--phase2-preview",
        action="store_true",
        help="Phase 2: open a live matplotlib preview window (uses native batch when available).",
    )
    parser.add_argument(
        "--preview-fps", type=float, default=2.0,
        help="Target FPS for --phase2-preview.",
    )
    parser.add_argument(
        "--preview-frames", type=int, default=None,
        help="Stop after N frames in --phase2-preview (for benchmarks/testing).",
    )
    parser.add_argument(
        "--phase3-render",
        action="store_true",
        help="Phase 3: Kerr Boyer-Lindquist pinhole PPM (Python integrator).",
    )
    parser.add_argument(
        "--phase3-report",
        action="store_true",
        help="Phase 3: print Kerr presets and single-ray benchmark.",
    )
    parser.add_argument(
        "--phase3-out",
        type=str,
        default="phase3_kerr.ppm",
        help="Output PPM for --phase3-render",
    )
    parser.add_argument(
        "--phase3-preset",
        choices=("fast", "balanced", "quality"),
        default=None,
        help="Phase 3: use fast/balanced/quality preset with --phase3-render.",
    )
    parser.add_argument(
        "--spin",
        type=float,
        default=0.5,
        help="Kerr spin parameter a/M for Phase 3 commands.",
    )
    parser.add_argument(
        "--phase3-disk",
        action="store_true",
        help="With --phase3-render: enable spin-dependent equatorial disk overlay.",
    )
    args = parser.parse_args()

    console = Console()
    if args.phase3_report:
        from .phase3_report import format_phase3_report

        console.print(format_phase3_report(m=1.0, spin=args.spin))
        return
    if args.phase3_render:
        from .phase3_render import render_kerr_image
        from .phase3_report import render_config_from_preset as phase3_render_config_from_preset
        from .phase3_types import KerrRenderConfig

        if args.phase3_preset is not None:
            phase3_cfg = phase3_render_config_from_preset(
                args.phase3_preset, m=1.0, spin=args.spin, sky_mode="gradient"
            )
            if args.phase3_disk:
                phase3_cfg = replace(
                    phase3_cfg, disk=disk_config_for_mass_spin(1.0, args.spin)
                )
        else:
            phase3_cfg = KerrRenderConfig(
                width=args.img_width,
                height=args.img_height,
                m=1.0,
                spin=args.spin,
                dlambda=0.08,
                max_steps=8000,
                disk=disk_config_for_mass_spin(1.0, args.spin)
                if args.phase3_disk
                else None,
            )
        rgb, stats = render_kerr_image(phase3_cfg)
        write_ppm_rgb(args.phase3_out, rgb)
        preset_note = (
            f", preset [bold]{args.phase3_preset}[/bold]" if args.phase3_preset else ""
        )
        console.print(
            f"Phase 3: wrote [bold]{args.phase3_out}[/bold] "
            f"({phase3_cfg.width}x{phase3_cfg.height} PPM, spin={phase3_cfg.spin})"
            f"{preset_note}; stats {stats}"
        )
        return
    if args.phase2_preview:
        from .preview import run_preview, _build_config  # local import to avoid matplotlib at startup
        use_native = not getattr(args, "no_native", False)
        if use_native and not batch_native_available():
            console.print(
                "[yellow]Native extension not available — using Python integrator.[/yellow]"
            )
            use_native = False
        preview_cfg = _build_config(
            args.phase2_preset,
            args.img_width,
            args.img_height,
            use_native,
            disk=args.phase2_disk,
        )
        run_preview(preview_cfg, target_fps=args.preview_fps, max_frames=args.preview_frames)
        return
    if args.phase2_report:
        console.print(format_phase2_report())
        return
    if args.phase2_render:
        if args.phase2_preset is not None:
            phase2_cfg = render_config_from_preset(
                args.phase2_preset,
                m=1.0,
                sky_mode="gradient",
                use_native_phase2=args.phase2_native,
            )
            if args.phase2_cuda:
                phase2_cfg = replace(phase2_cfg, use_cuda_batch=True)
            if args.phase2_disk:
                phase2_cfg = replace(phase2_cfg, disk=DiskConfig())
            if args.phase2_ml_gate is not None:
                phase2_cfg = replace(
                    phase2_cfg, ml_gate_model_path=args.phase2_ml_gate
                )
        else:
            phase2_cfg = Phase2RenderConfig(
                width=args.img_width,
                height=args.img_height,
                m=1.0,
                dlambda=0.06,
                max_steps=8000,
                use_native_phase2=args.phase2_native,
                use_cuda_batch=args.phase2_cuda,
                disk=DiskConfig() if args.phase2_disk else None,
                ml_gate_model_path=args.phase2_ml_gate,
            )
        rgb, stats = render_schwarzschild_3d_image(phase2_cfg)
        write_ppm_rgb(args.phase2_out, rgb)
        preset_note = (
            f", preset [bold]{args.phase2_preset}[/bold]" if args.phase2_preset else ""
        )
        console.print(
            f"Phase 2: wrote [bold]{args.phase2_out}[/bold] "
            f"({phase2_cfg.width}x{phase2_cfg.height} PPM)"
            f"{preset_note}; stats {stats}"
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
