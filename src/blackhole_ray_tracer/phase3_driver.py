"""CLI driver for Phase 3 Kerr null-geodesic rendering.

Usage examples
--------------
    # Quick 64×64 Schwarzschild-equivalent (a=0)
    uv run python -m blackhole_ray_tracer.phase3_driver --render

    # Kerr spin a=0.9M, 128×128
    uv run python -m blackhole_ray_tracer.phase3_driver --render --spin 0.9 --width 128

    # High-quality with 2× anti-aliasing
    uv run python -m blackhole_ray_tracer.phase3_driver --render --spin 0.5 --aa 2 --width 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_config(args):
    from .phase2_disk import DiskConfig
    from .phase3_types import KerrRenderConfig

    disk = DiskConfig() if args.disk else None
    return KerrRenderConfig(
        width=args.width,
        height=args.height,
        m=args.mass,
        a=args.spin,
        r_observer=args.r_observer,
        fov_deg=args.fov,
        dlambda=args.dlambda,
        max_steps=args.max_steps,
        r_escape=args.r_escape,
        sky_mode=args.sky,
        disk=disk,
        supersample=args.aa,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="phase3_driver",
        description="Render a Kerr black-hole image (Boyer-Lindquist null geodesics).",
    )
    parser.add_argument("--render", action="store_true", help="Render and save PPM.")
    parser.add_argument("--out", type=str, default="kerr_3d.ppm", help="Output PPM path.")
    parser.add_argument("--width",  type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--mass",   type=float, default=1.0, metavar="M")
    parser.add_argument("--spin",   type=float, default=0.0, metavar="A",
                        help="Kerr spin parameter a (0 = Schwarzschild, |a| ≤ M).")
    parser.add_argument("--r-observer", type=float, default=30.0)
    parser.add_argument("--fov",    type=float, default=60.0)
    parser.add_argument("--dlambda", type=float, default=0.06)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--r-escape", type=float, default=80.0)
    parser.add_argument("--sky", choices=("gradient", "flat"), default="gradient")
    parser.add_argument("--disk", action="store_true",
                        help="Overlay thin equatorial accretion disk.")
    parser.add_argument("--aa", "--supersample", dest="aa", type=int, default=1, metavar="S",
                        help="Super-sample factor for anti-aliasing (1 = off, 2 = 2×).")

    args = parser.parse_args(argv)

    if not args.render:
        parser.print_help()
        return

    from .phase1_image import write_ppm_rgb
    from .phase3_render import render_kerr_3d_image

    cfg = _build_config(args)
    spin_tag = f"a={cfg.a:.3f}"
    print(f"Rendering Kerr image {cfg.width}×{cfg.height} [{spin_tag}] …", file=sys.stderr)
    rgb, stats = render_kerr_3d_image(cfg)

    out = Path(args.out)
    write_ppm_rgb(str(out), rgb)
    frac = stats.get("frac_captured", float("nan"))
    backend = stats.get("backend", "?")
    print(
        f"  → {out}  captured={frac:.1%}  backend={backend}  "
        f"spin={spin_tag}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
