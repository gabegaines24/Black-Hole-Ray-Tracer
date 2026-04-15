"""Phase 1 drivers: Steps A–F (RK4 through tuning presets / benchmarks).

Run with:
    PYTHONPATH=src python -m blackhole_ray_tracer.phase1_driver
    PYTHONPATH=src python -m blackhole_ray_tracer.phase1_driver --step b
    PYTHONPATH=src python -m blackhole_ray_tracer.phase1_driver --step d
    PYTHONPATH=src python -m blackhole_ray_tracer.phase1_driver --step e --out einstein_ring.ppm
    PYTHONPATH=src python -m blackhole_ray_tracer.phase1_driver --step e --preset quality --out ring.ppm
    PYTHONPATH=src python -m blackhole_ray_tracer.phase1_driver --step f
"""

from __future__ import annotations

import argparse
import csv
from math import cos, sin

import numpy as np

from .phase1 import (
    BatchRayRow,
    batch_schwarzschild_rays,
    format_step_b_log,
    format_step_d_table,
    harmonic_oscillator_derivs,
    rk4_step,
    run_rk4_sanity,
    step_b_trajectory_is_finite,
    trace_single_schwarzschild_ray,
)
from .phase1_image import render_einstein_ring_image, write_ppm_rgb
from .phase1_tuning import PRESETS, format_step_f_report


def run_step_a_with_diagnostics(
    dt: float,
    total_time: float,
    omega: float,
) -> tuple[float, float, float]:
    """Return max position error, mean error, and max energy drift."""
    steps = int(total_time / dt)
    t = 0.0
    y = np.array([1.0, 0.0], dtype=float)
    e0 = 0.5 * (y[1] ** 2 + (omega**2) * (y[0] ** 2))
    max_energy_drift = 0.0

    for _ in range(steps):
        y = rk4_step(harmonic_oscillator_derivs, t, y, dt, omega)
        t += dt
        e = 0.5 * (y[1] ** 2 + (omega**2) * (y[0] ** 2))
        max_energy_drift = max(max_energy_drift, abs(e - e0))

    # Compare final state against analytic solution.
    x_exact = cos(omega * t)
    v_exact = -omega * sin(omega * t)
    final_state_error = float(np.hypot(y[0] - x_exact, y[1] - v_exact))

    max_err, mean_err = run_rk4_sanity(dt=dt, total_time=total_time, omega=omega)
    return max_err, mean_err, max(max_energy_drift, final_state_error)


def _write_step_d_csv(rows: list[BatchRayRow], path: str, m: float) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "impact_b",
                "status",
                "r_min",
                "phi_sweep_rad",
                "termination_r",
                "steps_taken",
                "four_M_over_b",
            ]
        )
        for row in rows:
            theory = 4.0 * m / row.impact_b if row.impact_b > 0 else ""
            w.writerow(
                [
                    row.impact_b,
                    row.status.value,
                    row.r_min,
                    row.phi_sweep_rad,
                    row.termination_r,
                    row.steps_taken,
                    theory,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(prog="phase1-driver")
    parser.add_argument(
        "--step",
        choices=("a", "b", "ab", "d", "e", "f"),
        default="a",
        help="a: RK4 sanity; b: single ray; ab: A+B; d: batch; e: PPM image; f: tune (presets + benchmark)",
    )
    parser.add_argument("--dt", type=float, default=0.02, help="RK4 step size (Step A)")
    parser.add_argument("--time", type=float, default=8.0, help="Total integration time (Step A)")
    parser.add_argument("--omega", type=float, default=1.0, help="Oscillator angular frequency (Step A)")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Pass threshold for max position error and energy drift (Step A)",
    )
    parser.add_argument(
        "--log-first",
        type=int,
        default=50,
        help="Number of (r, phi) rows to print for Step B",
    )
    parser.add_argument("--b", type=float, default=6.0, help="Impact parameter (Step B)")
    parser.add_argument("--m", type=float, default=1.0, help="Black hole mass M in geometric units (Step B/D)")
    parser.add_argument(
        "--b-min",
        type=float,
        default=2.5,
        help="Minimum impact parameter in Step D sweep",
    )
    parser.add_argument(
        "--b-max",
        type=float,
        default=10.0,
        help="Maximum impact parameter in Step D sweep",
    )
    parser.add_argument(
        "--n-b",
        type=int,
        default=24,
        help="Number of impact parameters in Step D (linspace)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional path to write Step D results as CSV",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="einstein_ring.ppm",
        help="Output PPM path for Step E (default: einstein_ring.ppm)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=72,
        help="Image width in pixels (Step E)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=72,
        help="Image height in pixels (Step E)",
    )
    parser.add_argument(
        "--phi-max",
        type=float,
        default=8.0,
        help="Max phi for ray integration (Step B–E)",
    )
    parser.add_argument(
        "--dphi",
        type=float,
        default=0.012,
        help="phi step for geodesic integration (Step B–E; larger = faster, coarser)",
    )
    parser.add_argument(
        "--r-escape",
        type=float,
        default=80.0,
        help="Radius above which a ray counts as escaped (Step B–E)",
    )
    parser.add_argument(
        "--preset",
        choices=("fast", "balanced", "quality"),
        default=None,
        help="Step E: apply quality/speed preset (overrides width/height/dphi/phi_max/r_escape/b_min/b_max)",
    )
    args = parser.parse_args()

    if args.step == "e" and args.preset is not None:
        p = PRESETS[args.preset]
        args.width = p["width"]
        args.height = p["height"]
        args.dphi = p["dphi"]
        args.phi_max = p["phi_max"]
        args.r_escape = p["r_escape"]
        args.b_min = p["b_min"]
        args.b_max = p["b_max"]

    run_a = args.step in ("a", "ab")
    run_b = args.step in ("b", "ab")
    run_d = args.step == "d"
    run_e = args.step == "e"
    run_f = args.step == "f"

    if run_a:
        max_err, mean_err, max_energy_drift = run_step_a_with_diagnostics(
            dt=args.dt,
            total_time=args.time,
            omega=args.omega,
        )
        passed = max_err < args.tol and max_energy_drift < args.tol
        status = "PASS" if passed else "FAIL"

        print("Step A: RK4 sanity (harmonic oscillator)")
        print(f"- dt: {args.dt}")
        print(f"- total_time: {args.time}")
        print(f"- omega: {args.omega}")
        print(f"- max |x_num - x_exact|: {max_err:.3e}")
        print(f"- mean |x_num - x_exact|: {mean_err:.3e}")
        print(f"- max energy/state drift: {max_energy_drift:.3e}")
        print(f"- tolerance: {args.tol:.1e}")
        print(f"- result: {status}")
        if not passed:
            raise SystemExit(1)
        if run_b:
            print()

    if run_b:
        result = trace_single_schwarzschild_ray(
            b=args.b,
            m=args.m,
            phi_max=args.phi_max,
            dphi=args.dphi,
            r_escape=args.r_escape,
        )
        print(format_step_b_log(result, first_n=args.log_first))
        b_ok = step_b_trajectory_is_finite(result)
        print(f"- result: {'PASS' if b_ok else 'FAIL'}")
        if not b_ok:
            raise SystemExit(1)

    if run_d:
        b_vals = np.linspace(args.b_min, args.b_max, args.n_b)
        rows = batch_schwarzschild_rays(
            b_vals,
            m=args.m,
            phi_max=args.phi_max,
            dphi=args.dphi,
            r_escape=args.r_escape,
        )
        print(format_step_d_table(rows, m=args.m))
        if args.csv:
            _write_step_d_csv(rows, args.csv, m=args.m)
            print(f"- wrote CSV: {args.csv}")

    if run_e:
        rgb, _ = render_einstein_ring_image(
            args.width,
            args.height,
            m=args.m,
            b_min=args.b_min,
            b_max=args.b_max,
            phi_max=args.phi_max,
            dphi=args.dphi,
            r_escape=args.r_escape,
        )
        write_ppm_rgb(args.out, rgb)
        print("Step E (Einstein-ring prototype image)")
        print(f"- wrote PPM: {args.out} ({args.width}x{args.height})")
        if args.preset:
            print(f"- preset: {args.preset}")
        print(f"- b maps from center={args.b_min} to edge={args.b_max} (see phase1_image.py)")

    if run_f:
        print(format_step_f_report())


if __name__ == "__main__":
    main()
