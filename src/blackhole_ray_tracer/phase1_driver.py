"""Phase 1 Step A driver: RK4 sanity test on a harmonic oscillator.

Run with:
    python -m blackhole_ray_tracer.phase1_driver
"""

from __future__ import annotations

import argparse
from math import cos, sin

import numpy as np

from .phase1 import harmonic_oscillator_derivs, rk4_step, run_rk4_sanity


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


def main() -> None:
    parser = argparse.ArgumentParser(prog="phase1-driver")
    parser.add_argument("--dt", type=float, default=0.02, help="RK4 step size")
    parser.add_argument("--time", type=float, default=8.0, help="Total integration time")
    parser.add_argument("--omega", type=float, default=1.0, help="Oscillator angular frequency")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Pass threshold for max position error and energy drift",
    )
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
