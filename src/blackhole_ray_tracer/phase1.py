"""Phase 1 helpers: RK4 sanity and single-ray Schwarzschild demo.

This module intentionally keeps the math compact and inspectable so it is easy
to learn while building.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import cos, sin

import numpy as np


def rk4_step(
    derivs: Callable[..., np.ndarray], x: float, y: np.ndarray, h: float, *args: float
) -> np.ndarray:
    """Advance one RK4 step for y' = f(x, y)."""
    k1 = derivs(x, y, *args)
    k2 = derivs(x + 0.5 * h, y + 0.5 * h * k1, *args)
    k3 = derivs(x + 0.5 * h, y + 0.5 * h * k2, *args)
    k4 = derivs(x + h, y + h * k3, *args)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def harmonic_oscillator_derivs(_: float, y: np.ndarray, omega: float) -> np.ndarray:
    """Toy ODE for Step A: y = [position, velocity]."""
    pos, vel = y
    return np.array([vel, -(omega**2) * pos], dtype=float)


def run_rk4_sanity(
    dt: float = 0.02, total_time: float = 8.0, omega: float = 1.0
) -> tuple[float, float]:
    """Run RK4 sanity test and return max/mean absolute error."""
    steps = int(total_time / dt)
    t = 0.0
    y = np.array([1.0, 0.0], dtype=float)  # x(0)=1, v(0)=0
    errs: list[float] = []

    for _ in range(steps):
        exact = cos(omega * t)
        errs.append(abs(y[0] - exact))
        y = rk4_step(harmonic_oscillator_derivs, t, y, dt, omega)
        t += dt

    return float(max(errs)), float(np.mean(errs))


def schwarzschild_u_derivs(_: float, y: np.ndarray, m: float) -> np.ndarray:
    """Equatorial null-geodesic ODE in u(phi)=1/r form.

    d2u/dphi2 + u = 3 M u^2  -> first-order system:
    y0 = u
    y1 = du/dphi
    """
    u, du_dphi = y
    d2u_dphi2 = 3.0 * m * (u**2) - u
    return np.array([du_dphi, d2u_dphi2], dtype=float)


@dataclass
class RayTraceResult:
    status: "RayStatus"
    steps_taken: int
    max_steps: int
    termination_phi: float
    termination_r: float
    phis: np.ndarray
    rs: np.ndarray
    xs: np.ndarray
    ys: np.ndarray


class RayStatus(str, Enum):
    CAPTURED = "captured"
    ESCAPED = "escaped"
    MAX_STEPS = "max_steps"
    NUMERICAL_ERROR = "numerical_error"


def trace_single_schwarzschild_ray(
    b: float = 6.0,
    m: float = 1.0,
    phi_start: float = 0.2,
    phi_max: float = 8.0,
    dphi: float = 0.002,
    r_capture: float | None = None,
    r_escape: float = 80.0,
) -> RayTraceResult:
    """Step B: integrate a single ray and return trajectory arrays.

    `b` is the impact parameter. Start far away with asymptotic initial data:
    u(0)=0, du/dphi(0)=1/b.
    """
    if r_capture is None:
        r_capture = 2.0 * m + 1e-3

    steps = int((phi_max - phi_start) / dphi)
    phi = phi_start
    # Flat-space asymptotic initial condition near infinity.
    y = np.array([sin(phi_start) / b, cos(phi_start) / b], dtype=float)

    phis: list[float] = []
    rs: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    status = RayStatus.MAX_STEPS
    termination_r = float("nan")
    steps_taken = 0

    for step_idx in range(steps):
        u = y[0]
        if not np.isfinite(u):
            status = RayStatus.NUMERICAL_ERROR
            steps_taken = step_idx
            break
        if u <= 0.0:
            status = RayStatus.ESCAPED
            termination_r = float("inf")
            steps_taken = step_idx
            break

        r = 1.0 / u
        if not np.isfinite(r):
            status = RayStatus.NUMERICAL_ERROR
            steps_taken = step_idx
            break
        x = r * cos(phi)
        y_cart = r * sin(phi)

        phis.append(phi)
        rs.append(r)
        xs.append(x)
        ys.append(y_cart)
        termination_r = r
        steps_taken = step_idx + 1

        if r < r_capture:
            status = RayStatus.CAPTURED
            break
        if r > r_escape:
            status = RayStatus.ESCAPED
            break

        y = rk4_step(schwarzschild_u_derivs, phi, y, dphi, m)
        phi += dphi

    return RayTraceResult(
        status=status,
        steps_taken=steps_taken,
        max_steps=steps,
        termination_phi=phi,
        termination_r=termination_r,
        phis=np.array(phis),
        rs=np.array(rs),
        xs=np.array(xs),
        ys=np.array(ys),
    )


def summarize_phase1_a_b() -> str:
    """Return a compact text summary for CLI printing.

    Includes Step C style termination diagnostics for the single-ray run.
    """
    max_err, mean_err = run_rk4_sanity()
    result = trace_single_schwarzschild_ray()
    if len(result.rs) > 0:
        r_min = float(np.min(result.rs))
        r_last = float(result.rs[-1])
    else:
        r_min = float("nan")
        r_last = float("nan")

    return (
        "Step A (RK4 sanity)\n"
        f"- max |x_num - x_exact|: {max_err:.3e}\n"
        f"- mean |x_num - x_exact|: {mean_err:.3e}\n\n"
        "Step B (single Schwarzschild ray)\n"
        f"- status: {result.status.value}\n"
        f"- steps: {result.steps_taken}/{result.max_steps}\n"
        f"- samples: {len(result.rs)}\n"
        f"- min radius: {r_min:.3f}\n"
        f"- final radius: {r_last:.3f}\n"
        f"- termination radius: {result.termination_r:.3f}\n"
        f"- termination phi: {result.termination_phi:.3f}\n"
        f"- schwarzschild radius (2M): {2.0 * 1.0:.3f}\n"
        f"- photon sphere (3M): {3.0 * 1.0:.3f}\n"
        "\nTip: change impact parameter b in trace_single_schwarzschild_ray()."
    )

