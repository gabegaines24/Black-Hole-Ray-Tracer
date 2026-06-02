"""Dataset generator: call the verified Phase 2 integrator to build ML training data.

Usage
-----
    uv run python -m ml.dataset --n-rays 5000 --out-path ml/data/train.npz

Output format
-------------
A compressed `.npz` file with two arrays:
    X: float32 (N, 6) — raw inputs (r, theta, phi, v_r, v_theta, v_phi)
    Y: float32 (N, 4) — outputs (status, r_min, termination_r, steps_taken)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# Add project src to path if running as a script from the ml/ directory
import sys
_root = Path(__file__).resolve().parents[1]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from blackhole_ray_tracer.phase1 import RayStatus
from blackhole_ray_tracer.phase2_camera import (
    make_camera_from_config,
    initial_position_observer,
    static_observer_null_direction,
)
from blackhole_ray_tracer.phase2_geodesic import trace_null_geodesic_3d
from ml.schema import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    STATUS_CAPTURED,
    STATUS_ESCAPED,
    STATUS_MAX_STEPS,
    STATUS_NUMERIC,
)

_STATUS_MAP = {
    RayStatus.CAPTURED: STATUS_CAPTURED,
    RayStatus.ESCAPED: STATUS_ESCAPED,
    RayStatus.MAX_STEPS: STATUS_MAX_STEPS,
    RayStatus.NUMERICAL_ERROR: STATUS_NUMERIC,
}


def _sample_random_rays(n: int, rng: np.random.Generator, m: float = 1.0):
    """Sample random valid initial conditions from a spherical shell."""
    r = rng.uniform(10.0, 50.0, n)
    theta = rng.uniform(0.1, math.pi - 0.1, n)
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    # Random unit spatial direction in local frame; mapped to coordinate velocities
    # by a simple equatorial-observer approximation (exact tetrad not critical for
    # generating varied data — accuracy is validated separately).
    vr_hat = rng.uniform(-1.0, 0.0, n)     # inward component
    vth_hat = rng.uniform(-0.5, 0.5, n)
    vph_hat = rng.uniform(-0.5, 0.5, n)
    norm = np.sqrt(vr_hat**2 + vth_hat**2 + vph_hat**2) + 1e-12
    vr_hat /= norm
    vth_hat /= norm
    vph_hat /= norm

    rays = []
    for i in range(n):
        ri, thi, phi_i = float(r[i]), float(theta[i]), float(phi[i])
        f = 1.0 - 2.0 * m / ri
        if f <= 0.0:
            continue
        vt_i = 1.0 / math.sqrt(f)
        vr_i = float(vr_hat[i]) * math.sqrt(f)
        vth_i = float(vth_hat[i]) / ri
        vph_i = float(vph_hat[i]) / (ri * math.sin(thi))
        rays.append((ri, thi, phi_i, vt_i, vr_i, vth_i, vph_i))
    return rays


def generate_dataset(
    n_rays: int,
    m: float = 1.0,
    dlambda: float = 0.1,
    max_steps: int = 4000,
    r_escape: float = 80.0,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset of (X, Y) pairs by calling the Python integrator.

    Returns
    -------
    X : float32 (N, 6) — inputs: r, theta, phi, v_r, v_theta, v_phi
    Y : float32 (N, 4) — outputs: status, r_min, termination_r, steps_taken
    """
    rng = np.random.default_rng(seed)
    rays = _sample_random_rays(n_rays * 2, rng, m=m)   # oversample, trim later

    X_rows, Y_rows = [], []
    for i, (ri, thi, phi_i, vt_i, vr_i, vth_i, vph_i) in enumerate(rays):
        if len(X_rows) >= n_rays:
            break
        x0 = np.array([0.0, ri, thi, phi_i], dtype=float)
        v0 = np.array([vt_i, vr_i, vth_i, vph_i], dtype=float)
        res = trace_null_geodesic_3d(
            x0, v0, m=m, dlambda=dlambda, max_steps=max_steps,
            r_escape=r_escape, store_samples=False,
        )
        status_int = float(_STATUS_MAP[res.status])
        r_min = float(res.r_min) if math.isfinite(res.r_min) else -1.0
        term_r = float(res.termination_r) if math.isfinite(res.termination_r) else -1.0
        X_rows.append([ri, thi, phi_i, vr_i, vth_i, vph_i])
        Y_rows.append([status_int, r_min, term_r, float(res.steps_taken)])
        if verbose and (i + 1) % max(1, n_rays // 10) == 0:
            print(f"  generated {len(X_rows)} / {n_rays} rays …")

    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_rows, dtype=np.float32)
    return X, Y


def save_dataset(X: np.ndarray, Y: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), X=X, Y=Y)
    print(f"Saved {len(X)} rays → {path}  (X:{X.shape} Y:{Y.shape})")


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(str(path))
    return data["X"].astype(np.float32), data["Y"].astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate ML training dataset from Phase 2 integrator")
    parser.add_argument("--n-rays", type=int, default=2000,
                        help="Number of ray samples to generate.")
    parser.add_argument("--out-path", type=str, default="ml/data/train.npz",
                        help="Output .npz file path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dlambda", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=4000)
    args = parser.parse_args(argv)

    print(f"Generating {args.n_rays} rays (seed={args.seed}) …")
    X, Y = generate_dataset(
        args.n_rays, dlambda=args.dlambda, max_steps=args.max_steps, seed=args.seed
    )
    save_dataset(X, Y, args.out_path)


if __name__ == "__main__":
    main()
