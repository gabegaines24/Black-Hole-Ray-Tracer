"""Dataset generator: call the verified Phase 2 integrator to build ML training data.

Usage
-----
    uv run python -m ml.dataset --n-rays 5000 --out-path ml/data/train.npz

Output format
-------------
A compressed `.npz` with float32 `X`/`Y` arrays and required JSON metadata.
Metadata records coordinate convention, schema columns, status mapping, sentinel policy,
and integrator parameters including seed.
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np

from blackhole_ray_tracer.phase1 import RayStatus
from blackhole_ray_tracer.phase2_geodesic import trace_null_geodesic_3d
from ml.schema import (
    COORDINATE_CONVENTION,
    DATASET_FORMAT,
    DATASET_SCHEMA_VERSION,
    INPUT_COLUMNS,
    N_INPUTS,
    N_OUTPUTS,
    OUTPUT_COLUMNS,
    SENTINEL_VALUE,
    STATE_ORDER,
    STATUS_MAPPING,
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


def dataset_metadata(
    *,
    m: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    seed: int,
) -> dict[str, object]:
    """Build reproducibility metadata for one generated dataset."""
    values = (m, dlambda, r_escape)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Dataset integrator values must be finite.")
    if m <= 0.0 or dlambda <= 0.0 or r_escape <= 0.0:
        raise ValueError("Dataset integrator values must be positive.")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("Dataset max_steps must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Dataset seed must be an integer.")
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "format": DATASET_FORMAT,
        "coordinate_convention": COORDINATE_CONVENTION,
        "state_order": STATE_ORDER,
        "input_columns": INPUT_COLUMNS,
        "output_columns": OUTPUT_COLUMNS,
        "status_mapping": STATUS_MAPPING,
        "sentinel_policy": {
            "unknown_float": SENTINEL_VALUE,
            "nonfinite_terminal_values": "replace_with_unknown_float",
        },
        "integrator": {
            "name": "phase2.rk4_step",
            "m": float(m),
            "dlambda": float(dlambda),
            "max_steps": max_steps,
            "r_escape": float(r_escape),
            "seed": seed,
        },
    }


def _validate_dataset_metadata(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise ValueError("Dataset metadata must be a JSON object.")
    if metadata.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("Unsupported dataset schema version.")
    if metadata.get("format") != DATASET_FORMAT:
        raise ValueError("Unsupported dataset format.")
    for key, expected in (
        ("coordinate_convention", COORDINATE_CONVENTION),
        ("state_order", STATE_ORDER),
        ("input_columns", INPUT_COLUMNS),
        ("output_columns", OUTPUT_COLUMNS),
        ("status_mapping", STATUS_MAPPING),
    ):
        if metadata.get(key) != expected:
            raise ValueError(f"Dataset metadata field {key!r} is invalid.")
    sentinel_policy = metadata.get("sentinel_policy")
    if not isinstance(sentinel_policy, dict):
        raise ValueError("Dataset sentinel policy is missing.")
    if sentinel_policy.get("unknown_float") != SENTINEL_VALUE:
        raise ValueError("Dataset sentinel value is invalid.")
    integrator = metadata.get("integrator")
    if not isinstance(integrator, dict):
        raise ValueError("Dataset integrator metadata is missing.")
    required = ("m", "dlambda", "max_steps", "r_escape", "seed")
    if any(key not in integrator for key in required):
        raise ValueError("Dataset integrator metadata is incomplete.")
    return metadata


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
    """Generate a dataset of ``(X, Y)`` pairs from the Phase 2 integrator."""
    if isinstance(n_rays, bool) or not isinstance(n_rays, int) or n_rays < 0:
        raise ValueError("n_rays must be a non-negative integer.")
    dataset_metadata(
        m=m,
        dlambda=dlambda,
        max_steps=max_steps,
        r_escape=r_escape,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    rays = _sample_random_rays(n_rays * 2, rng, m=m)

    X_rows: list[list[float]] = []
    Y_rows: list[list[float]] = []
    for i, (ri, thi, phi_i, vt_i, vr_i, vth_i, vph_i) in enumerate(rays):
        if len(X_rows) >= n_rays:
            break
        x0 = np.array([0.0, ri, thi, phi_i], dtype=float)
        v0 = np.array([vt_i, vr_i, vth_i, vph_i], dtype=float)
        res = trace_null_geodesic_3d(
            x0,
            v0,
            m=m,
            dlambda=dlambda,
            max_steps=max_steps,
            r_escape=r_escape,
            store_samples=False,
        )
        status_int = float(_STATUS_MAP[res.status])
        r_min = float(res.r_min) if math.isfinite(res.r_min) else SENTINEL_VALUE
        term_r = (
            float(res.termination_r)
            if math.isfinite(res.termination_r)
            else SENTINEL_VALUE
        )
        X_rows.append([ri, thi, phi_i, vr_i, vth_i, vph_i])
        Y_rows.append([status_int, r_min, term_r, float(res.steps_taken)])
        if verbose and (i + 1) % max(1, n_rays // 10) == 0:
            print(f"  generated {len(X_rows)} / {n_rays} rays …")

    X = np.asarray(X_rows, dtype=np.float32).reshape((-1, N_INPUTS))
    Y = np.asarray(Y_rows, dtype=np.float32).reshape((-1, N_OUTPUTS))
    return X, Y


def save_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    path: str | Path,
    *,
    metadata: dict[str, object],
) -> None:
    """Save validated arrays and JSON metadata in one reproducible NPZ."""
    X_array = np.asarray(X, dtype=np.float32)
    Y_array = np.asarray(Y, dtype=np.float32)
    if X_array.ndim != 2 or X_array.shape[1] != N_INPUTS:
        raise ValueError(f"X must have shape (N, {N_INPUTS}).")
    if Y_array.ndim != 2 or Y_array.shape[1] != N_OUTPUTS:
        raise ValueError(f"Y must have shape (N, {N_OUTPUTS}).")
    if X_array.shape[0] != Y_array.shape[0]:
        raise ValueError("X and Y must have equal row counts.")
    if not np.all(np.isfinite(X_array)) or not np.all(np.isfinite(Y_array)):
        raise ValueError("Dataset arrays must contain only finite values.")
    _validate_dataset_metadata(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    np.savez_compressed(
        str(path),
        X=X_array,
        Y=Y_array,
        metadata_json=np.array(metadata_json),
    )
    print(f"Saved {len(X_array)} rays → {path}  (X:{X_array.shape} Y:{Y_array.shape})")


def load_dataset_metadata(path: str | Path) -> dict[str, object]:
    """Load and validate dataset metadata without returning training arrays."""
    with np.load(str(path), allow_pickle=False) as data:
        if "metadata_json" not in data:
            raise ValueError("Dataset metadata_json is missing.")
        raw = np.asarray(data["metadata_json"])
        if raw.shape != ():
            raise ValueError("Dataset metadata_json must be a scalar string.")
        try:
            metadata = json.loads(str(raw.item()))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Dataset metadata_json is invalid.") from exc
    return _validate_dataset_metadata(metadata)


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate arrays and metadata from a reproducible NPZ."""
    load_dataset_metadata(path)
    with np.load(str(path), allow_pickle=False) as data:
        if "X" not in data or "Y" not in data:
            raise ValueError("Dataset must contain X and Y arrays.")
        X = np.asarray(data["X"])
        Y = np.asarray(data["Y"])
    if X.dtype != np.float32 or Y.dtype != np.float32:
        raise ValueError("Dataset X and Y arrays must use float32.")
    if X.ndim != 2 or X.shape[1] != N_INPUTS:
        raise ValueError(f"X must have shape (N, {N_INPUTS}).")
    if Y.ndim != 2 or Y.shape[1] != N_OUTPUTS:
        raise ValueError(f"Y must have shape (N, {N_OUTPUTS}).")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have equal row counts.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("Dataset arrays must contain only finite values.")
    return X.copy(), Y.copy()


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
        args.n_rays,
        dlambda=args.dlambda,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    save_dataset(
        X,
        Y,
        args.out_path,
        metadata=dataset_metadata(
            m=1.0,
            dlambda=args.dlambda,
            max_steps=args.max_steps,
            r_escape=80.0,
            seed=args.seed,
        ),
    )


if __name__ == "__main__":
    main()
