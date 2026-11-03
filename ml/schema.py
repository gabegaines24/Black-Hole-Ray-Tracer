"""Data contract for the Phase 3 ML surrogate model.

Input / output schema
---------------------
The surrogate operates in spherical Schwarzschild coordinates.

Inputs (6 scalars per ray, normalised):
    r           — initial radial coordinate  [M units]
    theta       — initial polar angle        [radians, 0..π]
    phi         — initial azimuthal angle    [radians, 0..2π]
    v_r         — initial v^r (Schwarzschild coordinate velocity)
    v_theta     — initial v^theta
    v_phi       — initial v^phi

The time component v^t is *not* an input; it is determined by the null
constraint and is always re-derived from the other six.

Outputs (4 values per ray):
    status      — integer BH_RT_STATUS_* code (0=captured, 1=escaped, 2=max_steps, 3=numeric)
    r_min       — minimum r reached           [M units] (NaN if not reached)
    termination_r — r at termination          [M units]
    steps_taken — integrator steps consumed

Normalisation (applied before feeding to the network):
    r       → r / r_ref          where r_ref = 30.0 M (typical observer distance)
    theta   → (theta − π/2) / π  (centred, scaled to [-0.5, 0.5])
    phi     → phi / (2π)
    v_r     → v_r / v_ref        where v_ref = 1.0 (geometric units)
    v_theta → v_theta * r_ref    (dimensionless)
    v_phi   → v_phi * r_ref      (dimensionless)

Fallback threshold:
    The surrogate is *only* invoked when the ray starts in a low-curvature
    region: r > SURROGATE_R_THRESHOLD.  Rays closer than this are always
    integrated with the full RK4 kernel.
"""

from __future__ import annotations

import numpy as np

# Normalisation constants
R_REF: float = 30.0       # M units — typical observer distance
V_REF: float = 1.0        # geometric units

# Only use surrogate for rays starting beyond this radius
SURROGATE_R_THRESHOLD: float = 10.0   # M units

# Status codes (mirrors bh_rt_status.h)
STATUS_CAPTURED: int = 0
STATUS_ESCAPED: int = 1
STATUS_MAX_STEPS: int = 2
STATUS_NUMERIC: int = 3

# Feature dimension
N_INPUTS: int = 6
# Label dimension
N_OUTPUTS: int = 4

# Feature and label column names (for dataset documentation)
INPUT_COLUMNS: list[str] = ["r", "theta", "phi", "v_r", "v_theta", "v_phi"]
OUTPUT_COLUMNS: list[str] = ["status", "r_min", "termination_r", "steps_taken"]


DATASET_SCHEMA_VERSION: int = 1
DATASET_FORMAT: str = "blackhole_phase2_dataset"
MODEL_SCHEMA_VERSION: int = 1
MODEL_FORMAT: str = "blackhole_numpy_mlp"
COORDINATE_CONVENTION: str = "Schwarzschild spherical coordinates"
STATE_ORDER: list[str] = [
    "t",
    "r",
    "theta",
    "phi",
    "v^t",
    "v^r",
    "v^theta",
    "v^phi",
]
SENTINEL_VALUE: float = -1.0
STATUS_MAPPING: dict[str, int] = {
    "captured": STATUS_CAPTURED,
    "escaped": STATUS_ESCAPED,
    "max_steps": STATUS_MAX_STEPS,
    "numerical_error": STATUS_NUMERIC,
}

def _validated_array(values: np.ndarray, width: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != width:
        raise ValueError(f"{name} must have trailing shape ({width},).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validated_max_steps(max_steps: float) -> float:
    value = float(max_steps)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("max_steps must be finite and positive.")
    return value


def normalize_inputs(x: np.ndarray) -> np.ndarray:
    """Normalise raw ray inputs (shape (..., 6)) to network-ready features."""
    x = _validated_array(x, N_INPUTS, "inputs")
    out = np.empty_like(x)
    out[..., 0] = x[..., 0] / R_REF
    out[..., 1] = (x[..., 1] - np.pi / 2.0) / np.pi
    out[..., 2] = x[..., 2] / (2.0 * np.pi)
    out[..., 3] = x[..., 3] / V_REF
    out[..., 4] = x[..., 4] * R_REF
    out[..., 5] = x[..., 5] * R_REF
    return out


def normalize_outputs(y: np.ndarray, *, max_steps: float = 4000.0) -> np.ndarray:
    """Scale training labels to O(1) for stable MLP regression."""
    y = _validated_array(y, N_OUTPUTS, "outputs").copy()
    y[..., 1] /= R_REF
    y[..., 2] /= R_REF
    y[..., 3] /= _validated_max_steps(max_steps)
    return y


def denormalize_outputs(y: np.ndarray, *, max_steps: float = 4000.0) -> np.ndarray:
    """Inverse of ``normalize_outputs`` for inference."""
    y = _validated_array(y, N_OUTPUTS, "outputs").copy()
    y[..., 1] *= R_REF
    y[..., 2] *= R_REF
    y[..., 3] *= _validated_max_steps(max_steps)
    return y
