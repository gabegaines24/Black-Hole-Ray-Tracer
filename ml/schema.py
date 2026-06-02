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


def normalize_inputs(x: np.ndarray) -> np.ndarray:
    """Normalise raw ray inputs (shape (..., 6)) to network-ready features.

    Columns: r, theta, phi, v_r, v_theta, v_phi.
    """
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    out[..., 0] = x[..., 0] / R_REF
    out[..., 1] = (x[..., 1] - np.pi / 2.0) / np.pi
    out[..., 2] = x[..., 2] / (2.0 * np.pi)
    out[..., 3] = x[..., 3] / V_REF
    out[..., 4] = x[..., 4] * R_REF
    out[..., 5] = x[..., 5] * R_REF
    return out


def denormalize_outputs(y: np.ndarray) -> np.ndarray:
    """Inverse-transform network outputs (..., 4) back to physical units.

    Output columns: status (int), r_min (M), termination_r (M), steps_taken.
    These are returned as-is (no normalisation applied in training yet);
    placeholder for future scaling if needed.
    """
    return np.asarray(y, dtype=np.float32)
