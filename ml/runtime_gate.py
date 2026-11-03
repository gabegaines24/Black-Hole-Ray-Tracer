"""Runtime gate: decide whether to use the surrogate or the RK4 kernel for each ray.

Policy
------
- Only invoke the surrogate for rays where r₀ > SURROGATE_R_THRESHOLD (low-curvature region).
- For rays near the horizon (r₀ ≤ SURROGATE_R_THRESHOLD), always use the full RK4 kernel.
- If the surrogate is not loaded or inference fails, fall back to RK4 silently.
- ML gate and native/CUDA batch backends are mutually exclusive in ``phase2_render``:
  the render loop uses either per-ray ML routing or batched C integration, not both.
  Use ML gate for accuracy-oriented Python renders; use native batch for throughput.

This implements the deterministic heuristic from the original plan spec:
    Δλ = k / (1 + |∂g/∂r|)  → approximate low-curvature cutoff at r > 10M
No RL agent is used (deferred per plan).

Usage
-----
    from ml.runtime_gate import RuntimeGate
    gate = RuntimeGate.from_model_path("ml/models/surrogate.npz")
    result = gate.infer_or_integrate(y0, m=1.0, dlambda=0.1, max_steps=4000, r_escape=80.0)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ml.surrogate import MLPWeights

from ml.schema import (
    SENTINEL_VALUE,
    SURROGATE_R_THRESHOLD,
    STATUS_CAPTURED,
    STATUS_ESCAPED,
    STATUS_MAX_STEPS,
    STATUS_NUMERIC,
    denormalize_outputs,
    normalize_inputs,
)


class SurrogateResult:
    """Surrogate inference result (mirrors GeodesicTraceResult minimally)."""

    __slots__ = ("status", "r_min", "termination_r", "steps_taken", "from_surrogate")

    def __init__(
        self,
        status: int,
        r_min: float,
        termination_r: float,
        steps_taken: int,
        from_surrogate: bool = True,
    ) -> None:
        self.status = status
        self.r_min = r_min
        self.termination_r = termination_r
        self.steps_taken = steps_taken
        self.from_surrogate = from_surrogate


class RuntimeGate:
    """Routes each ray to the surrogate or to the full RK4 integrator.

    Parameters
    ----------
    weights : MLPWeights | None
        Loaded surrogate weights.  If None, always uses RK4.
    r_threshold : float
        Rays with r₀ > threshold go to surrogate; others go to RK4.
    """

    def __init__(
        self, weights: MLPWeights | None, r_threshold: float = SURROGATE_R_THRESHOLD
    ) -> None:
        self._weights = weights
        self._threshold = r_threshold

    @classmethod
    def from_model_path(
        cls, path: str | Path, r_threshold: float = SURROGATE_R_THRESHOLD
    ) -> "RuntimeGate":
        try:
            from ml.surrogate import load_surrogate

            weights = load_surrogate(path)
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            weights = None
        return cls(weights, r_threshold)

    @property
    def surrogate_loaded(self) -> bool:
        return self._weights is not None

    def infer_or_integrate(
        self,
        y0: np.ndarray,
        m: float,
        dlambda: float,
        max_steps: int,
        r_escape: float,
        r_horizon_epsilon: float = 1e-3,
    ) -> "SurrogateResult":
        """Route one ray to surrogate or RK4 and return a unified result."""
        if self._weights is not None:
            try:
                if float(y0[1]) > self._threshold:
                    return self._infer_surrogate(y0, max_steps)
            except (IndexError, TypeError, ValueError, FloatingPointError, OverflowError):
                pass
        return self._full_integrate(y0, m, dlambda, max_steps, r_escape, r_horizon_epsilon)

    def _infer_surrogate(self, y0: np.ndarray, max_steps: int) -> SurrogateResult:
        """Call the surrogate MLP for a single ray."""
        from ml.surrogate import predict_batch

        if y0.shape != (8,) or not np.all(np.isfinite(y0)):
            raise ValueError("Surrogate state must be finite with shape (8,).")
        weights = self._weights
        if weights is None:
            raise RuntimeError("Surrogate weights are unavailable.")
        r, th, phi_, vr, vth, vph = (float(y0[k]) for k in (1, 2, 3, 5, 6, 7))
        x_raw = np.array([[r, th, phi_, vr, vth, vph]], dtype=np.float32)
        pred = predict_batch(weights, normalize_inputs(x_raw))
        if pred.shape != (1, 4):
            raise ValueError("Surrogate prediction must have shape (1, 4).")
        pred_phys = denormalize_outputs(pred, max_steps=float(max_steps))[0]
        if not np.all(np.isfinite(pred_phys)):
            raise ValueError("Surrogate prediction must contain finite values.")
        status_value = int(round(float(pred_phys[0])))
        valid_statuses = (
            STATUS_CAPTURED,
            STATUS_ESCAPED,
            STATUS_MAX_STEPS,
            STATUS_NUMERIC,
        )
        if status_value not in valid_statuses:
            raise ValueError("Surrogate predicted invalid status.")
        radii: list[float] = []
        for value in pred_phys[1:3]:
            radius = float(value)
            if radius < 0.0 and radius != SENTINEL_VALUE:
                raise ValueError("Surrogate predicted invalid radius.")
            radii.append(radius)
        steps_value = float(pred_phys[3])
        if not math.isfinite(steps_value) or max_steps <= 0:
            raise ValueError("Surrogate predicted invalid step count.")
        steps = min(max(0, int(round(steps_value))), max_steps)
        return SurrogateResult(
            status=status_value,
            r_min=radii[0],
            termination_r=radii[1],
            steps_taken=steps,
            from_surrogate=True,
        )

    def _full_integrate(
        self,
        y0: np.ndarray,
        m: float,
        dlambda: float,
        max_steps: int,
        r_escape: float,
        r_horizon_epsilon: float,
    ) -> SurrogateResult:
        """Fall back to the Python RK4 integrator."""
        from blackhole_ray_tracer.phase1 import RayStatus
        from blackhole_ray_tracer.phase2_geodesic import trace_null_geodesic_3d

        _STATUS_MAP = {
            RayStatus.CAPTURED: STATUS_CAPTURED,
            RayStatus.ESCAPED: STATUS_ESCAPED,
            RayStatus.MAX_STEPS: STATUS_MAX_STEPS,
            RayStatus.NUMERICAL_ERROR: STATUS_NUMERIC,
        }
        res = trace_null_geodesic_3d(
            y0[:4], y0[4:],
            m=m, dlambda=dlambda, max_steps=max_steps,
            r_escape=r_escape, r_horizon_epsilon=r_horizon_epsilon,
            store_samples=False,
        )
        return SurrogateResult(
            status=_STATUS_MAP[res.status],
            r_min=float(res.r_min),
            termination_r=float(res.termination_r),
            steps_taken=res.steps_taken,
            from_surrogate=False,
        )
