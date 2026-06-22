"""ML scaffold tests.

Tests
-----
1. test_schema_normalize_roundtrip   — normalize then check column bounds
2. test_dataset_shapes               — generate 50 rays, check X(50,6) Y(50,4)
3. test_surrogate_forward_shape      — random weights forward pass (N,6)→(N,4)
4. test_runtime_gate_routes_correctly — r > threshold → surrogate path, r ≤ → RK4 path
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure both `src/` and the repo root (for `ml/`) are on sys.path.
_repo_root = Path(__file__).resolve().parents[1]
for _p in [str(_repo_root / "src"), str(_repo_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml.schema import (
    N_INPUTS,
    N_OUTPUTS,
    R_REF,
    SURROGATE_R_THRESHOLD,
    V_REF,
    normalize_inputs,
    denormalize_outputs,
)
from ml.dataset import generate_dataset
from ml.surrogate import forward, init_random_weights
from ml.runtime_gate import RuntimeGate, SurrogateResult


# ── 1. Schema normalisation ────────────────────────────────────────────────────

class TestSchemaNormalize:
    def _raw_sample(self, n: int = 100) -> np.ndarray:
        rng = np.random.default_rng(0)
        r = rng.uniform(5.0, 60.0, (n, 1))
        theta = rng.uniform(0.05, math.pi - 0.05, (n, 1))
        phi = rng.uniform(0.0, 2 * math.pi, (n, 1))
        vr = rng.uniform(-1.0, 0.0, (n, 1))
        vth = rng.uniform(-0.2, 0.2, (n, 1))
        vph = rng.uniform(-0.2, 0.2, (n, 1))
        return np.concatenate([r, theta, phi, vr, vth, vph], axis=1).astype(np.float32)

    def test_output_shape(self):
        x = self._raw_sample(50)
        x_norm = normalize_inputs(x)
        assert x_norm.shape == (50, N_INPUTS)

    def test_r_column_scaled(self):
        x = self._raw_sample(200)
        x_norm = normalize_inputs(x)
        # r column should be x[:,0]/R_REF
        np.testing.assert_allclose(x_norm[:, 0], x[:, 0] / R_REF, rtol=1e-5)

    def test_theta_column_centred(self):
        """θ column should be (θ − π/2)/π ∈ (−0.5, 0.5) for θ ∈ (0, π)."""
        x = self._raw_sample(200)
        x_norm = normalize_inputs(x)
        assert x_norm[:, 1].min() > -0.6
        assert x_norm[:, 1].max() < 0.6

    def test_phi_column_in_unit_interval(self):
        x = self._raw_sample(200)
        x_norm = normalize_inputs(x)
        assert x_norm[:, 2].min() >= 0.0 - 1e-5
        assert x_norm[:, 2].max() <= 1.0 + 1e-5

    def test_denormalize_is_identity(self):
        """denormalize_outputs is currently a no-op (placeholder)."""
        y = np.random.rand(10, N_OUTPUTS).astype(np.float32)
        y2 = denormalize_outputs(y)
        np.testing.assert_array_equal(y, y2)

    def test_normalize_2d_batch(self):
        x = self._raw_sample(5)
        x_norm = normalize_inputs(x)
        assert x_norm.dtype == np.float32
        assert x_norm.shape == (5, 6)

    def test_normalize_1d_single_ray(self):
        x = np.array([20.0, math.pi / 2, 0.5, -0.3, 0.1, 0.05], dtype=np.float32)
        x_norm = normalize_inputs(x)
        assert x_norm.shape == (6,)
        assert x_norm[0] == pytest.approx(20.0 / R_REF, rel=1e-5)


# ── 2. Dataset generation ──────────────────────────────────────────────────────

class TestDatasetShapes:
    """generate_dataset(50) must return X(N,6) and Y(N,4) for N close to 50."""

    def test_shapes(self):
        X, Y = generate_dataset(50, dlambda=0.15, max_steps=1000, verbose=False)
        assert X.ndim == 2 and X.shape[1] == N_INPUTS, f"X.shape={X.shape}"
        assert Y.ndim == 2 and Y.shape[1] == N_OUTPUTS, f"Y.shape={Y.shape}"
        # Should have generated up to 50 valid rays (may be fewer if sampling
        # rejects some; allow ≥ 40 to pass)
        assert len(X) >= 40, f"Too few rays generated: {len(X)}"
        assert len(X) == len(Y)

    def test_dtype(self):
        X, Y = generate_dataset(10, dlambda=0.15, max_steps=500, verbose=False)
        assert X.dtype == np.float32
        assert Y.dtype == np.float32

    def test_status_column_valid(self):
        """Status column (Y[:,0]) should only contain {0,1,2,3} codes."""
        X, Y = generate_dataset(30, dlambda=0.15, max_steps=800, verbose=False)
        statuses = Y[:, 0].astype(int)
        assert set(statuses).issubset({0, 1, 2, 3}), f"Unexpected statuses: {set(statuses)}"

    def test_r_range(self):
        """Input r values should fall within the sampling range [10, 50]."""
        X, Y = generate_dataset(30, dlambda=0.15, max_steps=500, verbose=False)
        assert X[:, 0].min() >= 9.9, "r_min out of range"
        assert X[:, 0].max() <= 50.1, "r_max out of range"


# ── 3. Surrogate forward pass ──────────────────────────────────────────────────

class TestSurrogateForward:
    def test_forward_shape(self):
        rng = np.random.default_rng(42)
        weights = init_random_weights(rng)
        X = rng.standard_normal((20, 6)).astype(np.float32)
        Y = forward(weights, X)
        assert Y.shape == (20, N_OUTPUTS), f"Expected (20,4), got {Y.shape}"

    def test_forward_dtype(self):
        weights = init_random_weights()
        X = np.random.rand(5, 6).astype(np.float32)
        Y = forward(weights, X)
        # numpy matmul may promote float32 to float64 on some platforms; accept both
        assert Y.dtype in (np.float32, np.float64)

    def test_forward_single_row(self):
        weights = init_random_weights()
        X = np.zeros((1, 6), dtype=np.float32)
        Y = forward(weights, X)
        assert Y.shape == (1, N_OUTPUTS)

    def test_no_nan_in_output(self):
        weights = init_random_weights()
        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 6)).astype(np.float32)
        Y = forward(weights, X)
        assert np.all(np.isfinite(Y)), "NaN/Inf in forward pass output"

    def test_output_changes_with_input(self):
        weights = init_random_weights()
        X1 = np.ones((3, 6), dtype=np.float32)
        X2 = np.zeros((3, 6), dtype=np.float32)
        Y1 = forward(weights, X1)
        Y2 = forward(weights, X2)
        assert not np.allclose(Y1, Y2), "Forward pass insensitive to input"


# ── 4. RuntimeGate routing ─────────────────────────────────────────────────────

class TestRuntimeGateRouting:
    """r > threshold → surrogate path, r ≤ threshold → RK4 path."""

    def _make_y0(self, r: float, m: float = 1.0) -> np.ndarray:
        """Construct a minimal y0 = (t, r, π/2, 0, v^t, v^r, v^θ, v^φ)."""
        f = max(1.0 - 2.0 * m / r, 0.01)
        vt = 1.0 / math.sqrt(f)
        return np.array([0.0, r, math.pi / 2, 0.0, vt, -0.5, 0.0, 0.0], dtype=np.float64)

    def test_no_surrogate_always_integrates(self):
        """With weights=None, all rays must go to RK4 (from_surrogate=False)."""
        gate = RuntimeGate(weights=None, r_threshold=SURROGATE_R_THRESHOLD)
        for r in (5.0, 15.0, 40.0):
            y0 = self._make_y0(r)
            result = gate.infer_or_integrate(y0, m=1.0, dlambda=0.1, max_steps=500, r_escape=80.0)
            assert isinstance(result, SurrogateResult)
            assert result.from_surrogate is False, f"Expected RK4 for r={r}"

    def test_with_surrogate_routes_high_r_to_surrogate(self):
        """With surrogate loaded, rays at r >> threshold use surrogate."""
        weights = init_random_weights(np.random.default_rng(99))
        gate = RuntimeGate(weights=weights, r_threshold=10.0)
        y0_far = self._make_y0(r=40.0)
        result = gate.infer_or_integrate(y0_far, m=1.0, dlambda=0.1, max_steps=500, r_escape=80.0)
        assert result.from_surrogate is True

    def test_with_surrogate_routes_low_r_to_rk4(self):
        """With surrogate loaded, rays at r < threshold use RK4."""
        weights = init_random_weights(np.random.default_rng(99))
        gate = RuntimeGate(weights=weights, r_threshold=10.0)
        y0_close = self._make_y0(r=5.0)
        result = gate.infer_or_integrate(y0_close, m=1.0, dlambda=0.1, max_steps=500, r_escape=80.0)
        assert result.from_surrogate is False

    def test_surrogate_result_has_valid_status(self):
        """Surrogate result status should be in {0,1,2,3}."""
        weights = init_random_weights()
        gate = RuntimeGate(weights=weights, r_threshold=5.0)
        y0 = self._make_y0(r=20.0)
        result = gate.infer_or_integrate(y0, m=1.0, dlambda=0.1, max_steps=500, r_escape=80.0)
        assert result.status in (0, 1, 2, 3)

    def test_rk4_result_has_finite_r_min(self):
        """RK4 result must have finite r_min or NaN (not Inf)."""
        gate = RuntimeGate(weights=None)
        y0 = self._make_y0(r=12.0)
        result = gate.infer_or_integrate(y0, m=1.0, dlambda=0.1, max_steps=1000, r_escape=80.0)
        if not math.isnan(result.r_min):
            assert math.isfinite(result.r_min)

    def test_surrogate_loaded_property(self):
        gate_empty = RuntimeGate(weights=None)
        assert gate_empty.surrogate_loaded is False

        weights = init_random_weights()
        gate_full = RuntimeGate(weights=weights)
        assert gate_full.surrogate_loaded is True
