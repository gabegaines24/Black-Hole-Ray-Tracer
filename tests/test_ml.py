"""Smoke tests for the Phase 3 ML scaffold."""

from __future__ import annotations

import numpy as np

from ml.dataset import generate_dataset
from ml.runtime_gate import RuntimeGate
from ml.schema import N_INPUTS, N_OUTPUTS, normalize_inputs


def test_normalize_inputs_shape_and_finite_values() -> None:
    x = np.array([[30.0, np.pi / 2, np.pi, -0.5, 0.01, 0.02]], dtype=np.float32)
    z = normalize_inputs(x)
    assert z.shape == (1, N_INPUTS)
    assert np.all(np.isfinite(z))
    assert z[0, 0] == np.float32(1.0)
    assert abs(float(z[0, 1])) < 1e-7


def test_generate_dataset_tiny_smoke() -> None:
    x, y = generate_dataset(
        2,
        dlambda=0.2,
        max_steps=50,
        r_escape=60.0,
        seed=123,
        verbose=False,
    )
    assert x.shape == (2, N_INPUTS)
    assert y.shape == (2, N_OUTPUTS)
    assert x.dtype == np.float32
    assert y.dtype == np.float32
    assert np.all(np.isfinite(x))


def test_runtime_gate_falls_back_without_model() -> None:
    gate = RuntimeGate(weights=None)
    y0 = np.array([0.0, 12.0, np.pi / 2, 0.0, 1.1, -0.8, 0.0, 0.0], dtype=float)
    result = gate.infer_or_integrate(
        y0,
        m=1.0,
        dlambda=0.2,
        max_steps=50,
        r_escape=60.0,
    )
    assert not result.from_surrogate
    assert result.steps_taken >= 0
