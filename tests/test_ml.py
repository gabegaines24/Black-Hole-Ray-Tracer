"""Smoke tests for the Phase 3 ML scaffold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.dataset import (
    dataset_metadata,
    generate_dataset,
    load_dataset,
    load_dataset_metadata,
    save_dataset,
)
from ml.runtime_gate import RuntimeGate
from ml.schema import (
    N_INPUTS,
    N_OUTPUTS,
    STATUS_CAPTURED,
    STATUS_ESCAPED,
    STATUS_MAX_STEPS,
    STATUS_NUMERIC,
    denormalize_outputs,
    normalize_inputs,
    normalize_outputs,
)
from ml.surrogate import (
    init_random_weights,
    load_surrogate,
    predict_batch,
    save_surrogate,
    train,
)


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


def test_runtime_gate_loads_reference_model() -> None:
    model_path = Path(__file__).resolve().parents[1] / "ml" / "models" / "surrogate.npz"
    if not model_path.exists():
        pytest.skip("reference surrogate not generated")
    gate = RuntimeGate.from_model_path(model_path)
    assert gate.surrogate_loaded


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


def test_dataset_metadata_round_trip(tmp_path: Path) -> None:
    x, y = generate_dataset(
        2,
        dlambda=0.2,
        max_steps=50,
        r_escape=60.0,
        seed=123,
        verbose=False,
    )
    metadata = dataset_metadata(
        m=1.0,
        dlambda=0.2,
        max_steps=50,
        r_escape=60.0,
        seed=123,
    )
    path = tmp_path / "dataset.npz"
    save_dataset(x, y, path, metadata=metadata)
    loaded_x, loaded_y = load_dataset(path)
    loaded_metadata = load_dataset_metadata(path)
    np.testing.assert_array_equal(loaded_x, x)
    np.testing.assert_array_equal(loaded_y, y)
    assert loaded_metadata["integrator"] == metadata["integrator"]
    assert loaded_metadata["sentinel_policy"] == metadata["sentinel_policy"]


def test_dataset_loader_rejects_missing_metadata(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        X=np.zeros((1, N_INPUTS), dtype=np.float32),
        Y=np.zeros((1, N_OUTPUTS), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="metadata"):
        load_dataset(path)


def test_model_loader_rejects_missing_metadata_and_bad_shape(tmp_path: Path) -> None:
    weights = init_random_weights(np.random.default_rng(0))
    legacy_path = tmp_path / "legacy-model.npz"
    np.savez(legacy_path, **weights._asdict())
    with pytest.raises(ValueError, match="metadata"):
        load_surrogate(legacy_path)

    valid_path = tmp_path / "valid-model.npz"
    save_surrogate(weights, valid_path)
    with np.load(valid_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["W1"] = np.zeros((1, 1), dtype=np.float32)
    bad_path = tmp_path / "bad-shape-model.npz"
    np.savez(bad_path, **arrays)
    with pytest.raises(ValueError, match="W1"):
        load_surrogate(bad_path)

    corrupt_path = tmp_path / "corrupt-model.npz"
    corrupt_path.write_bytes(b"not an NPZ")
    assert not RuntimeGate.from_model_path(corrupt_path).surrogate_loaded


def test_runtime_gate_uses_requested_steps_and_falls_back_invalid_prediction() -> None:
    weights = init_random_weights(np.random.default_rng(0))
    for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
        getattr(weights, name).fill(0.0)
    weights.b3[:] = (1.0, 0.0, 0.0, 0.5)
    y0 = np.array([0.0, 25.0, np.pi / 2, 0.0, 1.05, -0.5, 0.0, 0.01])
    result = RuntimeGate(weights=weights).infer_or_integrate(
        y0, m=1.0, dlambda=0.2, max_steps=200, r_escape=60.0
    )
    assert result.from_surrogate
    assert result.status == STATUS_ESCAPED
    assert result.steps_taken == 100

    invalid_weights = init_random_weights(np.random.default_rng(0))
    for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
        getattr(invalid_weights, name).fill(0.0)
    invalid_weights.b3[0] = 1.0
    invalid_weights.b3[1] = -0.1
    fallback = RuntimeGate(weights=invalid_weights).infer_or_integrate(
        y0, m=1.0, dlambda=0.2, max_steps=2, r_escape=60.0
    )
    assert not fallback.from_surrogate
    assert fallback.steps_taken == 2


def test_surrogate_train_and_gate_accuracy_smoke() -> None:

    x, y = generate_dataset(
        40,
        dlambda=0.2,
        max_steps=200,
        r_escape=60.0,
        seed=7,
        verbose=False,
    )
    x_train, y_train = x[:30], y[:30]
    x_hold, y_hold = x[30:], y[30:]
    x_norm = normalize_inputs(x_train)
    y_norm = normalize_outputs(y_train, max_steps=200.0)
    weights = train(
        init_random_weights(np.random.default_rng(0)), x_norm, y_norm, epochs=40, verbose=False
    )
    pred = denormalize_outputs(
        predict_batch(weights, normalize_inputs(x_hold)), max_steps=200.0
    )

    status_hits = 0
    for i in range(len(x_hold)):
        pred_status = int(round(float(pred[i, 0])))
        true_status = int(round(float(y_hold[i, 0])))
        if pred_status == true_status:
            status_hits += 1
    status_accuracy = status_hits / max(len(x_hold), 1)
    assert status_accuracy >= 0.3
    assert np.all(np.isfinite(pred))

    gate = RuntimeGate(weights=weights, r_threshold=10.0)
    y0 = np.array([0.0, 25.0, np.pi / 2, 0.0, 1.05, -0.5, 0.0, 0.01], dtype=float)
    gate_result = gate.infer_or_integrate(
        y0, m=1.0, dlambda=0.2, max_steps=200, r_escape=60.0
    )
    assert gate_result.from_surrogate
    assert gate_result.status in (
        STATUS_CAPTURED,
        STATUS_ESCAPED,
        STATUS_MAX_STEPS,
        STATUS_NUMERIC,
    )
    assert gate_result.steps_taken >= 0
