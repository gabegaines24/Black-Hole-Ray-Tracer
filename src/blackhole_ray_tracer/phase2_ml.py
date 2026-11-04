"""Phase 2 ML surrogate batch dispatch via ``ml.runtime_gate.RuntimeGate``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .phase2_types import Phase2RenderConfig

_DEFAULT_MODEL = Path("ml/models/surrogate.npz")


def make_runtime_gate(cfg: Phase2RenderConfig):
    """Build a ``RuntimeGate`` from render config (loads weights when configured)."""
    from ml.runtime_gate import RuntimeGate

    model_path = cfg.ml_model_path or cfg.ml_gate_model_path
    if model_path is not None:
        return RuntimeGate.from_model_path(model_path, r_threshold=cfg.ml_r_threshold)
    if _DEFAULT_MODEL.is_file():
        return RuntimeGate.from_model_path(_DEFAULT_MODEL, r_threshold=cfg.ml_r_threshold)
    return RuntimeGate(weights=None, r_threshold=cfg.ml_r_threshold)


def ml_surrogate_available(cfg: Phase2RenderConfig) -> bool:
    """Return True when a surrogate model can be loaded for this config."""
    model_path = cfg.ml_model_path or cfg.ml_gate_model_path
    if model_path is not None:
        return Path(model_path).is_file()
    return _DEFAULT_MODEL.is_file()


def schwarzschild_phase2_batch_ml(
    y0_batch: np.ndarray,
    cfg: Phase2RenderConfig,
    *,
    gate=None,
) -> dict[str, Any]:
    """Trace a pixel batch with ``RuntimeGate`` (surrogate for far-field, RK4 near horizon)."""
    if gate is None:
        gate = make_runtime_gate(cfg)
    if not gate.surrogate_loaded:
        raise RuntimeError(
            "ML surrogate model not found. Train with "
            "`uv run python -m ml.dataset --n-rays 5000` then "
            "`uv run python -m ml.surrogate train`, or pass --ml-model-path."
        )
    return gate.infer_or_integrate_batch(
        y0_batch,
        m=cfg.m,
        dlambda=cfg.dlambda,
        max_steps=cfg.max_steps,
        r_escape=cfg.r_escape,
        r_horizon_epsilon=cfg.r_horizon_epsilon,
    )
