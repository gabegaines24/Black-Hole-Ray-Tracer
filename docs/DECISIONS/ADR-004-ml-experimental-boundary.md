# ADR-004: ML experimental boundary

## Context

A surrogate can accelerate classification but cannot silently replace trusted geodesic integration.
Unversioned or malformed artifacts can produce plausible but invalid states.

## Decision

Keep ML opt-in through the Phase 2 render gate.
Require versioned dataset/model metadata, exact tensor shapes, float32 finite weights, and bounded predictions.
Fall back to RK4 for missing, corrupt, incompatible, or invalid artifacts.
Package one deterministic reference model, but make no physical-accuracy claim.

## Consequences

- Users retain a trusted path without model installation.
- Artifact format changes fail closed instead of loading stale arrays.
- Accuracy remains an experiment and requires independent qualification.

## Evidence

- `ml/schema.py`
- `ml/dataset.py`
- `ml/surrogate.py`
- `ml/runtime_gate.py`
- `tests/test_ml.py`
- `ml/models/README.md`
