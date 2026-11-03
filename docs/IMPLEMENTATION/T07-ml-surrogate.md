# T07 — Experimental ML contract

**Status:** complete; experimental.

**Dependencies:** T02, T06.

**Scope:** Version dataset/model metadata, validate artifacts, test deterministic holdout behavior, package a reference model, and retain RK4 fallback.

**Acceptance:** Missing, corrupt, incompatible, non-finite, and out-of-bounds artifacts fall back explicitly.

**Evidence:** `ml/`, `tests/test_ml.py`, `ml/models/README.md`, ADR-004.
