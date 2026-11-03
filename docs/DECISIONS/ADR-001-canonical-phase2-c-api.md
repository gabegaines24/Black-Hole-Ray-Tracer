# ADR-001: Canonical Phase 2 C API

## Context

The project had overlapping 3D Schwarzschild C paths with divergent status and error behavior.
Python, native CPU, and future CUDA paths need one state and batch contract.

## Decision

Use `kernel/src/bh_rt_schwarzschild_phase2.c` and
`bh_rt_schwarzschild_phase2_batch.c` as the canonical Phase 2 implementations.
Use `bh_rt_status.h` for integer statuses and `docs/STATE_API.md` for array layout.
Retire duplicate 3D tracer files instead of preserving aliases.

## Consequences

- Native and Python parity tests target one C implementation.
- New backends must match status, termination, minimum-radius, and disk columns.
- Experimental or future coordinate systems require separate phase APIs.

## Evidence

- `tests/test_kernel_phase2_parity.py`
- `tests/test_bridge_native_phase2.py`
- `tests/test_phase2.py`
- `docs/STATE_API.md`
