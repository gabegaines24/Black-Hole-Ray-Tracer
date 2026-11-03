# ADR-007: Bounded Kerr native single-ray tracer

## Context

T10 established C parity for Kerr Boyer–Lindquist metric and first-order RHS algebra.
A complete native Kerr renderer would still combine camera initialization, trajectory storage,
disk observables, bridge behavior, and CUDA execution.

## Decision

Add only a CPU C single-ray tracer.
Reuse the T10 metric and RHS implementation, then mirror Python RK4 termination semantics:
pre-step capture and escape checks, periodic null-cone projection, and scalar result fields.
Keep trajectory samples, camera construction, disk crossing, rendering, bridge, and CUDA out of scope.

Use `bh_rt_kerr_trace_result` with the shared `BH_RT_STATUS_*` integer values.
Return `BH_RT_KERR_OK` for valid control inputs, including an initial non-finite state whose
result status is `BH_RT_STATUS_NUMERICAL_ERROR`; return explicit Kerr errors for invalid controls.

## Consequences

- C trajectory termination can be qualified before adding camera or rendering ABI.
- The result has no disk or sample observables, so it cannot replace Python Phase 3 rendering.
- The tracer evaluates valid algebra at RK4 trial states inside the horizon, then terminates on
  the next pre-step capture check; this preserves bounded Python reference behavior near capture.
- A future bridge must add a separate ownership and array contract.
- CUDA remains a later backend after CPU camera, disk, and batch parity decisions.

## Evidence

- `kernel/include/bh_rt_kerr_phase3.h`
- `kernel/src/bh_rt_kerr_phase3.c`
- `tests/test_kernel_phase3_trace.py`
