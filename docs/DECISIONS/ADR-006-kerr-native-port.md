# ADR-006: Incremental Kerr native port

## Context

Phase 3 Python validation passed, but a complete Kerr C/CUDA renderer would combine metric algebra,
state termination, camera initialization, disk observables, and new FFI behavior in one risky change.

## Decision

Port Kerr incrementally. First expose only a C probe for the BL covariant metric and first-order
geodesic RHS. Keep Python as the reference oracle. Require ctypes parity at exterior off-pole points
before adding a C tracer or bridge.

Use the existing eight-state order and dimensionless `spin = a/M` convention.
Return explicit integer domain/error codes. Do not silently emit zero matrices or derivatives for
invalid inputs.

## Consequences

- Algebra parity becomes testable before adding integration state and termination semantics.
- The probe has no user-facing renderer and cannot claim Kerr acceleration.
- Future tracer work must reuse this contract and add separate termination/disk/camera acceptance.
- CUDA remains a later backend after CPU C parity.

## Evidence

- `docs/IMPLEMENTATION/T10-kerr-native.md`
- `src/blackhole_ray_tracer/phase3_christoffel.py`
- `tests/test_kernel_phase3_algebra.py`
