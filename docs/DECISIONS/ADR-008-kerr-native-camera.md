# ADR-008: Kerr native ZAMO camera initializer

## Context

T12 qualified a bounded Kerr C single-ray tracer, but callers still needed Python to construct
ZAMO initial states. Directly porting a renderer would mix camera, disk, batch, and bridge concerns.

## Decision

Add one C camera initializer, `bh_rt_kerr_zamo_initial_state`.
Keep the Python camera as the reference oracle and preserve its Boyer–Lindquist state order,
normalized screen mapping, metric-derived lapse, frame dragging, and spatial scale factors.
Return a complete caller-owned `double[8]` state that feeds the T12 tracer.

Reject invalid camera domains explicitly. Keep disk, trajectory samples, batch memory ownership,
PyBind11, CUDA, and image rendering out of scope.

## Consequences

- C can reproduce validated Kerr observer seeds before native rendering work begins.
- Camera parity is separate from disk and render parity.
- The C initializer has no image or allocation behavior.
- Future batch or bridge work must define ownership and error propagation separately.

## Evidence

- `kernel/include/bh_rt_kerr_phase3.h`
- `kernel/src/bh_rt_kerr_phase3.c`
- `tests/test_kernel_phase3_camera.py`
