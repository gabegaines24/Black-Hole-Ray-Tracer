# T04 — Safe native bridge and parity

**Status:** complete.

**Dependencies:** T03.

**Scope:** Validate PyBind11 and Python façade shapes, dtypes, contiguity, batch limits, output allocation, and GIL behavior.

**Acceptance:** Scalar and batch native paths reject malformed inputs and match Python statuses and observables.

**Evidence:** `bridge/module_phase2.cpp`, `src/blackhole_ray_tracer/native_phase2.py`, `tests/test_bridge_native_phase2.py`.
