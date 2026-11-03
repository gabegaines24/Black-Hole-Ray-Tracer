# T05 — Disk observables and render parity

**Status:** complete.

**Dependencies:** T02, T03, T04.

**Scope:** Define first crossing, interpolate crossing observables, color with crossing velocities, and align Python/CPU/CUDA behavior.

**Acceptance:** Deterministic hit, miss, capture, boundary, and multiple-crossing tests pass with finite clamped colors.

**Evidence:** `src/blackhole_ray_tracer/phase2_disk.py`, kernel batch sources, `tests/test_phase2.py`, ADR-003.
