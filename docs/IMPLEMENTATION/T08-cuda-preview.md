# T08 — CUDA, preview, and benchmark qualification

**Status:** complete; CUDA runtime unavailable locally.

**Dependencies:** T04, T05.

**Scope:** Harden CUDA host errors and ABI, route preview backends, add boundary tests, and record a measured CPU benchmark.

**Acceptance:** CUDA unavailable diagnostics are explicit; preview routing works headlessly; CUDA claims remain conditional.

**Evidence:** `kernel/cuda/`, `src/blackhole_ray_tracer/native_phase2_cuda.py`, `src/blackhole_ray_tracer/preview.py`, `tests/test_cuda_batch_parity.py`, `tests/test_preview.py`, ADR-005.
