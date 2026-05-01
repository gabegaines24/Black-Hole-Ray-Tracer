# Ray batch API (draft for `kernel/` + `bridge/`)

This freezes **intent** before C/pybind glue lands. Align Python `phase1` / future `phase2` batch paths with numeric codes used in C headers later.

## Termination (`RayStatus`)

Same semantics as `phase1.RayStatus` (`src/blackhole_ray_tracer/phase1.py`). Proposed integers for FFI (TBD once header exists):

| Name | Meaning |
|------|--------|
| `0` captured | Radius inside capture threshold (\( \lesssim 2M \)). |
| `1` escaped | Ray crossed large-\(r\) escape threshold / outward infinity per integrator. |
| `2` max_steps | Step budget exhausted. |
| `3` numerical_error | NaN/non-finite trajectory. |

## SoA arrays (planned batch outputs)

Naming follows README “SoA-friendly” convention:

| Array | Dtype | Meaning |
|-------|-------|--------|
| `impact_b[]` | `float64` or `double` | Per-ray impact \(b\) (Phase 1 2D) or undefined for generic 3D until camera API is ported. |
| `r_min[]` | | Minimum \(r\) along stored samples (or sentinel if none). |
| `phi_last[]` | | Affine or \(\phi\) termination coordinate (experiment-specific). |
| `steps_taken[]` | `int32`/`int64` | Integrator iterations. |
| `status[]` | `uint8`/enum | Rows above. |

**Rule:** Extend this table rather than embedding per-ray structs in hot loops until the ABI is finalized.
