# Ray batch API (`kernel/` + `bridge/` + optional CUDA)

This file freezes the Phase 2 FFI contract used by C headers, PyBind11 bindings, and the optional CUDA launcher.
Align all Python façades and C implementations with names and types here.

---

## Termination (`RayStatus`)

Defined in [`kernel/include/bh_rt_status.h`](../kernel/include/bh_rt_status.h).
Matches `phase1.RayStatus` enum values exactly:

| C constant                     | Integer | Python `RayStatus` value  | Meaning |
|--------------------------------|---------|---------------------------|---------|
| `BH_RT_STATUS_CAPTURED`        | `0`     | `"captured"`              | r fell inside capture threshold (\( \lesssim 2M + \epsilon \)) |
| `BH_RT_STATUS_ESCAPED`         | `1`     | `"escaped"`               | r exceeded escape threshold |
| `BH_RT_STATUS_MAX_STEPS`       | `2`     | `"max_steps"`             | Step budget exhausted |
| `BH_RT_STATUS_NUMERICAL_ERROR` | `3`     | `"numerical_error"`       | NaN / non-finite state |

---

## Single-ray API (`bh_rt_schwarzschild_phase2_trace`)

Header: [`kernel/include/bh_rt_schwarzschild_phase2.h`](../kernel/include/bh_rt_schwarzschild_phase2.h)

```c
#include <stdint.h>
typedef struct bh_rt_phase2_trace_result {
  int32_t status;
  int32_t steps_taken;
  int32_t max_steps;
  double termination_r;
  double termination_lambda;
  double r_min;
} bh_rt_phase2_trace_result;

void bh_rt_schwarzschild_phase2_trace(
    const double *y0,        /* double[8]: t, r, theta, phi, vt, vr, vth, vph */
    double m, double dlambda, int32_t max_steps,
    double r_escape, double r_horizon_epsilon,
    bh_rt_phase2_trace_result *out);
```

---

## Batch API (`bh_rt_schwarzschild_phase2_batch_trace`)

Header: [`kernel/include/bh_rt_schwarzschild_phase2_batch.h`](../kernel/include/bh_rt_schwarzschild_phase2_batch.h)

### Input layout

`y0` is `double[N * 8]`, **row-major**: row `i` = `y0[i*8 .. i*8+7]`.

```
y0[i*8 + 0] = t
y0[i*8 + 1] = r
y0[i*8 + 2] = theta
y0[i*8 + 3] = phi
y0[i*8 + 4] = v^t
y0[i*8 + 5] = v^r
y0[i*8 + 6] = v^theta
y0[i*8 + 7] = v^phi
```

### Output arrays (all length N, caller-allocated)

| `out_status[i]`          | `int32_t` | `int32`     | `BH_RT_STATUS_*` code |
| `out_steps_taken[i]`     | `int32_t` | `int32`     | Steps consumed |
| `out_termination_r[i]`   | `double`  | `float64`   | r at termination |
| `out_r_min[i]`           | `double`  | `float64`   | Min r reached (NaN if none) |
| `out_disk_hit[i]`        | `int32_t` | `int32`     | 1 if first equatorial (θ=π/2) crossing detected |
| `out_disk_r_cross[i]`    | `double`  | `float64`   | Interpolated r at crossing (NaN if no hit) |
| `out_disk_vt[i]`         | `double`  | `float64`   | v^t at crossing (NaN if no hit) |
| `out_disk_vph[i]`        | `double`  | `float64`   | v^φ at crossing (NaN if no hit) |

### Shared scalars

All rays in a batch share: `m`, `dlambda`, `max_steps`, `r_escape`, `r_horizon_epsilon`.

The CPU bridge accepts empty batches and bounds `N` at `1_000_000` rays.
It rejects larger batches before native allocation. Python façades normalize
accepted inputs to contiguous `float64` storage.

### Disk crossing semantics

The tracer records only the first strict sign-changing crossing of `theta = pi / 2`.
It linearly interpolates `r`, `v^t`, and `v^phi` between pre-step and post-step states.
Capture before a crossing produces no disk hit. Disk color uses interpolated crossing velocities.

---

### Python bridge shape convention

```python
y0_batch: np.ndarray  # shape (N, 8), dtype float64, C-contiguous
out_status: np.ndarray        # shape (N,), dtype int32
out_steps_taken: np.ndarray   # shape (N,), dtype int32
out_termination_r: np.ndarray # shape (N,), dtype float64
out_r_min: np.ndarray         # shape (N,), dtype float64
out_disk_hit: np.ndarray      # shape (N,), dtype int32
out_disk_r_cross: np.ndarray  # shape (N,), dtype float64
out_disk_vt: np.ndarray       # shape (N,), dtype float64
out_disk_vph: np.ndarray      # shape (N,), dtype float64
```

---

## Optional CUDA launcher

Symbol: `bh_rt_schwarzschild_phase2_batch_trace_cuda` in
[`kernel/cuda/bh_rt_schwarzschild_phase2_batch.cu`](../kernel/cuda/bh_rt_schwarzschild_phase2_batch.cu).

The launcher uses the batch layouts above and returns `int32_t`.
Return `0` means `cudaSuccess`; any nonzero CUDA error code means no valid batch result.
The Python ctypes façade raises `RuntimeError` for nonzero return codes.
CUDA availability is optional and requires a separately built shared library.
The host wrapper checks allocation, copies, kernel launch, synchronization, and cleanup.

---

## Kerr CPU API

Header: [`kernel/include/bh_rt_kerr_phase3.h`](../kernel/include/bh_rt_kerr_phase3.h).

`bh_rt_kerr_zamo_initial_state` writes one complete `double[8]` state using the validated
ZAMO/local-orthonormal camera mapping:

```c
int32_t bh_rt_kerr_zamo_initial_state(
    double m, double spin, double r, double theta, double phi,
    double fov_deg, int32_t width, int32_t height,
    double sx, double sy, double *y_out);
```

`bh_rt_kerr_metric` writes row-major covariant `double[16]`.
`bh_rt_kerr_geodesic_rhs` reads and writes the eight-state `double[8]` layout.
Both algebra functions return `0` on success and a nonzero `BH_RT_KERR_*` code on invalid output.

`bh_rt_kerr_phase3_trace` adds a bounded CPU single-ray RK4 path:

```c
typedef struct bh_rt_kerr_trace_result {
  int32_t status;              /* BH_RT_STATUS_* */
  int32_t steps_taken;
  int32_t max_steps;
  double termination_r;
  double termination_lambda;
  double r_min;
} bh_rt_kerr_trace_result;
```

`bh_rt_kerr_phase3_trace_crossing` keeps the trace result unchanged and writes:

```c
typedef struct bh_rt_kerr_equatorial_crossing {
  int32_t hit;
  double r_cross;
  double vt_cross;
  double vphi_cross;
} bh_rt_kerr_equatorial_crossing;
```

It reports only the first strict sign-changing `theta = pi/2` crossing.
Crossing fields are NaN when `hit == 0`; it does not apply disk radius or color policy.

`bh_rt_kerr_disk_observable` applies geometry-independent disk policy:

```c
typedef struct bh_rt_kerr_disk_result {
  int32_t hit;
  double z_factor;
} bh_rt_kerr_disk_result;

int32_t bh_rt_kerr_disk_observable(
    double r_cross, double r_inner, double r_outer, double m,
    double vt, double vphi, double redshift_scale,
    bh_rt_kerr_disk_result *out);
```

Bounds are inclusive. NaN `r_cross` returns no hit with `z_factor == 1`.
The caller supplies disk bounds and redshift scale; C does not calculate ISCO, color, or rendering.

The trace accepts one `double[8]` state, returns shared `BH_RT_STATUS_*` values,
and uses the same pre-step capture/escape and fourth-step null projection semantics as Python.
It has no disk radius policy, color, batch, bridge, or CUDA behavior.
---

## Phase 1 SoA columns (2D equatorial batches — future)

| Array       | Dtype     | Meaning |
|-------------|-----------|---------|
| `impact_b`  | `float64` | Impact parameter b |
| `r_min`     | `float64` | Min r along trajectory |
| `phi_last`  | `float64` | φ at termination |
| `steps_taken` | `int32` | Integrator iterations |
| `status`    | `int32`   | `BH_RT_STATUS_*` code |

**Rule:** Do not embed per-ray structs in hot loops; use flat SoA arrays.
