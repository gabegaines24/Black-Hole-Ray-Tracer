# Ray batch API (`kernel/` + `bridge/`)

This file freezes the FFI contract used by C headers and PyBind11 bindings.
Align all Python façades and C implementations with the names and types here.

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
typedef struct bh_rt_phase2_trace_result {
  int    status;
  int    steps_taken;
  int    max_steps;
  double termination_r;
  double termination_lambda;
  double r_min;
} bh_rt_phase2_trace_result;

void bh_rt_schwarzschild_phase2_trace(
    const double *y0,        /* double[8]: t, r, theta, phi, vt, vr, vth, vph */
    double m, double dlambda, int max_steps,
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

| Array name           | C type     | NumPy dtype | Meaning |
|----------------------|------------|-------------|---------|
| `out_status[i]`        | `int`      | `int32`     | `BH_RT_STATUS_*` code |
| `out_steps_taken[i]`   | `int`      | `int32`     | Steps consumed |
| `out_termination_r[i]` | `double`   | `float64`   | r at termination |
| `out_r_min[i]`         | `double`   | `float64`   | Min r reached (NaN if none) |

### Shared scalars

All rays in a batch share: `m`, `dlambda`, `max_steps`, `r_escape`, `r_horizon_epsilon`.

### Python bridge shape convention

```python
y0_batch: np.ndarray  # shape (N, 8), dtype float64, C-contiguous
out_status: np.ndarray        # shape (N,), dtype int32
out_steps_taken: np.ndarray   # shape (N,), dtype int32
out_termination_r: np.ndarray # shape (N,), dtype float64
out_r_min: np.ndarray         # shape (N,), dtype float64
```

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
