#pragma once

/* Batched Kerr null-geodesic tracer (Boyer-Lindquist + RK4).
 *
 * Input layout is row-major double[N * 8], matching the Schwarzschild batch
 * API. The kernel converts state and RK4 stages to flat component-major
 * workspace arrays. All rays share m, a, dlambda, max_steps, r_escape,
 * r_horizon_epsilon. */

#ifdef __cplusplus
extern "C" {
#endif

void bh_rt_kerr_phase3_batch_trace(
    const double *y0,
    int n,
    double m,
    double a,
    double dlambda,
    int max_steps,
    double r_escape,
    double r_horizon_epsilon,
    int *out_status,
    int *out_steps_taken,
    double *out_termination_r,
    double *out_r_min,
    double *out_eq_r_cross);

#ifdef __cplusplus
}
#endif
