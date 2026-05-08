#pragma once

/* 3D null geodesics in Schwarzschild (Schwarzschild coordinates, Christoffel form).
 * State y = (t, r, theta, phi, v^t, v^r, v^theta, v^phi) affine in lambda.
 *
 * Mirrors phase2_geodesic.trace_null_geodesic_3d with store_samples == 0 —
 * parity vs Python regression tests only (no trajectory buffers here).
 *
 * Status codes align with docs/STATE_API.md / bh_rt_schwarzschild_u.h. */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bh_rt_phase2_trace_result {
  int status;
  int steps_taken;
  int max_steps;
  double termination_r;
  double termination_lambda;
  double r_min;
} bh_rt_phase2_trace_result;

/* Integrates from y0[8]; applies the same vr renormalization as Python before
 * the loop and again every fourth completed RK4 step. */
void bh_rt_schwarzschild_phase2_trace(const double *y0, double m, double dlambda,
                                    int max_steps, double r_escape,
                                    double r_horizon_epsilon,
                                    bh_rt_phase2_trace_result *out);

#ifdef __cplusplus
}
#endif
