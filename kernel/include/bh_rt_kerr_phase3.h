#pragma once

/* Kerr null geodesics in Boyer-Lindquist coordinates (Christoffel + RK4).
 * State y = (t, r, theta, phi, v^t, v^r, v^theta, v^phi) affine in lambda.
 *
 * Mirrors phase3_geodesic.trace_kerr_null_geodesic with store_samples == 0.
 * Status codes align with docs/STATE_API.md / bh_rt_status.h. */

#include "bh_rt_schwarzschild_phase2.h"

#ifdef __cplusplus
extern "C" {
#endif

void bh_rt_kerr_phase3_trace(const double *y0, double m, double a, double dlambda,
                             int max_steps, double r_escape, double r_horizon_epsilon,
                             bh_rt_phase2_trace_result *out);

#ifdef __cplusplus
}
#endif
