#pragma once

/* Batched 3D null-geodesic tracer for Schwarzschild (Christoffel + RK4).
 *
 * Memory layout (SoA-friendly, row-major):
 *   y0      — double[N * 8], row i = (t, r, theta, phi, v^t, v^r, v^theta, v^phi)
 *   out_*   — one value per ray (N elements each)
 *
 * Status codes: see bh_rt_status.h (0=captured, 1=escaped, 2=max_steps, 3=numeric)
 * Shared parameters apply to every ray in the batch.
 *
 * On entry, y0 is read-only.  All output arrays must be pre-allocated by caller. */

#ifdef __cplusplus
extern "C" {
#endif

/* Trace N null geodesics in one call.
 *
 * Parameters
 * ----------
 * y0                : double[N * 8]  — initial states, row-major (one state per row)
 * n                 : int            — number of rays in the batch
 * m                 : double         — black hole mass (geometric units)
 * dlambda           : double         — affine parameter step
 * max_steps         : int            — per-ray step budget
 * r_escape          : double         — escape radius
 * r_horizon_epsilon : double         — capture threshold offset above 2M
 *
 * Outputs (all length-N arrays, caller-allocated)
 * -------
 * out_status        : int[N]         — per-ray termination status (BH_RT_STATUS_*)
 * out_steps_taken   : int[N]         — steps consumed by each ray
 * out_termination_r : double[N]      — r coordinate at termination
 * out_r_min         : double[N]      — minimum r reached (NAN if never tracked)
 * out_eq_r_cross    : double[N]      — r at first equatorial (theta=pi/2) crossing, NAN if none */
void bh_rt_schwarzschild_phase2_batch_trace(
    const double *y0,
    int n,
    double m,
    double dlambda,
    int max_steps,
    double r_escape,
    double r_horizon_epsilon,
    int    *out_status,
    int    *out_steps_taken,
    double *out_termination_r,
    double *out_r_min,
    double *out_eq_r_cross);

#ifdef __cplusplus
}
#endif
