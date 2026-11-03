#pragma once

#include <stdint.h>

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
 * n                 : int32_t        — number of rays in the batch
 * m                 : double         — black hole mass (geometric units)
 * dlambda           : double         — affine parameter step
 * max_steps         : int32_t        — per-ray step budget
 * r_escape          : double         — escape radius
 * r_horizon_epsilon : double         — capture threshold offset above 2M
 *
 * out_status        : int32_t[N]     — per-ray termination status (BH_RT_STATUS_*)
 * out_steps_taken   : int32_t[N]     — steps consumed by each ray
 * out_termination_r : double[N]      — r coordinate at termination
 * out_r_min         : double[N]      — minimum r reached (NAN if never tracked)
 * out_disk_hit      : int32_t[N]     — 1 if first equatorial (θ=π/2) crossing detected
 * out_disk_r_cross  : double[N]      — interpolated r at crossing (NAN if no hit)
 * out_disk_vt       : double[N]      — v^t at crossing (NAN if no hit)
 * out_disk_vph      : double[N]      — v^φ at crossing (NAN if no hit) */
void bh_rt_schwarzschild_phase2_batch_trace(
    const double *y0,
    int32_t n,
    double m,
    double dlambda,
    int32_t max_steps,
    double r_escape,
    double r_horizon_epsilon,
    int32_t    *out_status,
    int32_t    *out_steps_taken,
    double *out_termination_r,
    double *out_r_min,
    int32_t    *out_disk_hit,
    double *out_disk_r_cross,
    double *out_disk_vt,
    double *out_disk_vph);

#ifdef __cplusplus
}
#endif
