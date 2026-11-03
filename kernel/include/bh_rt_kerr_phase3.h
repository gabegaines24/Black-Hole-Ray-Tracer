#pragma once

#include <stdint.h>
#include "bh_rt_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BH_RT_KERR_OK 0
#define BH_RT_KERR_INVALID_ARGUMENT 1
#define BH_RT_KERR_DOMAIN_ERROR 2
#define BH_RT_KERR_NUMERICAL_ERROR 3

/*
 * Kerr Boyer–Lindquist CPU contract.
 * Coordinates: (t, r, theta, phi), signature (-+++).
 * State: (t, r, theta, phi, vt, vr, vtheta, vphi).
 * spin is dimensionless a/M. Metric output is row-major g[mu*4 + nu].
 * Metric/RHS functions return zero on success and leave output invalid on error.
 */
int32_t bh_rt_kerr_metric(double r, double theta, double m, double spin,
                          double *g_out);

int32_t bh_rt_kerr_geodesic_rhs(const double *y, double m, double spin,
                                double *dy_out);

/*
 * Build one ZAMO camera state:
 * y_out = (t, r, theta, phi, vt, vr, vtheta, vphi).
 * sx and sy are normalized screen coordinates before FOV/aspect scaling.
 */
int32_t bh_rt_kerr_zamo_initial_state(
    double m, double spin, double r, double theta, double phi,
    double fov_deg, int32_t width, int32_t height, double sx, double sy,
    double *y_out);

typedef struct bh_rt_kerr_trace_result {
  int32_t status;
  int32_t steps_taken;
  int32_t max_steps;
  double termination_r;
  double termination_lambda;
  double r_min;
} bh_rt_kerr_trace_result;

typedef struct bh_rt_kerr_equatorial_crossing {
  int32_t hit;
  double r_cross;
  double vt_cross;
  double vphi_cross;
} bh_rt_kerr_equatorial_crossing;

typedef struct bh_rt_kerr_disk_result {
  int32_t hit;
  double z_factor;
} bh_rt_kerr_disk_result;

/*
 * Apply disk radius and Doppler policy to one equatorial crossing.
 * NaN or out-of-range r_cross returns hit == 0, z_factor == 1.
 */
int32_t bh_rt_kerr_disk_observable(
    double r_cross, double r_inner, double r_outer, double m,
    double vt, double vphi, double redshift_scale,
    bh_rt_kerr_disk_result *out);

/*
 * Trace one ray with RK4 and periodic null-cone projection.
 * The trace has no trajectory buffers, disk crossing, camera, bridge, or CUDA
 * behavior. Valid inputs return zero; result.status carries terminal state.
 */
int32_t bh_rt_kerr_phase3_trace(
    const double *y0, double m, double spin, double dlambda,
    int32_t max_steps, double r_escape, double r_horizon_epsilon,
    bh_rt_kerr_trace_result *out);

/*
 * Trace one ray and record the first strict theta = pi/2 crossing.
 * Crossing fields remain NaN when hit is zero.
 */
int32_t bh_rt_kerr_phase3_trace_crossing(
    const double *y0, double m, double spin, double dlambda,
    int32_t max_steps, double r_escape, double r_horizon_epsilon,
    bh_rt_kerr_trace_result *trace_out,
    bh_rt_kerr_equatorial_crossing *crossing_out);

#ifdef __cplusplus
}
#endif
