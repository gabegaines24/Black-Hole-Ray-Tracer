#include "bh_rt_schwarzschild_phase2_batch.h"
#include "bh_rt_schwarzschild_phase2_internal.h"
#include "bh_rt_status.h"

#include <math.h>
#include <string.h>

#ifndef INFINITY
#define INFINITY (1.0 / 0.0)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void bh_rt_p2_try_disk_cross(
    int *found,
    double *r_cross,
    double *vt_cross,
    double *vph_cross,
    double th0,
    double r0,
    double vt0,
    double vph0,
    double th1,
    double r1,
    double vt1,
    double vph1)
{
  if (*found)
    return;
  const double half_pi = 0.5 * M_PI;
  double d0 = th0 - half_pi;
  double d1 = th1 - half_pi;
  if (d0 * d1 >= 0.0)
    return;
  double alpha = fabs(d0) / (fabs(d0) + fabs(d1) + 1e-30);
  *found = 1;
  *r_cross = r0 * (1.0 - alpha) + r1 * alpha;
  *vt_cross = vt0 * (1.0 - alpha) + vt1 * alpha;
  *vph_cross = vph0 * (1.0 - alpha) + vph1 * alpha;
}

void bh_rt_schwarzschild_phase2_batch_trace(
    const double *y0,
    int32_t n,
    double m,
    double dlambda,
    int32_t max_steps,
    double r_escape,
    double r_horizon_epsilon,
    int32_t *out_status,
    int32_t *out_steps_taken,
    double *out_termination_r,
    double *out_r_min,
    int32_t *out_disk_hit,
    double *out_disk_r_cross,
    double *out_disk_vt,
    double *out_disk_vph) {
  double r_cap = 2.0 * m + r_horizon_epsilon;
  bh_rt_p2_userdata ud;
  ud.m = m;
  double rk_ws[BH_RT_P2_RK_WORK];

  for (int32_t ray = 0; ray < n; ++ray) {
    double y[BH_RT_P2_N_STATE];
    const double *src = y0 + ray * BH_RT_P2_N_STATE;
    for (int k = 0; k < BH_RT_P2_N_STATE; ++k)
      y[k] = src[k];

    bh_rt_p2_renormalize_vr(y, m);

    double r_min_val = INFINITY;
    double lam = 0.0;
    int status = BH_RT_STATUS_MAX_STEPS;
    double termination_r = NAN;
    int steps_taken = 0;
    int broke = 0;

    int disk_found = 0;
    double disk_r_cross = NAN;
    double disk_vt = NAN;
    double disk_vph = NAN;

    for (int32_t step_idx = 0; step_idx < max_steps; ++step_idx) {
      double r = y[1];
      if (!bh_rt_p2_all_finite(y)) {
        status = BH_RT_STATUS_NUMERICAL_ERROR;
        termination_r = NAN;
        steps_taken = step_idx;
        broke = 1;
        break;
      }
      if (r < r_cap) {
        status = BH_RT_STATUS_CAPTURED;
        termination_r = r;
        steps_taken = step_idx;
        broke = 1;
        break;
      }
      if (r > r_escape) {
        status = BH_RT_STATUS_ESCAPED;
        termination_r = r;
        steps_taken = step_idx;
        broke = 1;
        break;
      }
      if (isfinite(r))
        r_min_val = fmin(r_min_val, r);

      double th0 = y[2];
      double r0 = y[1];
      double vt0 = y[4];
      double vph0 = y[7];
      bh_rt_rk4_step(bh_rt_p2_deriv, lam, y, dlambda, &ud, BH_RT_P2_N_STATE, rk_ws);
      if ((step_idx % 4) == 0)
        bh_rt_p2_renormalize_vr(y, m);
      if (bh_rt_p2_all_finite(y) && y[1] >= r_cap) {
        bh_rt_p2_try_disk_cross(
            &disk_found, &disk_r_cross, &disk_vt, &disk_vph,
            th0, r0, vt0, vph0,
            y[2], y[1], y[4], y[7]);
      }
      lam += dlambda;
      steps_taken = step_idx + 1;
    }

    if (!broke) {
      termination_r = y[1];
    }
    if (status == BH_RT_STATUS_MAX_STEPS) {
      termination_r = y[1];
      {
        double r = y[1];
        if (isfinite(r))
          r_min_val = fmin(r_min_val, r);
      }
      if (!bh_rt_p2_all_finite(y))
        status = BH_RT_STATUS_NUMERICAL_ERROR;
    }
    if (!isfinite(r_min_val))
      r_min_val = NAN;

    out_status[ray]        = status;
    out_steps_taken[ray]   = steps_taken;
    out_termination_r[ray] = termination_r;
    out_r_min[ray]         = r_min_val;
    out_disk_hit[ray]      = disk_found;
    out_disk_r_cross[ray]  = disk_r_cross;
    out_disk_vt[ray]       = disk_vt;
    out_disk_vph[ray]      = disk_vph;
  }
}
