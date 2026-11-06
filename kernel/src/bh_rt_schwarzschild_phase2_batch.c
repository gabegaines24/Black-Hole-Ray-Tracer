#include "bh_rt_schwarzschild_phase2_batch.h"
#include "bh_rt_schwarzschild_phase2_internal.h"
#include "bh_rt_status.h"

#include <math.h>
#include <stddef.h>

#ifndef INFINITY
#define INFINITY (1.0 / 0.0)
#endif

static void trace_batch(
    const double *y0,
    int n,
    double m,
    double dlambda,
    int max_steps,
    double r_escape,
    double r_horizon_epsilon,
    int *out_status,
    int *out_steps_taken,
    double *out_termination_r,
    double *out_r_min,
    double *out_eq_r_cross,
    int *out_disk_hit,
    double *out_disk_r_cross,
    double *out_disk_vt,
    double *out_disk_vph) {
  static const double PI_HALF = 1.5707963267948966;
  const double r_cap = 2.0 * m + r_horizon_epsilon;
  bh_rt_p2_userdata ud;
  ud.m = m;
  double rk_ws[BH_RT_P2_RK_WORK];

  for (int ray = 0; ray < n; ++ray) {
    double y[BH_RT_P2_N_STATE];
    const double *src = y0 + ray * BH_RT_P2_N_STATE;
    for (int k = 0; k < BH_RT_P2_N_STATE; ++k)
      y[k] = src[k];

    bh_rt_p2_renormalize_vr(y, m);

    double r_min_val = INFINITY;
    double lam = 0.0;
    int status = BH_RT_STATUS_MAX_STEPS;
    double termination_r = NAN;
    double eq_r_cross = NAN;
    int disk_hit = 0;
    double disk_r_cross = NAN;
    double disk_vt = NAN;
    double disk_vph = NAN;
    int steps_taken = 0;
    int broke = 0;
    double previous_r = y[1];
    double previous_dth = y[2] - PI_HALF;
    double previous_vt = y[4];
    double previous_vph = y[7];

    for (int step_idx = 0; step_idx < max_steps; ++step_idx) {
      const double r = y[1];
      const double current_dth = y[2] - PI_HALF;

      if (!bh_rt_p2_all_finite(y)) {
        status = BH_RT_STATUS_NUMERICAL_ERROR;
        termination_r = r;
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

      if (step_idx > 0 && !disk_hit && previous_dth * current_dth < 0.0) {
        const double alpha =
            fabs(previous_dth) /
            (fabs(previous_dth) + fabs(current_dth) + 1e-30);
        disk_hit = 1;
        disk_r_cross =
            previous_r * (1.0 - alpha) + r * alpha;
        disk_vt = previous_vt * (1.0 - alpha) + y[4] * alpha;
        disk_vph = previous_vph * (1.0 - alpha) + y[7] * alpha;
        eq_r_cross = disk_r_cross;
      }
      previous_r = r;
      previous_dth = current_dth;
      previous_vt = y[4];
      previous_vph = y[7];

      bh_rt_rk4_step(
          bh_rt_p2_deriv, lam, y, dlambda, &ud, BH_RT_P2_N_STATE, rk_ws);
      if ((step_idx % 4) == 0)
        bh_rt_p2_renormalize_vr(y, m);
      lam += dlambda;
      steps_taken = step_idx + 1;
    }

    if (!broke)
      termination_r = y[1];
    if (status == BH_RT_STATUS_MAX_STEPS) {
      termination_r = y[1];
      if (isfinite(y[1]))
        r_min_val = fmin(r_min_val, y[1]);
      if (!bh_rt_p2_all_finite(y))
        status = BH_RT_STATUS_NUMERICAL_ERROR;
    }
    if (!isfinite(r_min_val))
      r_min_val = NAN;

    out_status[ray] = status;
    out_steps_taken[ray] = steps_taken;
    out_termination_r[ray] = termination_r;
    out_r_min[ray] = r_min_val;
    if (out_eq_r_cross != NULL)
      out_eq_r_cross[ray] = eq_r_cross;
    if (out_disk_hit != NULL)
      out_disk_hit[ray] = disk_hit;
    if (out_disk_r_cross != NULL)
      out_disk_r_cross[ray] = disk_r_cross;
    if (out_disk_vt != NULL)
      out_disk_vt[ray] = disk_vt;
    if (out_disk_vph != NULL)
      out_disk_vph[ray] = disk_vph;
  }
}

void bh_rt_schwarzschild_phase2_batch_trace(
    const double *y0,
    int n,
    double m,
    double dlambda,
    int max_steps,
    double r_escape,
    double r_horizon_epsilon,
    int *out_status,
    int *out_steps_taken,
    double *out_termination_r,
    double *out_r_min,
    double *out_eq_r_cross) {
  trace_batch(
      y0, n, m, dlambda, max_steps, r_escape, r_horizon_epsilon, out_status,
      out_steps_taken, out_termination_r, out_r_min, out_eq_r_cross, NULL, NULL,
      NULL, NULL);
}

void bh_rt_schwarzschild_phase2_batch_trace_disk(
    const double *y0,
    int n,
    double m,
    double dlambda,
    int max_steps,
    double r_escape,
    double r_horizon_epsilon,
    int *out_status,
    int *out_steps_taken,
    double *out_termination_r,
    double *out_r_min,
    int *out_disk_hit,
    double *out_disk_r_cross,
    double *out_disk_vt,
    double *out_disk_vph) {
  trace_batch(
      y0, n, m, dlambda, max_steps, r_escape, r_horizon_epsilon, out_status,
      out_steps_taken, out_termination_r, out_r_min, NULL, out_disk_hit,
      out_disk_r_cross, out_disk_vt, out_disk_vph);
}
