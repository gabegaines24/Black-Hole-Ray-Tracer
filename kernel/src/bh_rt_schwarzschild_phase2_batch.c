#include "bh_rt_schwarzschild_phase2_batch.h"
#include "bh_rt_schwarzschild_phase2_internal.h"
#include "bh_rt_status.h"

#include <math.h>
#include <string.h>

#ifndef INFINITY
#define INFINITY (1.0 / 0.0)
#endif

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
    double *out_eq_r_cross)
{
  static const double PI_HALF = 1.5707963267948966;
  double r_cap = 2.0 * m + r_horizon_epsilon;
  bh_rt_p2_userdata ud;
  ud.m = m;
  double rk_ws[BH_RT_P2_RK_WORK];

  for (int ray = 0; ray < n; ++ray) {
    /* Copy initial state for this ray. */
    double y[BH_RT_P2_N_STATE];
    const double *src = y0 + ray * BH_RT_P2_N_STATE;
    for (int k = 0; k < BH_RT_P2_N_STATE; ++k)
      y[k] = src[k];

    bh_rt_p2_renormalize_vr(y, m);

    double r_min_val = INFINITY;
    double lam = 0.0;
    int status = BH_RT_STATUS_MAX_STEPS;
    double termination_r = NAN;
    double eq_r_cross = NAN;   /* first equatorial-plane crossing */
    int steps_taken = 0;
    int broke = 0;

    /* theta offset from pi/2 at the start of each step (for sign-change detection). */
    double prev_dth = y[2] - PI_HALF;

    for (int step_idx = 0; step_idx < max_steps; ++step_idx) {
      double r = y[1];
      double th = y[2];

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

      /* Equatorial crossing detection: theta crosses pi/2 (sign change of theta - pi/2). */
      if (!isfinite(eq_r_cross)) {
        double cur_dth = th - PI_HALF;
        if (step_idx > 0 && prev_dth * cur_dth < 0.0 && isfinite(r)) {
          /* Linear interpolation to find crossing radius. */
          double frac = fabs(prev_dth) / (fabs(prev_dth) + fabs(cur_dth) + 1e-30);
          eq_r_cross = r;  /* use current r as approximation */
          (void)frac;      /* could refine: r_cross = r_prev + frac*(r - r_prev) */
        }
        prev_dth = cur_dth;
      }

      bh_rt_rk4_step(bh_rt_p2_deriv, lam, y, dlambda, &ud, BH_RT_P2_N_STATE, rk_ws);
      if ((step_idx % 4) == 0)
        bh_rt_p2_renormalize_vr(y, m);
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
    out_eq_r_cross[ray]    = eq_r_cross;
  }
}
