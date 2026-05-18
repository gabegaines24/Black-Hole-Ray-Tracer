#include "bh_rt_schwarzschild_3d.h"

#include <math.h>
#include <string.h>

#include "bh_rt_rk4.h"

static double f_schwarzschild(double r, double m) { return 1.0 - 2.0 * m / r; }

double bh_rt_schw3d_metric_invariant(const double y[8], double m) {
  double r = y[1];
  double th = y[2];
  double vt = y[4];
  double vr = y[5];
  double vth = y[6];
  double vph = y[7];
  double f = f_schwarzschild(r, m);
  double s = sin(th);
  return -f * vt * vt + (vr * vr) / f + r * r * vth * vth +
         r * r * s * s * vph * vph;
}

void bh_rt_schw3d_renormalize_null(double y[8], double m,
                                   int preserve_vr_sign) {
  double r = y[1];
  double th = y[2];
  double vt = y[4];
  double vr = y[5];
  double vth = y[6];
  double vph = y[7];
  double f = f_schwarzschild(r, m);
  if (f <= 0.0 || !isfinite(f))
    return;

  double s = sin(th);
  double angular = r * r * (vth * vth + s * s * vph * vph);
  double inner = f * f * vt * vt - f * angular;
  if (inner < 0.0 && inner > -1e-6)
    inner = 0.0;
  if (inner < 0.0 || !isfinite(inner))
    return;

  double sign = vr >= 0.0 ? 1.0 : -1.0;
  if (!preserve_vr_sign)
    sign = 1.0;
  y[5] = sign * sqrt(inner);
}

static void schw3d_rhs(double lam, const double *y, double *dy, void *userdata) {
  (void)lam;
  double m = *((const double *)userdata);
  double r = y[1];
  double th = y[2];
  double vt = y[4];
  double vr = y[5];
  double vth = y[6];
  double vph = y[7];

  dy[0] = vt;
  dy[1] = vr;
  dy[2] = vth;
  dy[3] = vph;

  if (r <= 2.0 * m || !(isfinite(r) && isfinite(th))) {
    dy[4] = 0.0;
    dy[5] = 0.0;
    dy[6] = 0.0;
    dy[7] = 0.0;
    return;
  }

  double f = f_schwarzschild(r, m);
  double s = sin(th);
  double c = cos(th);
  double s2 = s * s;
  double r2 = r * r;
  double gamma_t_tr = m / (r2 * f);
  double gamma_r_tt = m * f / r2;
  double gamma_r_rr = m / (r2 * f);
  double gamma_r_thth = -r * f;
  double gamma_r_phph = -r * f * s2;
  double gamma_th_rth = 1.0 / r;
  double gamma_th_phph = -s * c;
  double gamma_ph_rph = 1.0 / r;
  double gamma_ph_thph = fabs(s) > 1e-12 ? c / s : 0.0;

  dy[4] = -(2.0 * gamma_t_tr * vt * vr);
  dy[5] = -(gamma_r_tt * vt * vt + gamma_r_rr * vr * vr +
            gamma_r_thth * vth * vth + gamma_r_phph * vph * vph);
  dy[6] = -(2.0 * gamma_th_rth * vr * vth + gamma_th_phph * vph * vph);
  dy[7] = -(2.0 * gamma_ph_rph * vr * vph +
            2.0 * gamma_ph_thph * vth * vph);
}

static int all_state_finite(const double y[8]) {
  for (int i = 0; i < 8; ++i) {
    if (!isfinite(y[i]))
      return 0;
  }
  return 1;
}

void bh_rt_schwarzschild_3d_trace(const double x0[4], const double v0[4],
                                  double m, double dlambda, int max_steps,
                                  double r_escape,
                                  double r_horizon_epsilon,
                                  bh_rt_schw3d_trace_result *out) {
  double y[8] = {x0[0], x0[1], x0[2], x0[3],
                 v0[0], v0[1], v0[2], v0[3]};
  bh_rt_schw3d_renormalize_null(y, m, 1);

  out->status = BH_RT_STATUS_MAX_STEPS;
  out->steps_taken = 0;
  out->max_steps = max_steps;
  out->termination_r = NAN;
  out->termination_lambda = 0.0;
  out->r_min = NAN;
  memcpy(out->final_state, y, sizeof(out->final_state));

  double r_cap = 2.0 * m + r_horizon_epsilon;
  double r_min = HUGE_VAL;
  double lam = 0.0;
  double workspace[40]; /* 5 * n for n=8; see bh_rt_rk4.c */

  for (int step_idx = 0; step_idx < max_steps; ++step_idx) {
    if (!all_state_finite(y)) {
      out->status = BH_RT_STATUS_NUMERICAL_ERROR;
      out->steps_taken = step_idx;
      break;
    }

    double r = y[1];
    if (r < r_cap) {
      out->status = BH_RT_STATUS_CAPTURED;
      out->termination_r = r;
      out->termination_lambda = lam;
      out->steps_taken = step_idx;
      break;
    }
    if (r > r_escape) {
      out->status = BH_RT_STATUS_ESCAPED;
      out->termination_r = r;
      out->termination_lambda = lam;
      out->steps_taken = step_idx;
      break;
    }

    if (r < r_min)
      r_min = r;

    bh_rt_rk4_step(schw3d_rhs, lam, y, dlambda, (void *)&m, 8, workspace);
    if (step_idx % 4 == 0)
      bh_rt_schw3d_renormalize_null(y, m, 1);
    lam += dlambda;
    out->steps_taken = step_idx + 1;
  }

  if (out->status == BH_RT_STATUS_MAX_STEPS) {
    out->termination_r = y[1];
    out->termination_lambda = lam;
    if (isfinite(y[1]) && y[1] < r_min)
      r_min = y[1];
  }

  out->r_min = isfinite(r_min) ? r_min : NAN;
  memcpy(out->final_state, y, sizeof(out->final_state));
}
