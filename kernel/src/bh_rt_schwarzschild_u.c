#include "bh_rt_schwarzschild_u.h"

#include <math.h>

#include "bh_rt_rk4.h"

static void schw_u_deriv(double phi, const double *y, double *dy, void *userdata) {
  (void)phi;
  double M = *((const double *)userdata);
  double u = y[0];
  double du = y[1];
  double d2 = 3.0 * M * u * u - u;
  dy[0] = du;
  dy[1] = d2;
}

void bh_rt_schwarzschild_u_trace(double m, double b, double phi_start,
                                 double phi_max, double dphi, double r_capture,
                                 double r_escape,
                                 bh_rt_schw_u_trace_result *out) {
  double phi = phi_start;
  out->termination_r = NAN;
  out->steps_taken = 0;
  out->status = BH_RT_STATUS_MAX_STEPS;
  out->r_min = NAN;

  double y_data[2] = {sin(phi_start) / b, cos(phi_start) / b};
  double workspace[10];

  /*
   * Mirror phase1.trace_single_schwarzschild_ray():
   *   steps = int((phi_max - phi_start) / dphi)
   *   loops step_idx = 0 .. steps-1
   */
  int steps_total = (int)((phi_max - phi_start) / dphi);

  double y_agg_rmin = NAN;
  int any_sample = 0;

  long step_idx;
  for (step_idx = 0; step_idx < steps_total; ++step_idx) {
    double u = y_data[0];

    if (!isfinite(u)) {
      out->status = BH_RT_STATUS_NUMERICAL_ERROR;
      out->steps_taken = (int)step_idx;
      break;
    }
    if (u <= 0.0) {
      out->status = BH_RT_STATUS_ESCAPED;
      out->termination_r = HUGE_VAL;
      /* Python: steps_taken = step_idx here (before recording a sample). */
      out->steps_taken = (int)step_idx;
      break;
    }

    double r = 1.0 / u;
    if (!isfinite(r)) {
      out->status = BH_RT_STATUS_NUMERICAL_ERROR;
      out->steps_taken = (int)step_idx;
      break;
    }

    if (!any_sample) {
      y_agg_rmin = r;
      any_sample = 1;
    } else if (r < y_agg_rmin) {
      y_agg_rmin = r;
    }

    out->termination_r = r;
    /* After appending trajectory sample Python sets steps_taken = step_idx + 1. */
    out->steps_taken = (int)(step_idx + 1);

    if (r < r_capture) {
      out->status = BH_RT_STATUS_CAPTURED;
      break;
    }
    if (r > r_escape) {
      out->status = BH_RT_STATUS_ESCAPED;
      break;
    }

    bh_rt_rk4_step(schw_u_deriv, phi, y_data, dphi, (void *)&m, 2, workspace);
    phi += dphi;
  }

  /* Python returns final accumulated phi regardless of breakout path */
  out->termination_phi = phi;
  out->r_min = any_sample ? y_agg_rmin : NAN;

  /* If loop never executed, match Python sentinel fields */
  if (steps_total <= 0) {
    out->steps_taken = 0;
    out->termination_phi = phi_start;
  }
}
