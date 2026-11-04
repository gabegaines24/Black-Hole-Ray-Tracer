#include "bh_rt_kerr_phase3.h"
#include "bh_rt_kerr_phase3_batch.h"
#include "bh_rt_status.h"

#include <math.h>

enum { BH_RT_KERR_STATE_SIZE = 8 };

void bh_rt_kerr_phase3_batch_trace(
    const double *y0,
    int n,
    double m,
    double a,
    double dlambda,
    int max_steps,
    double r_escape,
    double r_horizon_epsilon,
    int *out_status,
    int *out_steps_taken,
    double *out_termination_r,
    double *out_r_min,
    double *out_eq_r_cross) {
  for (int ray = 0; ray < n; ++ray) {
    const double *src = y0 + ray * BH_RT_KERR_STATE_SIZE;
    bh_rt_kerr_trace_result trace = {0};
    bh_rt_kerr_equatorial_crossing crossing = {0};
    const int32_t rc = bh_rt_kerr_phase3_trace_crossing(
        src, m, a, dlambda, max_steps, r_escape, r_horizon_epsilon, &trace,
        &crossing);

    if (rc != BH_RT_KERR_OK) {
      out_status[ray] = BH_RT_STATUS_NUMERICAL_ERROR;
      out_steps_taken[ray] = 0;
      out_termination_r[ray] = NAN;
      out_r_min[ray] = NAN;
      out_eq_r_cross[ray] = NAN;
      continue;
    }

    out_status[ray] = trace.status;
    out_steps_taken[ray] = trace.steps_taken;
    out_termination_r[ray] = trace.termination_r;
    out_r_min[ray] = trace.r_min;
    out_eq_r_cross[ray] = crossing.hit ? crossing.r_cross : NAN;
  }
}
