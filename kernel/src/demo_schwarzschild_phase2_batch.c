/* Batch Phase 2 smoke demo: traces a small fixed set of rays and prints results.
 *
 * Usage: ./phase2_batch_demo
 *
 * Intended for quick build verification; not a correctness tool.
 */

#include <math.h>
#include <stdio.h>

#include "bh_rt_schwarzschild_phase2_batch.h"
#include "bh_rt_status.h"

#define N_RAYS 4

static const char *status_name(int s) {
  switch (s) {
    case BH_RT_STATUS_CAPTURED:       return "captured";
    case BH_RT_STATUS_ESCAPED:        return "escaped";
    case BH_RT_STATUS_MAX_STEPS:      return "max_steps";
    case BH_RT_STATUS_NUMERICAL_ERROR: return "numeric";
    default:                           return "unknown";
  }
}

int main(void) {
  /* Four rays from a static observer at r=30, equatorial, varying screen x */
  double m = 1.0;
  double r_obs = 30.0;
  double theta = 1.5707963267948966; /* pi/2 */

  /* Manual tetrad: vt=1/sqrt(f), vr=-sqrt(f) (inward), vth=0, vph=0 */
  double f = 1.0 - 2.0 * m / r_obs;
  double vt = 1.0 / sqrt(f);
  double vr = -sqrt(f);   /* straight inward */

  double y0[N_RAYS * 8];
  for (int i = 0; i < N_RAYS; ++i) {
    double offset = -0.15 + i * 0.10; /* small angular spread */
    y0[i*8+0] = 0.0;       /* t */
    y0[i*8+1] = r_obs;     /* r */
    y0[i*8+2] = theta;     /* theta */
    y0[i*8+3] = 0.0;       /* phi */
    y0[i*8+4] = vt;        /* v^t */
    y0[i*8+5] = vr;        /* v^r */
    y0[i*8+6] = offset / r_obs; /* v^theta */
    y0[i*8+7] = 0.0;       /* v^phi */
  }

  int    status[N_RAYS];
  int    steps[N_RAYS];
  double term_r[N_RAYS];
  double r_min[N_RAYS];

  bh_rt_schwarzschild_phase2_batch_trace(
      y0, N_RAYS, m, 0.1, 4000, 80.0, 1e-3,
      status, steps, term_r, r_min);

  printf("BH_PHASE2_BATCH_RESULT  ray  status         steps  term_r       r_min\n");
  for (int i = 0; i < N_RAYS; ++i) {
    printf("BH_PHASE2_BATCH_RESULT  %3d  %-14s %5d  %11.4f  %11.4f\n",
           i, status_name(status[i]), steps[i], term_r[i], r_min[i]);
  }
  return 0;
}
