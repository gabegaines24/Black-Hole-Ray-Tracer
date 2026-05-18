/*
 * One-line Schwarzschild 3D trace for manual parity checks vs phase2_geodesic.
 *
 * Default initial state matches a static observer at r=30, theta=pi/2,
 * centered on the black hole (same as phase2_camera center pixel direction).
 */

#include "bh_rt_schwarzschild_3d.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  double m = 1.0;
  double dlambda = 0.1;
  int max_steps = 3000;
  double r_escape = 80.0;
  double r_horizon_epsilon = 1e-3;

  if (argc >= 2)
    m = strtod(argv[1], NULL);
  if (argc >= 3)
    dlambda = strtod(argv[2], NULL);
  if (argc >= 4)
    max_steps = (int)strtol(argv[3], NULL, 10);
  if (argc >= 5)
    r_escape = strtod(argv[4], NULL);
  if (argc >= 6)
    r_horizon_epsilon = strtod(argv[5], NULL);

  double r0 = 30.0;
  double th0 = 1.5707963267948966;
  double f = 1.0 - 2.0 * m / r0;
  if (!(isfinite(m) && m > 0.0 && isfinite(dlambda) && dlambda > 0.0 &&
        max_steps >= 0 && isfinite(r_escape) && f > 0.0))
    return 2;

  double x0[4] = {0.0, r0, th0, 0.0};
  double v0[4] = {1.0 / sqrt(f), -sqrt(f), 0.0, 0.0};
  bh_rt_schw3d_trace_result out;

  bh_rt_schwarzschild_3d_trace(x0, v0, m, dlambda, max_steps, r_escape,
                               r_horizon_epsilon, &out);

  printf("BH_SCHW3D_RESULT status=%d steps_taken=%d termination_r=%.17g "
         "termination_lambda=%.17g r_min=%.17g final_r=%.17g\n",
         out.status, out.steps_taken, out.termination_r,
         out.termination_lambda, out.r_min, out.final_state[1]);
  return 0;
}
