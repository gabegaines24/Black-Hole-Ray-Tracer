/*
 * Matches Phase A (harmonic oscillator RK4 sanity) for parity against Python phase1.run_rk4_sanity:
 * same comparison order vs cos(omega*t) before each rk4_step.
 *
 * Prints one parseable line:
 * BH_RK4_RESULT dt=... t_total=... omega=... steps=... max_abs_err_pos=... mean_abs_err_pos=...
 */

#include "bh_rt_rk4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  double omega_sq;
} HarmCtx;

static void harmonic_deriv(double x, const double *restrict y,
                           double *restrict out_dy, void *userdata) {
  (void)x;
  const HarmCtx *c = (const HarmCtx *)userdata;
  double pos = y[0];
  double vel = y[1];
  out_dy[0] = vel;
  out_dy[1] = -c->omega_sq * pos;
}

int main(int argc, char **argv) {
  double dt = 0.02;
  double t_total = 8.0;
  double omega = 1.0;

  if (argc >= 2)
    dt = strtod(argv[1], NULL);
  if (argc >= 3)
    t_total = strtod(argv[2], NULL);
  if (argc >= 4)
    omega = strtod(argv[3], NULL);

  HarmCtx ctx;
  ctx.omega_sq = omega * omega;

  int steps = (int)(t_total / dt + 1e-9);
  if (steps <= 0) {
    fprintf(stderr, "steps must be positive\n");
    return 2;
  }

  double y[2] = {1.0, 0.0};
  double workspace[10]; /* n=2 → 5*n */

  double max_err = 0.0;
  double sum_err = 0.0;

  double t = 0.0;
  for (int s = 0; s < steps; ++s) {
    double exact = cos(omega * t);
    double err_pos = fabs(y[0] - exact);
    if (err_pos > max_err)
      max_err = err_pos;
    sum_err += err_pos;
    bh_rt_rk4_step(harmonic_deriv, t, y, dt, &ctx, 2, workspace);
    t += dt;
  }

  double mean_err = sum_err / (double)steps;
  printf("BH_RK4_RESULT dt=%.17g t_total=%.17g omega=%.17g steps=%d "
         "max_abs_err_pos=%.17g mean_abs_err_pos=%.17g\n",
         dt, t_total, omega, steps, max_err, mean_err);
  return 0;
}
