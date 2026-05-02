/*
 * One-line Schwarzschild u(phi) trace for manual parity checks vs phase1.trace_single_schwarzschild_ray.
 *
 * Prints:
 *   BH_SCHW_U_RESULT status=... termination_phi=... termination_r=... steps_taken=... r_min=...
 *
 * CLI — all floats optional after argv[0]:
 *   m b phi_start phi_max dphi [r_capture] [r_escape]
 * If `r_capture` is omitted, use Phase 1 default `(2*m + 1e-3)`. If present, `-` selects that default.
 */

#include "bh_rt_schwarzschild_u.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  double m = 1.0;
  double b = 6.0;
  double phi_start = 0.2;
  double phi_max = 8.0;
  double dphi = 0.002;
  double r_capture = NAN;
  double r_escape = 80.0;

  if (argc >= 2)
    m = strtod(argv[1], NULL);
  if (argc >= 3)
    b = strtod(argv[2], NULL);
  if (argc >= 4)
    phi_start = strtod(argv[3], NULL);
  if (argc >= 5)
    phi_max = strtod(argv[4], NULL);
  if (argc >= 6)
    dphi = strtod(argv[5], NULL);
  if (argc >= 7) {
    /* optional r_capture — use NAN if argv is "-" */
    if (argv[6][0] == '-' && argv[6][1] == '\0')
      r_capture = NAN;
    else
      r_capture = strtod(argv[6], NULL);
  }
  if (argc >= 8)
    r_escape = strtod(argv[7], NULL);

  if (!(isfinite(m) && isfinite(b) && b > 0.0 && isfinite(phi_start) &&
        isfinite(phi_max) && isfinite(dphi) && dphi > 0.0 && isfinite(r_escape)))
    return 2;
  /* Default r_capture matches phase1.py */
  if (isnan(r_capture))
    r_capture = 2.0 * m + 1e-3;

  bh_rt_schw_u_trace_result out;
  bh_rt_schwarzschild_u_trace(m, b, phi_start, phi_max, dphi, r_capture, r_escape,
                               &out);

  printf(
      "BH_SCHW_U_RESULT status=%d termination_phi=%.17g termination_r=%.17g "
      "steps_taken=%d r_min=%.17g\n",
      out.status, out.termination_phi, out.termination_r, out.steps_taken,
      out.r_min);
  return 0;
}
