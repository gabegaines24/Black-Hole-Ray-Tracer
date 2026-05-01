#include "bh_rt_rk4.h"
#include <string.h>

void bh_rt_rk4_step(bh_rt_deriv_fn f, double x, double *restrict y, double h,
                    void *userdata, int n, double *restrict workspace) {
  /* workspace: yt[n], k1[n], k2[n], k3[n], k4[n] → 5 * n doubles */
  double *yt = workspace;
  double *k1 = workspace + n;
  double *k2 = workspace + 2 * n;
  double *k3 = workspace + 3 * n;
  double *k4 = workspace + 4 * n;

  double h2 = h * 0.5;
  double h6 = h / 6.0;

  f(x, y, k1, userdata);

  for (int i = 0; i < n; ++i)
    yt[i] = y[i] + h2 * k1[i];
  f(x + h2, yt, k2, userdata);

  for (int i = 0; i < n; ++i)
    yt[i] = y[i] + h2 * k2[i];
  f(x + h2, yt, k3, userdata);

  for (int i = 0; i < n; ++i)
    yt[i] = y[i] + h * k3[i];
  f(x + h, yt, k4, userdata);

  for (int i = 0; i < n; ++i)
    y[i] += h6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}
