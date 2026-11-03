/*
 * CUDA batch port of bh_rt_schwarzschild_phase2_batch.c
 *
 * Each CUDA thread traces one null geodesic.  Input/output layout is identical
 * to the CPU batch API (docs/STATE_API.md) so the same PyBind11 bridge can
 * dispatch to either backend at runtime.
 *
 * Build (requires CUDA 11+ and MSVC/GCC):
 *   nvcc -O2 -arch=sm_75 -Iinclude --compiler-options -fPIC -shared \
 *        -o libbh_rt_phase2_cuda.so \
 *        kernel/cuda/bh_rt_schwarzschild_phase2_batch.cu \
 *        -lm
 *
 * Runtime selection: bridge/module_phase2_cuda.cpp (planned) checks for
 * `libbh_rt_phase2_cuda.so` at import time and falls back to CPU if absent.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>
/* Status constants (mirror bh_rt_status.h for CUDA compilation). */
#define BH_RT_STATUS_CAPTURED        0
#define BH_RT_STATUS_ESCAPED         1
#define BH_RT_STATUS_MAX_STEPS       2
#define BH_RT_STATUS_NUMERICAL_ERROR 3

#define N_STATE 8
#define RK_WORK (5 * N_STATE)

/* ── device helpers ────────────────────────────────────────────────────────── */

__device__ static double d_schwarzschild_f(double r, double m) {
  return 1.0 - 2.0 * m / r;
}

__device__ static double d_christoffel(int mu, int a, int b,
                                        double r, double th, double m) {
  if (r <= 2.0 * m || !isfinite(r) || !isfinite(th)) return 0.0;
  double ff = d_schwarzschild_f(r, m);
  double s = sin(th), c = cos(th), s2 = s * s;
  double r2 = r * r;
  double m_r2f = m / (r2 * ff);
  if (a > b) { int t = a; a = b; b = t; }
  if (mu == 0) { if (a == 0 && b == 1) return m_r2f; return 0.0; }
  if (mu == 1) {
    if (a == 0 && b == 0) return m * ff / r2;
    if (a == 1 && b == 1) return -m / (r2 * ff);
    if (a == 2 && b == 2) return -r * ff;
    if (a == 3 && b == 3) return -r * ff * s2;
    return 0.0;
  }
  if (mu == 2) {
    if (a == 1 && b == 2) return 1.0 / r;
    if (a == 3 && b == 3) return -s * c;
    return 0.0;
  }
  if (mu == 3) {
    if (a == 1 && b == 3) return 1.0 / r;
    if (a == 2 && b == 3) return (fabs(s) > 1e-12) ? (c / s) : 0.0;
    return 0.0;
  }
  return 0.0;
}

__device__ static void d_deriv(double lam, const double *y, double *dy, double m) {
  (void)lam;
  double r = y[1], th = y[2];
  dy[0] = y[4]; dy[1] = y[5]; dy[2] = y[6]; dy[3] = y[7];
  for (int mu = 0; mu < 4; ++mu) {
    double sum = 0.0;
    for (int aa = 0; aa < 4; ++aa)
      for (int bb = 0; bb < 4; ++bb)
        sum += d_christoffel(mu, aa, bb, r, th, m) * y[4+aa] * y[4+bb];
    dy[4 + mu] = -sum;
  }
}

__device__ static int d_all_finite(const double *y) {
  for (int i = 0; i < N_STATE; ++i) if (!isfinite(y[i])) return 0;
  return 1;
}

__device__ static void d_renormalize(double *y, double m) {
  double r = y[1], th = y[2];
  double vt = y[4], vr = y[5], vth = y[6], vph = y[7];
  double f = d_schwarzschild_f(r, m);
  if (f <= 0.0 || !isfinite(f)) return;
  double s = sin(th);
  double a_part = (r * r) * (vth * vth + s * s * vph * vph);
  double inner = f * f * vt * vt - f * a_part;
  if (inner < 0.0 && inner > -1e-6) inner = 0.0;
  if (inner < 0.0 || !isfinite(inner)) return;
  double sgn = (vr >= 0.0) ? 1.0 : -1.0;
  y[5] = sgn * sqrt(inner);
}

/* RK4 step, inline to keep register usage low. */
__device__ static void d_rk4_step(double lam, double *y, double dlambda, double m) {
  double k1[N_STATE], k2[N_STATE], k3[N_STATE], k4[N_STATE], ytmp[N_STATE];
  double h = dlambda, h2 = h * 0.5;

  d_deriv(lam,         y,     k1, m);
  for (int i = 0; i < N_STATE; ++i) ytmp[i] = y[i] + h2 * k1[i];
  d_deriv(lam + h2,    ytmp,  k2, m);
  for (int i = 0; i < N_STATE; ++i) ytmp[i] = y[i] + h2 * k2[i];
  d_deriv(lam + h2,    ytmp,  k3, m);
  for (int i = 0; i < N_STATE; ++i) ytmp[i] = y[i] + h  * k3[i];
  d_deriv(lam + h,     ytmp,  k4, m);
  for (int i = 0; i < N_STATE; ++i)
    y[i] += (h / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
}

__device__ static void d_try_disk_cross(
    int *found, double *r_cross, double *vt_cross, double *vph_cross,
    double th0, double r0, double vt0, double vph0,
    double th1, double r1, double vt1, double vph1)
{
  if (*found) return;
  const double half_pi = 0.5 * 3.14159265358979323846;
  double d0 = th0 - half_pi;
  double d1 = th1 - half_pi;
  if (d0 * d1 >= 0.0) return;
  double alpha = fabs(d0) / (fabs(d0) + fabs(d1) + 1e-30);
  *found = 1;
  *r_cross = r0 * (1.0 - alpha) + r1 * alpha;
  *vt_cross = vt0 * (1.0 - alpha) + vt1 * alpha;
  *vph_cross = vph0 * (1.0 - alpha) + vph1 * alpha;
}

/* ── kernel ────────────────────────────────────────────────────────────────── */

extern "C" __global__ void bh_rt_phase2_batch_kernel(
    const double *y0,          /* (N, 8) row-major */
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
    int    *out_disk_hit,
    double *out_disk_r_cross,
    double *out_disk_vt,
    double *out_disk_vph)
{
  int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= n) return;

  double y[N_STATE];
  const double *src = y0 + ray * N_STATE;
  for (int k = 0; k < N_STATE; ++k) y[k] = src[k];

  double r_cap = 2.0 * m + r_horizon_epsilon;
  d_renormalize(y, m);

  double r_min_val = INFINITY;
  double lam = 0.0;
  int status = BH_RT_STATUS_MAX_STEPS;
  double termination_r = NAN;
  int steps_taken = 0;
  int broke = 0;

  int disk_found = 0;
  double disk_r_cross = NAN;
  double disk_vt = NAN;
  double disk_vph = NAN;

  for (int step_idx = 0; step_idx < max_steps; ++step_idx) {
    double r = y[1];
    if (!d_all_finite(y)) {
      status = BH_RT_STATUS_NUMERICAL_ERROR;
      termination_r = NAN; steps_taken = step_idx; broke = 1; break;
    }
    if (r < r_cap) {
      status = BH_RT_STATUS_CAPTURED;
      termination_r = r; steps_taken = step_idx; broke = 1; break;
    }
    if (r > r_escape) {
      status = BH_RT_STATUS_ESCAPED;
      termination_r = r; steps_taken = step_idx; broke = 1; break;
    }
    if (isfinite(r)) r_min_val = fmin(r_min_val, r);

    double th0 = y[2];
    double r0 = y[1];
    double vt0 = y[4];
    double vph0 = y[7];
    d_rk4_step(lam, y, dlambda, m);
    if ((step_idx % 4) == 0) d_renormalize(y, m);
    if (d_all_finite(y) && y[1] >= r_cap) {
      d_try_disk_cross(
          &disk_found, &disk_r_cross, &disk_vt, &disk_vph,
          th0, r0, vt0, vph0,
          y[2], y[1], y[4], y[7]);
    }
    lam += dlambda;
    steps_taken = step_idx + 1;
  }

  if (!broke) termination_r = y[1];
  if (status == BH_RT_STATUS_MAX_STEPS) {
    termination_r = y[1];
    if (isfinite(y[1])) r_min_val = fmin(r_min_val, y[1]);
    if (!d_all_finite(y)) status = BH_RT_STATUS_NUMERICAL_ERROR;
  }
  if (!isfinite(r_min_val)) r_min_val = NAN;

  out_status[ray]        = status;
  out_steps_taken[ray]   = steps_taken;
  out_termination_r[ray] = termination_r;
  out_r_min[ray]         = r_min_val;
  out_disk_hit[ray]      = disk_found;
  out_disk_r_cross[ray]  = disk_r_cross;
  out_disk_vt[ray]       = disk_vt;
  out_disk_vph[ray]      = disk_vph;
}

/* ── host-side launcher (C linkage, callable from Python ctypes or bridge) ── */

extern "C" int32_t bh_rt_schwarzschild_phase2_batch_trace_cuda(
    const double *h_y0,
    int32_t n,
    double m,
    double dlambda,
    int32_t max_steps,
    double r_escape,
    double r_horizon_epsilon,
    int32_t *h_status,
    int32_t *h_steps,
    double *h_term_r,
    double *h_r_min,
    int32_t *h_disk_hit,
    double *h_disk_r_cross,
    double *h_disk_vt,
    double *h_disk_vph)
{
  if (n < 0) return static_cast<int32_t>(cudaErrorInvalidValue);
  if (n == 0) return static_cast<int32_t>(cudaSuccess);
  if (h_y0 == nullptr || h_status == nullptr || h_steps == nullptr ||
      h_term_r == nullptr || h_r_min == nullptr || h_disk_hit == nullptr ||
      h_disk_r_cross == nullptr || h_disk_vt == nullptr || h_disk_vph == nullptr) {
    return static_cast<int32_t>(cudaErrorInvalidValue);
  }

  const size_t y0_bytes = static_cast<size_t>(n) * N_STATE * sizeof(double);
  const size_t int_bytes = static_cast<size_t>(n) * sizeof(int32_t);
  const size_t double_bytes = static_cast<size_t>(n) * sizeof(double);
  double *d_y0 = nullptr;
  int32_t *d_status = nullptr;
  int32_t *d_steps = nullptr;
  int32_t *d_disk_hit = nullptr;
  double *d_term_r = nullptr;
  double *d_r_min = nullptr;
  double *d_disk_r = nullptr;
  double *d_disk_vt = nullptr;
  double *d_disk_vph = nullptr;
  cudaError_t status = cudaSuccess;
  cudaError_t free_status = cudaSuccess;
  int32_t rc = static_cast<int32_t>(cudaSuccess);
  int block = 128;
  int grid = (n + block - 1) / block;

#define BH_CUDA_CHECK(call) \
  do { \
    status = (call); \
    if (status != cudaSuccess) { \
      rc = static_cast<int32_t>(status); \
      goto cleanup; \
    } \
  } while (0)

  BH_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y0), y0_bytes));
  BH_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_status), int_bytes));
  BH_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_steps), int_bytes));
  BH_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_term_r), double_bytes));
  BH_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_r_min), double_bytes));
  BH_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_disk_hit), int_bytes));
  BH_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_disk_r), double_bytes));
  BH_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_disk_vt), double_bytes));
  BH_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_disk_vph), double_bytes));
  BH_CUDA_CHECK(cudaMemcpy(d_y0, h_y0, y0_bytes, cudaMemcpyHostToDevice));

  bh_rt_phase2_batch_kernel<<<grid, block>>>(
      d_y0, n, m, dlambda, max_steps, r_escape, r_horizon_epsilon,
      d_status, d_steps, d_term_r, d_r_min,
      d_disk_hit, d_disk_r, d_disk_vt, d_disk_vph);
  BH_CUDA_CHECK(cudaGetLastError());
  BH_CUDA_CHECK(cudaDeviceSynchronize());
  BH_CUDA_CHECK(cudaMemcpy(h_status, d_status, int_bytes, cudaMemcpyDeviceToHost));
  BH_CUDA_CHECK(cudaMemcpy(h_steps, d_steps, int_bytes, cudaMemcpyDeviceToHost));
  BH_CUDA_CHECK(cudaMemcpy(h_term_r, d_term_r, double_bytes, cudaMemcpyDeviceToHost));
  BH_CUDA_CHECK(cudaMemcpy(h_r_min, d_r_min, double_bytes, cudaMemcpyDeviceToHost));
  BH_CUDA_CHECK(cudaMemcpy(h_disk_hit, d_disk_hit, int_bytes, cudaMemcpyDeviceToHost));
  BH_CUDA_CHECK(cudaMemcpy(h_disk_r_cross, d_disk_r, double_bytes, cudaMemcpyDeviceToHost));
  BH_CUDA_CHECK(cudaMemcpy(h_disk_vt, d_disk_vt, double_bytes, cudaMemcpyDeviceToHost));
  BH_CUDA_CHECK(cudaMemcpy(h_disk_vph, d_disk_vph, double_bytes, cudaMemcpyDeviceToHost));

cleanup:
  if (d_y0 != nullptr) {
    free_status = cudaFree(d_y0);
    if (rc == static_cast<int32_t>(cudaSuccess) && free_status != cudaSuccess)
      rc = static_cast<int32_t>(free_status);
  }
  if (d_status != nullptr) {
    free_status = cudaFree(d_status);
    if (rc == static_cast<int32_t>(cudaSuccess) && free_status != cudaSuccess)
      rc = static_cast<int32_t>(free_status);
  }
  if (d_steps != nullptr) {
    free_status = cudaFree(d_steps);
    if (rc == static_cast<int32_t>(cudaSuccess) && free_status != cudaSuccess)
      rc = static_cast<int32_t>(free_status);
  }
  if (d_term_r != nullptr) {
    free_status = cudaFree(d_term_r);
    if (rc == static_cast<int32_t>(cudaSuccess) && free_status != cudaSuccess)
      rc = static_cast<int32_t>(free_status);
  }
  if (d_r_min != nullptr) {
    free_status = cudaFree(d_r_min);
    if (rc == static_cast<int32_t>(cudaSuccess) && free_status != cudaSuccess)
      rc = static_cast<int32_t>(free_status);
  }
  if (d_disk_hit != nullptr) {
    free_status = cudaFree(d_disk_hit);
    if (rc == static_cast<int32_t>(cudaSuccess) && free_status != cudaSuccess)
      rc = static_cast<int32_t>(free_status);
  }
  if (d_disk_r != nullptr) {
    free_status = cudaFree(d_disk_r);
    if (rc == static_cast<int32_t>(cudaSuccess) && free_status != cudaSuccess)
      rc = static_cast<int32_t>(free_status);
  }
  if (d_disk_vt != nullptr) {
    free_status = cudaFree(d_disk_vt);
    if (rc == static_cast<int32_t>(cudaSuccess) && free_status != cudaSuccess)
      rc = static_cast<int32_t>(free_status);
  }
  if (d_disk_vph != nullptr) {
    free_status = cudaFree(d_disk_vph);
    if (rc == static_cast<int32_t>(cudaSuccess) && free_status != cudaSuccess)
      rc = static_cast<int32_t>(free_status);
  }
#undef BH_CUDA_CHECK
  return rc;
}
