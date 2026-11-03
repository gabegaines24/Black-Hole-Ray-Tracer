#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "bh_rt_schwarzschild_phase2.h"
#include "bh_rt_schwarzschild_phase2_batch.h"

namespace py = pybind11;

constexpr std::int64_t kMaxBatchRays = 1'000'000;

PYBIND11_MODULE(_native_phase2, m) {
  m.doc() = R"pbdoc(Schwarzschild 3D null-ray kernel (Phase 2 Christoffel path).)pbdoc";

  /* ── single-ray ────────────────────────────────────────────────────────── */
  m.def(
      "schwarzschild_phase2_trace",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> y0,
         double mass, double dlambda, int max_steps,
         double r_escape, double r_horizon_epsilon) -> py::dict {
        py::buffer_info buf = y0.request();
        if (buf.ndim != 1 || buf.shape[0] != 8)
          throw std::invalid_argument("y0 must be shape (8,) ndim=1");
        if (static_cast<size_t>(buf.itemsize) != sizeof(double))
          throw std::invalid_argument("y0 must be float64");

        double y_copy[8];
        std::memcpy(y_copy, buf.ptr, sizeof(y_copy));

        bh_rt_phase2_trace_result out{};
        {
          py::gil_scoped_release release;
          bh_rt_schwarzschild_phase2_trace(y_copy, mass, dlambda, max_steps,
                                           r_escape, r_horizon_epsilon, &out);
        }

        py::dict d;
        d["status"]             = out.status;
        d["steps_taken"]        = out.steps_taken;
        d["max_steps"]          = out.max_steps;
        d["termination_r"]      = out.termination_r;
        d["termination_lambda"] = out.termination_lambda;
        d["r_min"]              = out.r_min;
        return d;
      },
      py::arg("y0"), py::arg("m"), py::arg("dlambda"), py::arg("max_steps"),
      py::arg("r_escape"), py::arg("r_horizon_epsilon") = 1e-3,
      R"pbdoc(Trace one null geodesic. y0 shape (8,) float64.)pbdoc");

  /* ── batch ─────────────────────────────────────────────────────────────── */
  m.def(
      "schwarzschild_phase2_batch_trace",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> y0_batch,
         double mass, double dlambda, int max_steps,
         double r_escape, double r_horizon_epsilon) -> py::dict {
        py::buffer_info buf = y0_batch.request();
        if (buf.ndim != 2 || buf.shape[1] != 8)
          throw std::invalid_argument("y0_batch must be shape (N, 8) float64");

        const std::int64_t n64 = static_cast<std::int64_t>(buf.shape[0]);
        if (n64 > kMaxBatchRays ||
            n64 > std::numeric_limits<std::int32_t>::max()) {
          throw std::invalid_argument(
              "y0_batch exceeds maximum supported batch size");
        }
        std::int32_t n = static_cast<std::int32_t>(n64);
        const double *y0_ptr = static_cast<const double *>(buf.ptr);

        /* Allocate output arrays. */
        auto out_status   = py::array_t<std::int32_t>(n);
        auto out_steps    = py::array_t<std::int32_t>(n);
        auto out_term_r   = py::array_t<double>(n);
        auto out_r_min    = py::array_t<double>(n);
        auto out_disk_hit = py::array_t<std::int32_t>(n);
        auto out_disk_r   = py::array_t<double>(n);
        auto out_disk_vt  = py::array_t<double>(n);
        auto out_disk_vph = py::array_t<double>(n);

        {
          py::gil_scoped_release release;
          bh_rt_schwarzschild_phase2_batch_trace(
              y0_ptr, n, mass, dlambda, max_steps, r_escape, r_horizon_epsilon,
              static_cast<std::int32_t *>(
                  out_status.mutable_unchecked<1>().mutable_data(0)),
              static_cast<std::int32_t *>(
                  out_steps.mutable_unchecked<1>().mutable_data(0)),
              static_cast<double *>(
                  out_term_r.mutable_unchecked<1>().mutable_data(0)),
              static_cast<double *>(
                  out_r_min.mutable_unchecked<1>().mutable_data(0)),
              static_cast<std::int32_t *>(
                  out_disk_hit.mutable_unchecked<1>().mutable_data(0)),
              static_cast<double *>(
                  out_disk_r.mutable_unchecked<1>().mutable_data(0)),
              static_cast<double *>(
                  out_disk_vt.mutable_unchecked<1>().mutable_data(0)),
              static_cast<double *>(
                  out_disk_vph.mutable_unchecked<1>().mutable_data(0)));
        }

        py::dict d;
        d["status"]          = out_status;
        d["steps_taken"]     = out_steps;
        d["termination_r"]   = out_term_r;
        d["r_min"]           = out_r_min;
        d["disk_hit"]        = out_disk_hit;
        d["disk_r_cross"]    = out_disk_r;
        d["disk_vt"]         = out_disk_vt;
        d["disk_vph"]        = out_disk_vph;
        return d;
      },
      py::arg("y0_batch"), py::arg("m"), py::arg("dlambda"), py::arg("max_steps"),
      py::arg("r_escape"), py::arg("r_horizon_epsilon") = 1e-3,
      R"pbdoc(Trace N null geodesics. y0_batch shape (N, 8) float64, C-contiguous.
Returns dict with arrays: status(N,int32), steps_taken(N,int32),
termination_r(N,float64), r_min(N,float64), disk_hit(N,int32),
disk_r_cross(N,float64), disk_vt(N,float64), disk_vph(N,float64).)pbdoc");
}
