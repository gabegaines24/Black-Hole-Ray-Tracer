#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "bh_rt_schwarzschild_phase2.h"
#include "bh_rt_schwarzschild_phase2_batch.h"

namespace py = pybind11;

PYBIND11_MODULE(_native_phase2, m) {
  m.doc() = R"pbdoc(Schwarzschild 3D null-ray kernel (Phase 2 Christoffel path).)pbdoc";

  /* ── single-ray ────────────────────────────────────────────────────────── */
  m.def(
      "schwarzschild_phase2_trace",
      [](py::array_t<double> y0, double mass, double dlambda, int max_steps,
         double r_escape, double r_horizon_epsilon) -> py::dict {
        py::buffer_info buf = y0.request();
        if (buf.ndim != 1 || buf.shape[0] != 8)
          throw std::invalid_argument("y0 must be shape (8,) ndim=1");
        if (static_cast<size_t>(buf.itemsize) != sizeof(double))
          throw std::invalid_argument("y0 must be float64");

        double y_copy[8];
        std::memcpy(y_copy, buf.ptr, sizeof(y_copy));

        bh_rt_phase2_trace_result out{};
        bh_rt_schwarzschild_phase2_trace(y_copy, mass, dlambda, max_steps,
                                         r_escape, r_horizon_epsilon, &out);

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

        int n = static_cast<int>(buf.shape[0]);
        const double *y0_ptr = static_cast<const double *>(buf.ptr);

        /* Allocate output arrays. */
        auto out_status    = py::array_t<int>(n);
        auto out_steps     = py::array_t<int>(n);
        auto out_term_r    = py::array_t<double>(n);
        auto out_r_min     = py::array_t<double>(n);
        auto out_eq_r_cross = py::array_t<double>(n);

        bh_rt_schwarzschild_phase2_batch_trace(
            y0_ptr, n, mass, dlambda, max_steps, r_escape, r_horizon_epsilon,
            static_cast<int *>(out_status.mutable_unchecked<1>().mutable_data(0)),
            static_cast<int *>(out_steps.mutable_unchecked<1>().mutable_data(0)),
            static_cast<double *>(out_term_r.mutable_unchecked<1>().mutable_data(0)),
            static_cast<double *>(out_r_min.mutable_unchecked<1>().mutable_data(0)),
            static_cast<double *>(out_eq_r_cross.mutable_unchecked<1>().mutable_data(0)));

        py::dict d;
        d["status"]          = out_status;
        d["steps_taken"]     = out_steps;
        d["termination_r"]   = out_term_r;
        d["r_min"]           = out_r_min;
        d["eq_r_cross"]      = out_eq_r_cross;
        return d;
      },
      py::arg("y0_batch"), py::arg("m"), py::arg("dlambda"), py::arg("max_steps"),
      py::arg("r_escape"), py::arg("r_horizon_epsilon") = 1e-3,
      R"pbdoc(Trace N null geodesics. y0_batch shape (N, 8) float64, C-contiguous.
Returns dict with arrays: status(N,int32), steps_taken(N,int32),
termination_r(N,float64), r_min(N,float64), eq_r_cross(N,float64).)pbdoc");
}
