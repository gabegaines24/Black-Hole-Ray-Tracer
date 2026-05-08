#include <cstddef>
#include <cstring>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "bh_rt_schwarzschild_phase2.h"

namespace py = pybind11;

PYBIND11_MODULE(_native_phase2, m) {
  m.doc() = R"pbdoc(Schwarzschild 3D null-ray kernel (Phase 2 Christoffel path).)pbdoc";

  m.def(
      "schwarzschild_phase2_trace",
      [](py::array_t<double> y0, double mass, double dlambda, int max_steps,
         double r_escape, double r_horizon_epsilon) -> py::dict {
        py::buffer_info buf = y0.request();
        if (buf.ndim != 1 || buf.shape[0] != 8) {
          throw std::invalid_argument("y0 must be shape (8,) ndim=1");
        }
        if (static_cast<size_t>(buf.itemsize) != sizeof(double)) {
          throw std::invalid_argument("y0 must be float64");
        }

        double y_copy[8];
        std::memcpy(y_copy, buf.ptr, sizeof(y_copy));

        bh_rt_phase2_trace_result out{};
        bh_rt_schwarzschild_phase2_trace(y_copy, mass, dlambda, max_steps,
                                         r_escape, r_horizon_epsilon, &out);

        py::dict d;
        d["status"] = out.status;
        d["steps_taken"] = out.steps_taken;
        d["max_steps"] = out.max_steps;
        d["termination_r"] = out.termination_r;
        d["termination_lambda"] = out.termination_lambda;
        d["r_min"] = out.r_min;
        return d;
      },
      py::arg("y0"), py::arg("m"), py::arg("dlambda"), py::arg("max_steps"),
      py::arg("r_escape"), py::arg("r_horizon_epsilon") = 1e-3);
}
