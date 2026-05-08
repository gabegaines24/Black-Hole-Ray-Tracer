#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_native_phase2, m) {
  m.doc() = R"pbdoc(Schwarzschild 3D null-ray kernel bindings (thin bridge only).)pbdoc";
}
