#include <pybind11/pybind11.h>
#include <cosy/geo.h>
#include <xtensor-python/pytensor.hpp>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_proj(py::module& proj);

PYBIND11_MODULE(backend, m)
{
#ifdef COSY_BUILD_PROJ
  auto proj = m.def_submodule("proj");
  register_proj(proj);
#endif

  py::class_<cosy::NamedAxes<2>>(m, "NamedAxes2")
    .def(py::init([](std::pair<std::string, std::string> axis1, std::pair<std::string, std::string> axis2){
        return cosy::NamedAxes<2>({{axis1.first, axis1.second}, {axis2.first, axis2.second}});
      }),
      py::arg("axis1"),
      py::arg("axis2")
    )
    .def("__getitem__", [](const cosy::NamedAxes<2>& axes, size_t index){return axes[index];})
  ;
  m.def("NamedAxesTransformation", [](cosy::NamedAxes<2> axes1, cosy::NamedAxes<2> axes2) -> cosy::Rotation<double, 2>{
    return cosy::NamedAxesTransformation<double, 2>(axes1, axes2);
  });

  auto geo = m.def_submodule("geo");
  // geo.def("move_from_latlon", &cosy::geo::move_from_latlon,
  //   py::arg("latlon"),
  //   py::arg("bearing"),
  //   py::arg("distance")
  // );
  py::class_<cosy::geo::CompassAxes, cosy::NamedAxes<2>>(geo, "CompassAxes")
    .def(py::init<std::string, std::string>(),
      py::arg("axis1"),
      py::arg("axis2")
    )
    .def_property_readonly("axis1", [](const cosy::geo::CompassAxes& axes){return axes[0];})
    .def_property_readonly("axis2", [](const cosy::geo::CompassAxes& axes){return axes[1];})
  ;
}
