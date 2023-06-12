#ifdef COSY_BUILD_PROJ
#include <pybind11/pybind11.h>
#include <cosy/proj.h>
#include <cosy/python.h>
#include <xtensor-python/pytensor.hpp>
#include <filesystem>

namespace py = pybind11;

// Ignore multithreading issues, keep one context for all threads
std::shared_ptr<cosy::proj::Context> proj_context = std::make_shared<cosy::proj::Context>(false);

std::shared_ptr<cosy::proj::Transformer> epsg4326_to_epsg3857 = std::make_shared<cosy::proj::Transformer>(
  std::make_shared<cosy::proj::CRS>(proj_context, "epsg:4326"),
  std::make_shared<cosy::proj::CRS>(proj_context, "epsg:3857")
);
std::shared_ptr<cosy::proj::Transformer> epsg3857_to_epsg4326 = epsg4326_to_epsg3857->inverse();

void register_proj(py::module& proj)
{
  py::class_<cosy::proj::CRS, std::shared_ptr<cosy::proj::CRS>>(proj, "CRS")
    .def(py::init([](std::string desc){
        return std::make_shared<cosy::proj::CRS>(proj_context, desc);
      }),
      py::arg("desc")
    )
    .def("get_vector", &cosy::proj::CRS::get_vector,
      py::arg("name")
    )
    .def(py::pickle(
      [](const cosy::proj::CRS& x) { // __getstate__
        return py::make_tuple(
          x.get_description()
        );
      },
      [](py::tuple t) { // __setstate__
        if (t.size() != 1)
        {
          throw std::runtime_error("Invalid state");
        }
        return std::make_shared<cosy::proj::CRS>(
          proj_context,
          t[0].cast<std::string>()
        );
      }
    ))
  ;

  py::class_<cosy::proj::Transformer, std::shared_ptr<cosy::proj::Transformer>>(proj, "Transformer")
    .def(py::init([](std::string from_crs, std::string to_crs){
        return std::make_shared<cosy::proj::Transformer>(proj_context, from_crs, to_crs);
      }),
      py::arg("from_crs"),
      py::arg("to_crs")
    )
    .def(py::init([](std::shared_ptr<cosy::proj::CRS> from_crs, std::string to_crs){
        return std::make_shared<cosy::proj::Transformer>(proj_context, from_crs, to_crs);
      }),
      py::arg("from_crs"),
      py::arg("to_crs")
    )
    .def(py::init([](std::string from_crs, std::shared_ptr<cosy::proj::CRS> to_crs){
        return std::make_shared<cosy::proj::Transformer>(proj_context, from_crs, to_crs);
      }),
      py::arg("from_crs"),
      py::arg("to_crs")
    )
    .def(py::init([](std::shared_ptr<cosy::proj::CRS> from_crs, std::shared_ptr<cosy::proj::CRS> to_crs){
        return std::make_shared<cosy::proj::Transformer>(proj_context, from_crs, to_crs);
      }),
      py::arg("from_crs"),
      py::arg("to_crs")
    )
    .def("transform", [](const cosy::proj::Transformer& transformer, xti::vec2d coords){
        return transformer.transform(coords);
      },
      py::arg("coords")
    )
    .def("transform_inverse", [](const cosy::proj::Transformer& transformer, xti::vec2d coords){
        return transformer.transform_inverse(coords);
      },
      py::arg("coords")
    )
    .def("transform_angle", [](const cosy::proj::Transformer& transformer, double angle){
        return transformer.transform_angle(angle);
      },
      py::arg("angle")
    )
    .def("transform_angle_inverse", [](const cosy::proj::Transformer& transformer, double angle){
        return transformer.transform_angle_inverse(angle);
      },
      py::arg("angle")
    )
    .def("__call__", [](const cosy::proj::Transformer& transformer, xti::vec2d coords){
        return transformer.transform(coords);
      },
      py::arg("coords")
    )
    .def("inverse", &cosy::proj::Transformer::inverse)
    .def_property_readonly("from_crs", &cosy::proj::Transformer::get_from_crs)
    .def_property_readonly("to_crs", &cosy::proj::Transformer::get_to_crs)
    .def(py::pickle(
      [](const cosy::proj::Transformer& x) { // __getstate__
        return py::make_tuple(
          x.get_from_crs(),
          x.get_to_crs()
        );
      },
      [](py::tuple t) { // __setstate__
        if (t.size() != 2)
        {
          throw std::runtime_error("Invalid state");
        }
        return std::make_shared<cosy::proj::Transformer>(
          proj_context,
          t[0].cast<std::shared_ptr<cosy::proj::CRS>>(),
          t[1].cast<std::shared_ptr<cosy::proj::CRS>>()
        );
      }
    ))
  ;

  proj.def("eastnorthmeters_at_latlon_to_epsg3857", [](xti::vec2d latlon){
      return cosy::proj::eastnorthmeters_at_latlon_to_epsg3857(latlon, *epsg4326_to_epsg3857);
    },
    py::arg("latlon")
  );
  proj.def("geopose_to_epsg3857", [](xti::vec2d latlon, double bearing){
      return cosy::proj::geopose_to_epsg3857(latlon, bearing, *epsg4326_to_epsg3857);
    },
    py::arg("latlon"),
    py::arg("bearing")
  );

  proj.attr("__setattr__")("epsg4326_to_epsg3857", epsg4326_to_epsg3857);
  proj.attr("__setattr__")("epsg3857_to_epsg4326", epsg3857_to_epsg4326);
}
#endif
