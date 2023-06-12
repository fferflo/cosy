#pragma once

#include <xtensor-python/pytensor.hpp>
#include <pybind11/pybind11.h>

#include <cosy/affine.h>
#include <cosy/camera.h>

namespace py = pybind11;

namespace pybind11::detail {

template <typename TElementType, size_t TRank>
struct type_caster<cosy::Rigid<TElementType, TRank>>
{
public:
  using Type = cosy::Rigid<TElementType, TRank>;
  PYBIND11_TYPE_CASTER(Type, const_name("Rigid"));

  bool load(py::handle src, bool)
  {
    if (!src || !py::hasattr(src, "rotation") || !py::hasattr(src, "translation"))
    {
      return false;
    }

    xt::xtensor<TElementType, 2> rotation = src.attr("rotation").cast<xt::xtensor<TElementType, 2>>(); // TODO: assert batchsize == ()
    xt::xtensor<TElementType, 1> translation = src.attr("translation").cast<xt::xtensor<TElementType, 1>>();

    value = cosy::Rigid<TElementType, TRank>(rotation, translation);

    return true;
  }

  static py::handle cast(cosy::Rigid<TElementType, TRank> src, py::return_value_policy /* policy */, py::handle /* parent */)
  {
    py::module_ cosy = py::module_::import("cosy");
    py::object dest = cosy.attr("Rigid")(src.get_rotation(), src.get_translation());
    return dest.release();
  }
};

template <typename TElementType, size_t TRank>
struct type_caster<cosy::ScaledRigid<TElementType, TRank>>
{
public:
  using Type = cosy::ScaledRigid<TElementType, TRank>;
  PYBIND11_TYPE_CASTER(Type, const_name("ScaledRigid"));

  bool load(py::handle src, bool)
  {
    if (!src || !py::hasattr(src, "rotation") || !py::hasattr(src, "translation"))
    {
      return false;
    }

    xt::xtensor<TElementType, 2> rotation = src.attr("rotation").cast<xt::xtensor<TElementType, 2>>();
    xt::xtensor<TElementType, 1> translation = src.attr("translation").cast<xt::xtensor<TElementType, 1>>();
    TElementType scale = 1;
    if (py::hasattr(src, "scale"))
    {
      scale = src.attr("scale").cast<TElementType>();
    }

    value = cosy::ScaledRigid<TElementType, TRank>(rotation, translation, scale);

    return true;
  }

  static py::handle cast(cosy::ScaledRigid<TElementType, TRank> src, py::return_value_policy /* policy */, py::handle /* parent */)
  {
    py::module_ cosy = py::module_::import("cosy");
    py::object dest = cosy.attr("ScaledRigid")(src.get_rotation(), src.get_translation(), src.get_scale());
    return dest.release();
  }
};

template <typename TElementType, size_t TRank>
struct type_caster<cosy::PinholeK<TElementType, TRank>> // TODO: add python class for PinholeK, PinholeFC
{
public:
  using Type = cosy::PinholeK<TElementType, TRank>;
  PYBIND11_TYPE_CASTER(Type, const_name("PinholeK"));

  bool load(py::handle src, bool)
  {
    if (!src)
    {
      return false;
    }
    try
    {
      xti::matXT<TElementType, TRank> intr = src.cast<xti::matXT<TElementType, TRank>>();
      value = cosy::PinholeK<TElementType, TRank>(std::move(intr));
    }
    catch (py::cast_error e)
    {
      return false;
    }

    return true;
  }

  static py::handle cast(cosy::PinholeK<TElementType, TRank> src, py::return_value_policy /* policy */, py::handle /* parent */)
  {
    py::object dest = py::cast(src.get_matrix());
    return dest.release();
  }
};

} // end of ns pybind11::detail
