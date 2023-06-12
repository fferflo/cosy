#pragma once

#include <xti/typedefs.h>
#include <xti/util.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <cosy/affine/rotation.h>
#include <cosy/affine/rigid.h>

#ifdef COSY_CEREAL_INCLUDED
#include <cereal/access.hpp>
#include <xti/cereal.h>
#endif

namespace cosy {

template <typename TScalar, size_t TRank>
class ScaledRigid
{
private:
  xti::matXT<TScalar, TRank> m_rotation;
  xti::vecXT<TScalar, TRank> m_translation;
  TScalar m_scale;

public:
  ScaledRigid()
    : m_rotation(xt::eye<TScalar>(TRank))
    , m_scale(1)
  {
    m_translation.fill(0);
  }

  template <typename TScalar2>
  ScaledRigid(const Rigid<TScalar2, TRank>& other)
    : m_rotation(other.get_rotation())
    , m_translation(other.get_translation())
    , m_scale(1)
  {
  }

  template <typename TScalar2>
  ScaledRigid(const Rotation<TScalar2, TRank>& other)
    : m_rotation(other.get_rotation())
    , m_scale(1)
  {
    m_translation.fill(0);
  }

  ScaledRigid(xti::matXT<TScalar, TRank> rotation, xti::vecXT<TScalar, TRank> translation, TScalar scale = 1)
    : m_rotation(rotation)
    , m_translation(translation)
    , m_scale(scale)
  {
  }

  template <bool TDummy = true, typename = std::enable_if_t<TDummy && TRank == 2, void>>
  ScaledRigid(TScalar angle, xti::vecXT<TScalar, TRank> translation, TScalar scale)
    : m_rotation(angle_to_rotation_matrix(angle))
    , m_translation(translation)
    , m_scale(scale)
  {
  }

  template <typename TScalar2>
  ScaledRigid(const ScaledRigid<TScalar2, TRank>& other)
    : m_rotation(other.m_rotation)
    , m_translation(other.m_translation)
    , m_scale(other.m_scale)
  {
  }

  template <typename TScalar2>
  ScaledRigid<TScalar, TRank>& operator=(const ScaledRigid<TScalar2, TRank>& other)
  {
    this->m_rotation = other.m_rotation;
    this->m_translation = other.m_translation;
    this->m_scale = other.m_scale;
    return *this;
  }

  auto transform(xti::vecXT<TScalar, TRank> point) const
  {
    return m_scale * xt::linalg::dot(m_rotation, point) + m_translation;
  }

  template <typename TTensor>
  auto transform_all(TTensor&& points) const
  {
    if (points.shape()[1] != TRank)
    {
      throw std::invalid_argument(XTI_TO_STRING("Points tensor must have shape (n, " << TRank << "), got shape " << xt::adapt(points.shape())));
    }
    return xt::transpose(xt::eval(m_scale * xt::linalg::dot(m_rotation, xt::transpose(xt::eval(std::forward<TTensor>(points)), {1, 0})) + xt::view(m_translation, xt::all(), xt::newaxis())), {1, 0});
  }

  auto transform_inverse(xti::vecXT<TScalar, TRank> point) const
  {
    return xt::linalg::dot(xt::transpose(m_rotation, {1, 0}), point - m_translation) / m_scale;
  }

  template <typename TTensor>
  auto transform_all_inverse(TTensor&& points) const
  {
    if (points.shape()[1] != TRank)
    {
      throw std::invalid_argument(XTI_TO_STRING("Points tensor must have shape (n, " << TRank << "), got shape " << xt::adapt(points.shape())));
    }
    return xt::transpose(xt::eval(xt::linalg::dot(xt::transpose(m_rotation, {1, 0}), xt::transpose(xt::eval(std::forward<TTensor>(points)), {1, 0}) - xt::view(m_translation, xt::all(), xt::newaxis())), {1, 0})) / m_scale;
  }

  ScaledRigid<TScalar, TRank> inverse() const
  {
    ScaledRigid<TScalar, TRank> result;
    result.get_rotation() = xt::transpose(m_rotation, {1, 0});
    result.get_translation() = xt::linalg::dot(result.get_rotation(), -m_translation) / m_scale;
    result.get_scale() = 1 / m_scale;
    return result;
  }

  ScaledRigid<TScalar, TRank>& operator*=(const ScaledRigid<TScalar, TRank>& right)
  {
    m_translation = this->transform(right.get_translation());
    m_rotation = xt::linalg::dot(m_rotation, right.get_rotation());
    m_scale = m_scale * right.m_scale;
    return *this;
  }

  xti::matXT<TScalar, TRank>& get_rotation()
  {
    return m_rotation;
  }

  const xti::matXT<TScalar, TRank>& get_rotation() const
  {
    return m_rotation;
  }

  xti::vecXT<TScalar, TRank>& get_translation()
  {
    return m_translation;
  }

  const xti::vecXT<TScalar, TRank>& get_translation() const
  {
    return m_translation;
  }

  TScalar& get_scale()
  {
    return m_scale;
  }

  const TScalar& get_scale() const
  {
    return m_scale;
  }

  xti::matXT<TScalar, TRank + 1> to_matrix() const
  {
    xti::matXT<TScalar, TRank + 1> result;
    for (int32_t r = 0; r < TRank; r++)
    {
      for (int32_t c = 0; c < TRank; c++)
      {
        result(r, c) = m_rotation(r, c) * m_scale;
      }
      result(r, TRank) = m_translation(r);
      result(TRank, r) = 0;
    }
    result(TRank, TRank) = 1;
    return result;
  }

  template <typename TScalar2, size_t TRank2>
  friend class ScaledRigid;
};

template <typename TScalar, size_t TRank>
ScaledRigid<TScalar, TRank> operator*(const ScaledRigid<TScalar, TRank>& left, const ScaledRigid<TScalar, TRank>& right)
{
  return ScaledRigid<TScalar, TRank>(xt::linalg::dot(left.get_rotation(), right.get_rotation()), left.transform(right.get_translation()), left.get_scale() * right.get_scale());
}

template <typename TScalar, size_t TRank>
ScaledRigid<TScalar, TRank> operator/(const ScaledRigid<TScalar, TRank>& left, const ScaledRigid<TScalar, TRank>& right)
{
  return left * right.inverse();
}

template <typename TScalar, size_t TRank>
std::ostream& operator<<(std::ostream& stream, const ScaledRigid<TScalar, TRank>& transform)
{
  return stream << "ScaledRigid(t=" << transform.get_translation() << " R=" << transform.get_rotation() << " s=" << transform.get_scale() << ")";
}

#ifdef COSY_CEREAL_INCLUDED
template <typename TArchive, typename TScalar, size_t TRank>
void save(TArchive& archive, const cosy::ScaledRigid<TScalar, TRank>& transform)
{
  archive(transform.get_rotation());
  archive(transform.get_translation());
  archive(transform.get_scale());
}

template <typename TArchive, typename TScalar, size_t TRank>
void load(TArchive& archive, cosy::ScaledRigid<TScalar, TRank>& transform)
{
  archive(transform.get_rotation());
  archive(transform.get_translation());
  archive(transform.get_scale());
}
#endif

} // cosy
