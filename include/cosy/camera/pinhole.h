#pragma once

#include <xti/typedefs.h>
#include <xti/util.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>

#ifdef COSY_CEREAL_INCLUDED
#include <cereal/access.hpp>
#include <xti/cereal.h>
#endif

namespace cosy {

template <typename TTensor>
auto to_homogeneous(TTensor&& tensor)
{
  std::vector<size_t> shape = tensor.shape();
  shape[shape.size() - 1] = 1;
  return xt::concatenate(xt::xtuple(std::forward<TTensor>(tensor), xt::ones<xti::elementtype_t<TTensor&&>>(shape)), shape.size() - 1);
}

template <typename TTensor>
auto from_homogeneous(TTensor&& tensor)
{
  // auto shared_tensor = xt::make_xshared(std::forward<TTensor>(tensor));
  // return xt::view(shared_tensor, xt::range(0, -1)) / xt::view(shared_tensor, -1);
  auto last_dim = xt::eval(xt::view(tensor, -1));
  return xt::view(std::forward<TTensor>(tensor), xt::range(0, -1)) / std::move(last_dim);
}

template <typename TTensor>
auto from_homogeneous_all(TTensor&& tensor)
{
  // auto shared_tensor = xt::make_xshared(std::forward<TTensor>(tensor));
  // return xt::view(shared_tensor, xt::range(0, -1)) / xt::view(shared_tensor, -1);
  auto last_dim = xt::eval(xt::view(tensor, xt::all(), -1));
  return xt::view(std::forward<TTensor>(tensor), xt::all(), xt::range(0, -1)) / xt::view(std::move(last_dim), xt::all(), xt::newaxis());
}

template <typename TScalar, size_t TRank>
class PinholeK;

template <typename TScalar, size_t TRank>
class PinholeFC
{
public:
  PinholeFC(xti::vecXT<TScalar, TRank - 1> f, xti::vecXT<TScalar, TRank - 1> c)
    : m_f(f)
    , m_1_over_f(1 / f)
    , m_c(c)
  {
  }

  PinholeFC()
  {
    m_f.fill(1);
    m_1_over_f.fill(1);
    m_c.fill(0);
  }

  template <typename TScalar2>
  PinholeFC(const PinholeFC<TScalar2, TRank>& other)
    : m_f(other.m_f)
    , m_1_over_f(other.m_1_over_f)
    , m_c(other.m_c)
  {
  }

  template <typename TScalar2>
  PinholeFC<TScalar, TRank>& operator=(const PinholeFC<TScalar2, TRank>& other)
  {
    m_f = other.m_f;
    m_1_over_f = other.m_1_over_f;
    m_c = other.m_c;
    return *this;
  }

  xti::vecXT<TScalar, TRank - 1> transform(xti::vecXT<TScalar, TRank> point) const
  {
    return from_homogeneous(point) * m_f + m_c;
  }

  template <typename TTensor>
  auto transform_all(TTensor&& points) const
  {
    if (points.shape()[1] != TRank)
    {
      throw std::invalid_argument(XTI_TO_STRING("Points tensor must have shape (n, " << TRank << "), got shape " << xt::adapt(points.shape())));
    }
    return from_homogeneous_all(std::forward<TTensor>(points)) * xt::view(m_f, xt::newaxis(), xt::all()) + xt::view(m_c, xt::newaxis(), xt::all());
  }

  xti::vecXT<TScalar, TRank> transform_inverse(xti::vecXT<TScalar, TRank - 1> point) const
  {
    return to_homogeneous((point - m_c) * m_1_over_f);
  }

  template <typename TTensor>
  auto transform_all_inverse(TTensor&& points) const
  {
    if (points.shape()[1] != TRank - 1)
    {
      throw std::invalid_argument(XTI_TO_STRING("Points tensor must have shape (n, " << TRank - 1 << "), got shape " << xt::adapt(points.shape())));
    }
    return to_homogeneous((std::forward<TTensor>(points) - xt::view(m_c, xt::newaxis(), xt::all())) * xt::view(m_1_over_f, xt::newaxis(), xt::all()));
  }

  const xti::vecXT<TScalar, TRank - 1>& get_focal_lengths() const
  {
    return m_f;
  }

  const xti::vecXT<TScalar, TRank - 1>& get_principal_point() const
  {
    return m_c;
  }

  template <typename TScalar2, size_t TRank2>
  friend class PinholeFC;
  template <typename TScalar2, size_t TRank2>
  friend class PinholeK;

private:
  xti::vecXT<TScalar, TRank - 1> m_f;
  xti::vecXT<TScalar, TRank - 1> m_1_over_f;
  xti::vecXT<TScalar, TRank - 1> m_c;
};

template <typename TScalar, size_t TRank>
std::ostream& operator<<(std::ostream& stream, const PinholeFC<TScalar, TRank>& p)
{
  stream << "Pinhole(f=" << p.get_focal_lengths() << " c=" << p.get_principal_point() << ")";
  return stream;
}

#ifdef COSY_CEREAL_INCLUDED
template <typename TArchive, typename TScalar, size_t TRank>
void save(TArchive& archive, const cosy::PinholeFC<TScalar, TRank>& projection)
{
  archive(projection.get_focal_lengths());
  archive(projection.get_principal_point());
}

template <typename TArchive, typename TScalar, size_t TRank>
void save(TArchive& archive, cosy::PinholeFC<TScalar, TRank>& projection)
{
  xti::vecXT<TScalar, TRank - 1> f;
  archive(f);
  xti::vecXT<TScalar, TRank - 1> c;
  archive(c);
  projection = cosy::PinholeFC<TScalar, TRank>(projection);
}
#endif

template <typename TScalar, size_t TRank>
class PinholeK
{
public:
  PinholeK(xti::matXT<TScalar, TRank> k)
    : m_k(k)
    , m_k_inv(xt::linalg::inv(k))
  {
  }

  PinholeK()
    : PinholeK(xt::eye<TScalar>(TRank))
  {
  }

  template <typename TScalar2>
  PinholeK(const PinholeK<TScalar2, TRank>& other)
    : m_k(other.m_k)
    , m_k_inv(other.m_k_inv)
  {
  }

  template <typename TScalar2>
  PinholeK(const PinholeFC<TScalar2, TRank>& other)
  {
    *this = other;
  }

  template <typename TScalar2>
  PinholeK<TScalar, TRank>& operator=(const PinholeK<TScalar2, TRank>& other)
  {
    m_k = other.m_k;
    m_k_inv = other.m_k_inv;
    return *this;
  }

  template <typename TScalar2>
  PinholeK<TScalar, TRank>& operator=(const PinholeFC<TScalar2, TRank>& other)
  {
    m_k = 0;
    for (size_t i = 0; i < TRank - 1; i++)
    {
      m_k(i, i) = other.m_f(i);
      m_k(i, TRank - 1) = other.m_c(i);
    }
    m_k(TRank - 1, TRank - 1) = 1;

    m_k_inv = 0;
    for (size_t i = 0; i < TRank - 1; i++)
    {
      m_k_inv(i, i) = other.m_1_over_f(i);
      m_k_inv(i, TRank - 1) = -other.m_c(i) * other.m_1_over_f(i);
    }
    m_k_inv(TRank - 1, TRank - 1) = 1;

    return *this;
  }

  xti::vecXT<TScalar, TRank - 1> transform(xti::vecXT<TScalar, TRank> point) const
  {
    return from_homogeneous(xt::linalg::dot(m_k, point));
  }

  template <typename TTensor>
  auto transform_all(TTensor&& points) const
  {
    if (points.shape()[1] != TRank)
    {
      throw std::invalid_argument(XTI_TO_STRING("Points tensor must have shape (n, " << TRank << "), got shape " << xt::adapt(points.shape())));
    }
    return from_homogeneous_all(xt::transpose(xt::eval(xt::linalg::dot(m_k, xt::transpose(xt::eval(std::forward<TTensor>(points)), {1, 0}))), {1, 0}));
  }

  xti::vecXT<TScalar, TRank> transform_inverse(xti::vecXT<TScalar, TRank - 1> point) const
  {
    return xt::linalg::dot(m_k_inv, to_homogeneous(point));
  }

  template <typename TTensor>
  auto transform_all_inverse(TTensor&& points) const
  {
    if (points.shape()[1] != TRank - 1)
    {
      throw std::invalid_argument(XTI_TO_STRING("Points tensor must have shape (n, " << TRank - 1 << "), got shape " << xt::adapt(points.shape())));
    }
    return xt::transpose(xt::eval(xt::linalg::dot(m_k_inv, xt::transpose(xt::eval(to_homogeneous(std::forward<TTensor>(points))), {1, 0})), {1, 0}));
  }

  const xti::matXT<TScalar, TRank>& get_matrix() const
  {
    return m_k;
  }

  template <typename TScalar2, size_t TRank2>
  friend class PinholeK;

private:
  xti::matXT<TScalar, TRank> m_k;
  xti::matXT<TScalar, TRank> m_k_inv;
};

template <typename TScalar, size_t TRank>
std::ostream& operator<<(std::ostream& stream, const PinholeK<TScalar, TRank>& p)
{
  return stream << "Pinhole(K=" << p.get_matrix() << ")";
}

#ifdef COSY_CEREAL_INCLUDED
template <typename TArchive, typename TScalar, size_t TRank>
void save(TArchive& archive, const cosy::PinholeK<TScalar, TRank>& projection)
{
  archive(projection.get_matrix());
}

template <typename TArchive, typename TScalar, size_t TRank>
void load(TArchive& archive, cosy::PinholeK<TScalar, TRank>& projection)
{
  xti::matXT<TScalar, TRank> matrix;
  archive(matrix);
  projection = cosy::PinholeK<TScalar, TRank>(matrix);
}
#endif

} // end of ns cosy
