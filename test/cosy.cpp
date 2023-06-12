#include <cosy/affine.h>
#include <catch2/catch_test_macros.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xadapt.hpp>

TEST_CASE("NamedAxes")
{
  {
    cosy::NamedAxes<2> axes1({{"north", "south"}, {"east", "west"}});
    cosy::NamedAxes<2> axes2({{"east", "west"}, {"south", "north"}});
    cosy::NamedAxesTransformation<double, 2> axes1_to_axes2(axes1, axes2);

    REQUIRE(xt::abs(xt::mean(xti::vec2d({0, -1}) - axes1_to_axes2.transform(xti::vec2d({1, 0}))))() < 1e-6);
    REQUIRE(xt::abs(xt::mean(xti::vec2d({0, 1}) - axes1_to_axes2.transform(xti::vec2d({-1, 0}))))() < 1e-6);
    REQUIRE(xt::abs(xt::mean(xti::vec2d({1, 0}) - axes1_to_axes2.transform(xti::vec2d({0, 1}))))() < 1e-6);
    REQUIRE(xt::abs(xt::mean(xti::vec2d({-1, 0}) - axes1_to_axes2.transform(xti::vec2d({0, -1}))))() < 1e-6);
  }

  {
    cosy::NamedAxes<2> axes1({{"east", "west"}, {"north", "south"}});
    cosy::NamedAxes<2> axes2({{"south", "north"}, {"east", "west"}});
    cosy::NamedAxesTransformation<double, 2> axes1_to_axes2(axes1, axes2);

    REQUIRE(xt::abs(xt::mean(xti::vec2d({0, 1}) - axes1_to_axes2.transform(xti::vec2d({1, 0}))))() < 1e-6);
    REQUIRE(xt::abs(xt::mean(xti::vec2d({0, -1}) - axes1_to_axes2.transform(xti::vec2d({-1, 0}))))() < 1e-6);
    REQUIRE(xt::abs(xt::mean(xti::vec2d({-1, 0}) - axes1_to_axes2.transform(xti::vec2d({0, 1}))))() < 1e-6);
    REQUIRE(xt::abs(xt::mean(xti::vec2d({1, 0}) - axes1_to_axes2.transform(xti::vec2d({0, -1}))))() < 1e-6);
  }
}

TEST_CASE("quaternion")
{
  xti::vec4d q1 = cosy::axisangle_to_quaternion(xti::vec3d({1.0, 0.0, 0.0}), 0.0);
  xti::vec4d q2 = cosy::axisangle_to_quaternion(xti::vec3d({1.0, 0.0, 0.0}), xt::numeric_constants<double>::PI / 2);
  xti::vec4d q3 = cosy::slerp(q1, q2, 0.5);
  xti::vec4d q4 = cosy::axisangle_to_quaternion(xti::vec3d({1.0, 0.0, 0.0}), xt::numeric_constants<double>::PI / 4);
  REQUIRE(xt::abs(xt::mean(q3 - q4))() < 1e-6);

  xti::vec4d q5 = cosy::axisangle_to_quaternion(xti::vec3d({1.0, -2.0, 1.4}), 2.0);
  q5 = cosy::rotation_matrix_to_quaternion(cosy::quaternion_to_rotation_matrix(q5));
  xti::vec4d q6 = cosy::rotation_matrix_to_quaternion(cosy::quaternion_to_rotation_matrix(q5));
  REQUIRE(xt::abs(xt::mean(q5 - q6))() < 1e-6);
}

TEST_CASE("slerp")
{
  cosy::Rigid<double, 3> transform1(
      cosy::axisangle_to_rotation_matrix(xti::vec3d({1.0, -2.0, 1.4}), 2.0),
      xti::vec3d({1.0, -2.0, 3.0})
  );
  cosy::Rigid<double, 3> transform2(
      cosy::axisangle_to_rotation_matrix(xti::vec3d({-1.0, 1.0, 2.3}), -1.0),
      xti::vec3d({3.0, -2.54, 1.3})
  );

  REQUIRE(xt::abs(xt::mean(cosy::slerp(transform1, transform2, 0.0).to_matrix() - transform1.to_matrix()))() < 1e-6);
  REQUIRE(xt::abs(xt::mean(cosy::slerp(transform1, transform2, 1.0).to_matrix() - transform2.to_matrix()))() < 1e-6);
}
