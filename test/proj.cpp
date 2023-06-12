#include <cosy/proj.h>
#include <catch2/catch_test_macros.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xadapt.hpp>

TEST_CASE("proj transform")
{
  std::shared_ptr<cosy::proj::Context> context = std::make_shared<cosy::proj::Context>();
  cosy::proj::Transformer transform(context, "EPSG:4326", "EPSG:3857");

  xti::vec2d latlon({49.011384, 8.416419});

  REQUIRE(xt::mean(latlon - transform.transform_inverse(transform.transform(latlon)))() < 1e-6);
}

TEST_CASE("proj area_of_use")
{
  std::shared_ptr<cosy::proj::Context> context = std::make_shared<cosy::proj::Context>();
  cosy::proj::CRS crs(context, "EPSG:3857");

  crs.get_area_of_use();
}
