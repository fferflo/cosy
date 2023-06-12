#include <cereal/archives/binary.hpp>
#include <cosy/affine.h>
#include <cosy/proj.h>
#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <cereal/types/memory.hpp>

#ifdef WITH_PROJ
TEST_CASE("cereal proj crs")
{
  std::string buffer;

  std::shared_ptr<cosy::proj::CRS> crs1 = std::make_shared<cosy::proj::CRS>("EPSG:3857");
  {
    std::ostringstream stream;
    cereal::BinaryOutputArchive archive(stream);
    archive(crs1);
    buffer = stream.str();
  }
  {
    std::shared_ptr<cosy::proj::CRS> crs2;
    std::istringstream stream;
    stream.str(buffer);
    cereal::BinaryInputArchive archive(stream);
    archive(crs2);
    REQUIRE(crs1->get_description() == crs2->get_description());
  }
}

TEST_CASE("cereal proj transformer")
{
  std::string buffer;

  std::shared_ptr<cosy::proj::Transformer> transformer1 = std::make_shared<cosy::proj::Transformer>("EPSG:4326", "EPSG:3857");
  {
    std::ostringstream stream;
    cereal::BinaryOutputArchive archive(stream);
    archive(transformer1);
    buffer = stream.str();
  }
  {
    std::shared_ptr<cosy::proj::Transformer> transformer2;
    std::istringstream stream;
    stream.str(buffer);
    cereal::BinaryInputArchive archive(stream);
    archive(transformer2);
    REQUIRE(transformer1->get_from_crs()->get_description() == transformer2->get_from_crs()->get_description());
    REQUIRE(transformer1->get_to_crs()->get_description() == transformer2->get_to_crs()->get_description());
  }
}
#endif

TEST_CASE("cereal cosy")
{
  std::string buffer;

  cosy::Rigid<double, 2> t1(xti::mat2d({{0.0, 1.0}, {1.0, 0.0}}), xti::vec2d({13.0, -14.5}));
  {
    std::ostringstream stream;
    cereal::BinaryOutputArchive archive(stream);
    archive(t1);
    buffer = stream.str();
  }
  {
    cosy::Rigid<double, 2> t2;
    std::istringstream stream;
    stream.str(buffer);
    cereal::BinaryInputArchive archive(stream);
    archive(t2);
    REQUIRE(t1.get_rotation() == t2.get_rotation());
    REQUIRE(t1.get_translation() == t2.get_translation());
  }
}
