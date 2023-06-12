#pragma once

#include <xti/typedefs.h>
#include <xti/util.h>
#include <cosy/affine.h>

namespace cosy::geo {

static const double EARTH_RADIUS_METERS = 6.378137e6;

class CompassAxes : public NamedAxes<2>
{
public:
  static std::string opposite_of(std::string direction)
  {
    if (direction == "north")
    {
      return "south";
    }
    else if (direction == "south")
    {
      return "north";
    }
    else if (direction == "east")
    {
      return "west";
    }
    else if (direction == "west")
    {
      return "east";
    }
    else
    {
      throw std::invalid_argument(XTI_TO_STRING("Got invalid compass direction " << direction));
    }
  }

  CompassAxes()
  {
  }

  CompassAxes(std::string axis1, std::string axis2)
    : NamedAxes<2>({{axis1, opposite_of(axis1)}, {axis2, opposite_of(axis2)}})
  {
  }
};

static const CompassAxes epsg4326_axes("north", "east");

inline xti::vec2d meters_per_deg_at_latlon(xti::vec2d latlon)
{
  static_assert(xti::is_xtensor_v<xti::vec2d>, "");
  xti::vec2d latlon_rad = cosy::radians(latlon);
  double meter_per_deg_lat = 111132.954 - 559.822 * std::cos(2 * latlon_rad(0)) + 1.175 * std::cos(4 * latlon_rad(0));
  double meter_per_deg_lon = 111132.954 * std::cos(latlon_rad(0));
  return xti::vec2d({meter_per_deg_lat, meter_per_deg_lon});
}

inline xti::vec2d move_from_latlon(xti::vec2d latlon, double bearing, double distance)
{
  static const double pi = xt::numeric_constants<double>::PI;
  bearing = cosy::radians(bearing);

  xti::vec2d latlon_rad = cosy::radians(latlon);

  double angular_distance = distance / EARTH_RADIUS_METERS;

  double target_lat = std::asin(
      std::sin(latlon_rad[0]) * std::cos(angular_distance) +
      std::cos(latlon_rad[0]) * std::sin(angular_distance) * std::cos(bearing)
  );
  double target_lon = latlon_rad[1] + std::atan2(
      std::sin(bearing) * std::sin(angular_distance) * std::cos(latlon_rad[0]),
      std::cos(angular_distance) - std::sin(latlon_rad[0]) * std::sin(target_lat)
  );
  target_lon = cosy::normalize_angle(target_lon);

  return xti::vec2d({target_lat, target_lon}) * (180.0 / pi);
}

} // end of ns cosy::geo
