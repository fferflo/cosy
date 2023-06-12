get_filename_component(cosy_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

if(NOT TARGET cosy::cosy)
  find_package(xtl REQUIRED)
  find_package(xtensor REQUIRED)
  find_package(xtensor-blas REQUIRED)

  find_package(PROJ QUIET)

  include("${cosy_CMAKE_DIR}/cosyTargets.cmake")
endif()
