cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(cosy LANGUAGES CXX)

enable_testing()

include(GNUInstallDirs)
include(ExternalProject)

set(PACKAGE_NAME cosy)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/${PACKAGE_NAME}/cmake)
set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR}/${PACKAGE_NAME})


# Base library
add_library(cosy INTERFACE)
target_include_directories(cosy
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cosy INTERFACE cxx_std_17)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(xtensor-interfaces REQUIRED)
target_link_libraries(cosy INTERFACE
  xtensor
  xtensor::use_xsimd
  xtensor-blas
  xtensor-interfaces::base
)

install(
  TARGETS cosy
  EXPORT install_targets
)

######################## INTERFACES ########################

# PROJ
if(NOT DEFINED COSY_PROJ)
  find_package(PROJ QUIET)
  option(COSY_PROJ "Build proj interface" ${PROJ_FOUND})
endif()
if(COSY_PROJ)
  find_package(CURL REQUIRED)
  find_package(PROJ REQUIRED)
  add_library(proj INTERFACE)
  target_link_libraries(proj INTERFACE cosy PROJ::proj CURL::libcurl)
  install(
    TARGETS proj
    EXPORT install_targets
  )
endif()

# Cereal
if(NOT DEFINED COSY_CEREAL)
  find_package(cereal QUIET)
  option(COSY_CEREAL "Build cereal interface" ${cereal_FOUND})
endif()
if(COSY_CEREAL)
  find_package(cereal REQUIRED)
  add_library(cereal INTERFACE)
  target_link_libraries(cereal INTERFACE cosy cereal::cereal)
  target_compile_definitions(cereal INTERFACE COSY_CEREAL_INCLUDED)
  install(
    TARGETS cereal
    EXPORT install_targets
  )
endif()



######################## TESTS ########################

add_custom_target(tests)
add_subdirectory(test)



######################## PYTHON ########################

option(COSY_BUILD_PYTHON_INTERFACE "Build python interface" ON)
if(COSY_BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()



######################## INSTALL ########################

# Install headers
set(INSTALL_HEADERS_AS_SYMLINK OFF CACHE BOOL "Installs headers as a symlink to the source directory instead of copying all files to install directory")
if(INSTALL_HEADERS_AS_SYMLINK)
  install(CODE "file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})")
  install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/include/cosy ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/cosy)")
  install(CODE "message(\"-- Installed symlink: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/cosy -> ${CMAKE_CURRENT_SOURCE_DIR}/include/cosy\")")
else()
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# Install targets file
install(EXPORT install_targets
  FILE
    ${PACKAGE_NAME}Targets.cmake
  NAMESPACE
    ${PACKAGE_NAME}::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)
# Install ${PACKAGE_NAME}Config.cmake
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PACKAGE_NAME}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)
