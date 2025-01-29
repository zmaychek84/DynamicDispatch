# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

find_package(aiebu CONFIG QUIET)
if(NOT aiebu_FOUND)
  message(STATUS "Using aiebu from FetchContent")
  FetchContent_Declare(
    AIEBU GIT_REPOSITORY "https://gitenterprise.xilinx.com/XRT/aiebu.git"
    GIT_TAG "main"
  )

  FetchContent_MakeAvailable(AIEBU)
  set(aiebu_FOUND TRUE)
  if(NOT TARGET aiebu)
    add_library(aiebu::aiebu_static ALIAS aiebu_static)
  endif()
  install(TARGETS aiebu EXPORT aiebu-targets)
  install(
    EXPORT aiebu-targets
    FILE AIEBUConfig.cmake
    NAMESPACE aiebu::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aiebu
  )
endif()
