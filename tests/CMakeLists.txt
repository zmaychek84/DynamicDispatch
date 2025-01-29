# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.

find_package(GTest REQUIRED)
include(GoogleTest)
include(DdTesting)

add_subdirectory(cpp)
add_subdirectory(dd_runner)
