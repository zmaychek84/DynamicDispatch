# Copyright (c) 2025 Advanced Micro Devices, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
  message(STATUS "Using GTest from FetchContent")
  # While we can use an installed version of GTest and use find_package to get it,
  # GTest's official guide recommends this approach of building the library with
  # the same compile options as the executable being tested instead of linking
  # against a precompiled library.
  FetchContent_Declare(
    googletest GIT_REPOSITORY "https://github.com/google/googletest"
    GIT_TAG "v1.14.0"
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE INTERNAL "")

  FetchContent_MakeAvailable(googletest)

  # move all include directories to system directories
  list(APPEND gtest_targets gtest gtest_main gmock)
  foreach(target ${gtest_targets})
    get_target_property(INCLUDE_DIRS ${target} INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(
      ${target} PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                           "${INCLUDE_DIRS}"
    )
  endforeach()
endif()
