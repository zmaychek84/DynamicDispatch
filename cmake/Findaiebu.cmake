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

find_package(aiebu CONFIG QUIET)
if((NOT aiebu_FOUND) AND WIN32)
  message(STATUS "Using aiebu from FetchContent")

  FetchContent_Declare(
    AIEBU
    GIT_REPOSITORY "https://gitenterprise.xilinx.com/XRT/aiebu.git"
    GIT_TAG "main"
    GIT_SUBMODULES_RECURSE TRUE
  )

  set(AIEBU_FULL OFF)
  set(AIEBU_GIT_SUBMODULE True)
  set(AIEBU_MSVC_LEGACY_LINKING True)
  set(AIEBU_AIE_RT_BIN_DIR ${CMAKE_BINARY_DIR}/_deps/aiert-build)
  set(AIEBU_AIE_RT_HEADER_DIR ${CMAKE_BINARY_DIR}/_deps/aiert-build/include)
  FetchContent_MakeAvailable(AIEBU)
  add_library(AIEBU::aiebu_static ALIAS aiebu_static)
  set(aiebu_FOUND TRUE)
  file(
    COPY ${aiebu_SOURCE_DIR}/src/cpp/aiebu/src/include/aiebu.h
         ${aiebu_SOURCE_DIR}/src/cpp/aiebu/src/include/aiebu_assembler.h
         ${aiebu_SOURCE_DIR}/src/cpp/aiebu/src/include/aiebu_error.h
    DESTINATION ${aiebu_BINARY_DIR}/include/
  )
  set(AIEBU_INCLUDE_DIRS ${aiebu_BINARY_DIR}/include/)
endif()
