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
set(ZLIB_USE_STATIC_LIBS ON)
find_package(ZLIB QUIET)
if(NOT ZLIB_FOUND)
  message(STATUS "Using ZLIB from FetchContent")
  set(ZLIB_BUILD_EXAMPLES OFF CACHE INTERNAL "")
  FetchContent_Declare(
    ZLIB GIT_REPOSITORY "https://github.com/madler/zlib.git" GIT_TAG v1.3.1
  )
  FetchContent_MakeAvailable(ZLIB)
  if(NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
  endif()
  target_include_directories(
    zlibstatic PUBLIC ${zlib_BINARY_DIR} ${zlib_SOURCE_DIR}
  ) # weird bug
  set(ZLIB_FOUND TRUE)
endif()
