# Copyright (c) 2024 Advanced Micro Devices, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

message("-- Loading Dependencies URLs ...")

if(CMAKE_VERSION VERSION_GREATER "3.24")
  cmake_policy(SET CMP0135 NEW)
endif()

file(STRINGS ${PROJECT_SOURCE_DIR}/cmake/deps.txt ONNXRUNTIME_DEPS_LIST)
foreach(ONNXRUNTIME_DEP IN LISTS ONNXRUNTIME_DEPS_LIST)
  # Lines start with "#" are comments
  if(NOT ONNXRUNTIME_DEP MATCHES "^#")
    # The first column is name
    list(POP_FRONT ONNXRUNTIME_DEP ONNXRUNTIME_DEP_NAME)
    # The second column is URL
    # The URL below may be a local file path or an HTTPS URL
    list(POP_FRONT ONNXRUNTIME_DEP ONNXRUNTIME_DEP_URL)
    set(DEP_URL_${ONNXRUNTIME_DEP_NAME} ${ONNXRUNTIME_DEP_URL})
    # The third column is SHA1 hash value
    set(DEP_SHA1_${ONNXRUNTIME_DEP_NAME} ${ONNXRUNTIME_DEP})
  endif()
endforeach()

include(FetchContent)
FetchContent_Declare(
  GSL
  URL ${DEP_URL_microsoft_gsl}
  URL_HASH SHA1=${DEP_SHA1_microsoft_gsl}
  FIND_PACKAGE_ARGS 4.0 NAMES Microsoft.GSL
)

set(ONNXRUNTIME_VER "1.16.3")
set(ONNXRUNTIME_BINARY_PLATFORM "x64")
if(WIN32)
  set(ONNXRUNTIME_URL
      "v${ONNXRUNTIME_VER}/onnxruntime-win-${ONNXRUNTIME_BINARY_PLATFORM}-${ONNXRUNTIME_VER}.zip"
  )
else()
  set(ONNXRUNTIME_URL
      "v${ONNXRUNTIME_VER}/onnxruntime-linux-${ONNXRUNTIME_BINARY_PLATFORM}-${ONNXRUNTIME_VER}.tgz"
  )
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ONNXRUNTIME_URL
        "v${ONNXRUNTIME_VER}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VER}.tgz"
    )
  endif()
endif()

set(ort_fetch_URL
    "https://github.com/microsoft/onnxruntime/releases/download/${ONNXRUNTIME_URL}"
)
FetchContent_Declare(onnxruntime URL ${ort_fetch_URL})

FetchContent_MakeAvailable(GSL onnxruntime)
set(GSL_TARGET "Microsoft.GSL::GSL")
set(GSL_INCLUDE_DIR
    "$<TARGET_PROPERTY:${GSL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>"
)

set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)
