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

# This function is used to build all test executables
function(dd_configure_test target use_gtest)
  set(INCL_DIRS
      ${PROJECT_SOURCE_DIR}/include ${XRT_INCLUDE_DIRS}
      ${PROJECT_SOURCE_DIR}/tests/cpp/include ${DD_SRC_INCLUDE_DIRS}
      ${PROJECT_SOURCE_DIR}/src/passes
  )

  find_package(Torch REQUIRED)
  if(NOT Torch_FOUND)
    message(FATAL_ERROR "Torch package not found. Aborting.")
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

  set(LINK_LIBS dyn_dispatch_core "${TORCH_LIBRARIES}")

  if(ENABLE_SIMNOWLITE_BUILD)
    set(LINK_LIBS ${XRT_HWEMU_LIBRARIES} ${LINK_LIBS})
  endif()

  if(${use_gtest})
    list(APPEND LINK_LIBS GTest::gtest_main)
  endif()

  target_include_directories(${target} PRIVATE ${INCL_DIRS})
  target_link_libraries(${target} PRIVATE ${LINK_LIBS})
  target_compile_options(${target} PRIVATE ${DD_DEFAULT_COMPILE_OPTIONS})

  install(TARGETS ${target} DESTINATION tests)

  # The following is removed to avoid build errors
  # The build error happen because the cpp_tests.exe is called
  # during the build, but it crashes do to incomplete PATH settings
  # The consequence is that ctest will not be able to
  # run gtest testcases
  # In the future we may want ctest integration
  # So leaving this here commented out
  #if(${use_gtest})
  #  gtest_discover_tests(${target} DISCOVERY_TIMEOUT 120)
  #endif()
endfunction()
