# Copyright (c) 2025 Advanced Micro Devices, Inc
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

set(LIBRARY_NAME transaction)

if(DISABLE_LARGE_TXN_OPS)
  set(LARGE_TXN_OPT "--disable-large-txn-ops")
endif(DISABLE_LARGE_TXN_OPS)

set(generate_cmd
    ${Python3_EXECUTABLE}
    "${PROJECT_SOURCE_DIR}/tools/parse_transaction_binary.py"
    "--root"
    "${PROJECT_SOURCE_DIR}"
    "--out-dir"
    "${CMAKE_CURRENT_BINARY_DIR}"
    "--quiet"
    "${LARGE_TXN_OPT}"
)

execute_process(
  COMMAND ${generate_cmd} "--list" OUTPUT_VARIABLE transaction_package_files
)
string(REPLACE "\n" ";" transaction_package_files_list
               ${transaction_package_files}
)

set(LIB_SRC src/txn_container.cpp)

add_custom_command(
  OUTPUT ${transaction_package_files_list} COMMAND ${generate_cmd}
  DEPENDS ${PROJECT_SOURCE_DIR}/tools/parse_transaction_binary.py
)

add_library(${LIBRARY_NAME} ${transaction_package_files_list} ${LIB_SRC})
if(MSVC)
  target_compile_options(${LIBRARY_NAME} PRIVATE /bigobj "/guard:cf")
endif()
target_link_libraries(${LIBRARY_NAME} PRIVATE ZLIB::ZLIB)
target_include_directories(
  ${LIBRARY_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/transaction/include
)
target_include_directories(
  ${LIBRARY_NAME} PRIVATE ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR}
)

include(GNUInstallDirs)

install(
  TARGETS ${LIBRARY_NAME}
  EXPORT ${PROJECT_NAME}-targets
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ryzenai/dynamic_dispatch
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
