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

# Parse the VERSION file in the current directory and extract it into variables.
# The VERSION file should be a file containing the semantic version of the
# project on a single line of the form x.y.z[-prerelease]
#
# This function sets the following variables in the parent scope:
#   - version_full: x.y.z-prerelease
#   - version_core: x.y.z
#   - version_major: x
#   - version_minor: y
#   - version_patch: z
#   - version_prerelease: prerelease
# If no prerelease is specified, it's empty.
#
# Example usage: dd_parse_version(${CMAKE_CURRENT_SOURCE_DIR}/VERSION)
function(dd_parse_version version_file)
  file(READ "${version_file}" ver)
  string(REPLACE "\n" "" ver ${ver})
  string(REGEX MATCHALL "([0-9]+)|-(.*)\\+" result ${ver})
  list(GET result 0 ver_major)
  list(GET result 1 ver_minor)
  list(GET result 2 ver_patch)
  list(LENGTH result result_len)
  if(result_len EQUAL "4")
    list(GET result 3 ver_prerelease)
  else()
    set(ver_prerelease "")
  endif()
  message(STATUS "Building version ${ver}")

  set(version_full ${ver} PARENT_SCOPE)
  set(version_core ${ver_major}.${ver_minor}.${ver_patch} PARENT_SCOPE)
  set(version_major ${ver_major} PARENT_SCOPE)
  set(version_minor ${ver_minor} PARENT_SCOPE)
  set(version_patch ${ver_patch} PARENT_SCOPE)
  set(version_prerelease ${ver_prerelease} PARENT_SCOPE)
endfunction()
