// Copyright (c) 2025 Advanced Micro Devices, Inc
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "test_common.hpp"

namespace mladfsoftmax_helpers {

template <typename T>
void read_bin_to_vector(const std::string &file_path, std::vector<T> &vec) {
  std::ifstream ifs(file_path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Failed to open the file.");
  }

  // Get the file size
  ifs.seekg(0, std::ios::end);
  std::streamsize file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  // Ensure the vector has the correct size
  std::streamsize element_num = file_size / sizeof(T);
  if (vec.size() != static_cast<size_t>(element_num)) {
    throw std::runtime_error(
        "The vector size does not match the number of elements in the file.");
  }

  // Read the data into the vector
  if (!ifs.read(reinterpret_cast<char *>(vec.data()), file_size)) {
    throw std::runtime_error("Failed to read the data into the vector.");
  }
}

} // namespace mladfsoftmax_helpers
