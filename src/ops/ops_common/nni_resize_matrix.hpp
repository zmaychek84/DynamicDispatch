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

#include <assert.h>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <vector>

namespace nni_resize_matrix {
template <typename T> struct TensorMatrix {
  int const num_rows;
  int const num_cols;
  int const num_channels;
  T *const data;

  TensorMatrix(int num_rows, int num_cols, int num_channels, void *data)
      : num_rows(num_rows), num_cols(num_cols), num_channels(num_channels),
        data(static_cast<T *>(data)) {}

  T &at(int row, int col, int channel) {
    assert(row < num_rows);
    assert(col < num_cols);
    assert(channel < num_channels);
    int const idx =
        (row * num_cols * num_channels) + col * num_channels + channel;
    return data[idx];
  }

  static int size(int num_rows, int num_cols) {
    return num_rows * num_cols * sizeof(T);
  }
};

template <typename T>
void cpu_nni(TensorMatrix<T> input_tensor, TensorMatrix<T> output_tensor,
             int num_interpolations) {

  for (int h = 0; h < input_tensor.num_rows * num_interpolations; ++h) {
    for (int w = 0; w < input_tensor.num_cols * num_interpolations; ++w) {
      for (int c = 0; c < input_tensor.num_channels; ++c) {
        int i, j, k;
        if (h % num_interpolations != 0 && w % num_interpolations == 0) {
          i = h - 1;
          j = w;
          k = c;
        } else if (h % num_interpolations == 0 && w % num_interpolations != 0) {
          i = h;
          j = w - 1;
          k = c;
        } else if (h % num_interpolations != 0 && w % num_interpolations != 0) {
          i = h - 1;
          j = w - 1;
          k = c;
        } else {
          i = h;
          j = w;
          k = c;
        }
        i = i / num_interpolations;
        j = j / num_interpolations;
        k = k;
        output_tensor.at(h, w, c) = input_tensor.at(i, j, k);
      }
    }
  }
}

template <typename T> int check_result(T cpu_Y, T aie_Y) {
  int fail = 0;
  int err_count = 0;
  int max_diff = 0;
  float L2_norm = 0;
  for (int h = 0; h < aie_Y.num_rows; ++h) {
    for (int w = 0; w < aie_Y.num_cols; ++w) {
      for (int c = 0; c < aie_Y.num_channels; ++c) {
        int32_t diff = std::abs(cpu_Y.at(h, w, c) - aie_Y.at(h, w, c));
        L2_norm += ((float)diff * (float)diff);
        if (diff > max_diff) {
          max_diff = diff;
        }
        if (diff > 1) {
          // std::cout << "ERROR: Y[" << h << ", " << w << "," << c << "]: "
          //           << "Expected: " << int(cpu_Y.at(h, w, c)) << ", "
          //           << "Received: " << int(aie_Y.at(h, w, c)) << ", "
          //           << "Diff: " << int(diff) << "\n";
          fail = 1;
          err_count++;
        }
      }
    }
  }
  L2_norm = std::sqrt(L2_norm);
  std::cout << "max_diff is " << max_diff << std::endl;
  std::cout << "L2_norm is " << L2_norm << std::endl;
  return err_count;
}
} // namespace nni_resize_matrix
