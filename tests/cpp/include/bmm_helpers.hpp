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

namespace bmm_helpers {
template <typename Tx, typename Tw, typename Ty>
void cpu_bmm(Tx X, Tw W, Ty Y, bool trans) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      float acc = 0.0;
      for (int k = 0; k < X.num_cols; ++k) {
        float fx = ryzenai::bfloat16_to_float(X.at(r, k));
        float fw = 0.0;
        if (trans) {
          fw = ryzenai::bfloat16_to_float(W.at(c, k));
        } else {
          fw = ryzenai::bfloat16_to_float(W.at(k, c));
        }
        acc += fx * fw;
      }
      Y.at(r, c) = ryzenai::float_to_bfloat16(acc);
    }
  }
}

template <typename Tx, typename Tw, typename Ty>
void cpu_bmm2(Tx X, Tw W, Ty Y, int B) {
  // golden computation including transpose of heads in ofm
  int M = Y.num_rows / B;
  int K = X.num_cols;
  int N = Y.num_cols;
  for (int m = 0; m < M; ++m) {
    for (int b = 0; b < B; ++b) {
      for (int n = 0; n < N; ++n) {
        float acc = 0.0;
        for (int k = 0; k < K; ++k) {
          float fx = ryzenai::bfloat16_to_float(X.at(b * M + m, k));
          float fw = ryzenai::bfloat16_to_float(W.at(b * K + k, n));
          acc += fx * fw;
        }
        Y.at(m * B + b, n) = ryzenai::float_to_bfloat16(acc);
      }
    }
  }
}

template <typename Tx, typename Tw, typename Ty>
void cpu_bmmb1(Tx X, Tw W, Ty Y, bool trans) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      float acc = 0.0;
      for (int k = 0; k < X.num_cols; ++k) {
        float fx = ryzenai::bfloat16_to_float(X.at(r, k));
        float fw = 0.0;
        if (trans) {
          fw = ryzenai::bfloat16_to_float(W.at(c, k));
        } else {
          fw = ryzenai::bfloat16_to_float(W.at(k, c));
        }
        acc += fx * fw;
      }
      Y.at(r, c) = ryzenai::float_to_bfloat16(acc);
    }
  }
}

template <typename Tx, typename Tw, typename Ty>
void cpu_bmmb2(Tx X, Tw W, Ty Y, int B0, int B1) {
  // golden computation including transpose of heads in ofm
  int M = Y.num_rows / B0;
  int K = X.num_cols;
  int N = Y.num_cols;
  int dv = B0 / B1;
  for (int b = 0; b < B0; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float acc = 0.0;
        for (int k = 0; k < K; ++k) {
          float fx = ryzenai::bfloat16_to_float(X.at(b * M + m, k));
          float fw = ryzenai::bfloat16_to_float(W.at((b / dv) * K + k, n));
          acc += fx * fw;
        }
        Y.at(m * B0 + b, n) = ryzenai::float_to_bfloat16(acc);
      }
    }
  }
}

} // namespace bmm_helpers
