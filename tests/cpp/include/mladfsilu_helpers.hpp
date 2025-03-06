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

namespace mladfsilu_helpers {

template <typename InT = int16_t, typename OuT = uint16_t> struct SiLU {

  size_t M, N;
  std::vector<InT> x;
  std::vector<float> y_golden;

  SiLU(size_t M, size_t N) : M(M), N(N) {
    x.resize(M * N);
    y_golden.resize(M * N);
  }

  void InitializeRandom() {
    initialize_random_mladf(x, x.size(), 42, "bfloat16");
  }

  void ForwardCPU() {
    for (int i = 0; i < M * N; ++i) {
      float xf = ryzenai::bfloat16_to_float(x[i]);
      float sigmoid = 1.0f / (1.0f + std::exp(-xf));
      float intermediate = xf * sigmoid;
      y_golden[i] = intermediate;
    }
  }
};

} // namespace mladfsilu_helpers
