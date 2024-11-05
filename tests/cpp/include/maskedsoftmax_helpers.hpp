// Copyright (c) 2024 Advanced Micro Devices, Inc
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

namespace maskedsoftmax_helpers {
std::vector<float>
golden_maskedsoftmax(const std::tuple<size_t, size_t, size_t> &shape,
                     const std::vector<uint16_t> &a,
                     const std::vector<uint16_t> &mask,
                     const float pre_mask_scale) {
  auto [B, M, K] = shape;
  std::vector<float> cpu_float(B * M * K);
  for (int batch = 0; batch < B; batch++) {
    for (int m = 0; m < M; m++) {
      const auto partial2dIndex = m * K;
      const auto partial3dIndex = batch * (M * K) + partial2dIndex;
      // compute runningSum to use in softmax dividend
      float runSum = 0;
      // Masking and exponentiating
      for (int k = 0; k < K; k++) {
        cpu_float.at(partial3dIndex + k) =
            std::exp(ryzenai::bfloat16_to_float(a.at(partial3dIndex + k)) *
                         pre_mask_scale +
                     ryzenai::bfloat16_to_float(mask.at(partial2dIndex + k)));
        runSum += cpu_float.at(partial3dIndex + k);
      }
      // Softmaxing
      for (int k = 0; k < K; k++) {
        cpu_float.at(partial3dIndex + k) /= runSum;
      }
    }
  }
  return cpu_float;
}
} // namespace maskedsoftmax_helpers
