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

namespace mladfmatmulbias_helpers {

template <typename InT = int16_t, typename WgT = int8_t, typename OuT = int16_t>
struct MladfMatMulBias {

  size_t M, K, N, G;
  std::vector<InT> a;
  std::vector<WgT> b;
  std::vector<OuT> c;
  std::vector<float> bias;
  std::vector<float> scales;
  std::vector<WgT> zeros;
  std::vector<float> c_golden;

  MladfMatMulBias(size_t M, size_t K, size_t N, size_t G)
      : M(M), K(K), N(N), G(G), a(M * K), b(K * N), c(M * N, 0), bias(N),
        scales(K * N / G), zeros(K * N / G), c_golden(M * N, -1) {}

  void InitializeRandom() {
    initialize_random_mladf<InT>(a, M * K, 1, "bfloat16");
    initialize_random_mladf<WgT>(b, K * N, 7, "uint4");
    initialize_random_mladf<WgT>(zeros, K * N / G, 7, "uint4");
    initialize_random_mladf<float>(bias, N, 1, "float32");
    initialize_random_mladf<float>(scales, K * N / G, 1, "float32");
  }

  void WriteParams(std::string &matmul_dir, std::string &weights_name,
                   std::string &bias_name, std::string &scales_name,
                   std::string &zeros_name) {

    std::ofstream wts_f(matmul_dir + "/" + weights_name,
                        std::ios::out | std::ios::binary);
    std::ofstream bias_f(matmul_dir + "/" + bias_name,
                         std::ios::out | std::ios::binary);
    std::ofstream scales_f(matmul_dir + "/" + scales_name,
                           std::ios::out | std::ios::binary);
    std::ofstream zeros_f(matmul_dir + "/" + zeros_name,
                          std::ios::out | std::ios::binary);

    confirmOpen(wts_f);
    confirmOpen(bias_f);
    confirmOpen(scales_f);
    confirmOpen(zeros_f);

    wts_f.write((char *)b.data(), b.size() * sizeof(WgT));
    bias_f.write((char *)bias.data(), bias.size() * sizeof(float));
    scales_f.write((char *)scales.data(), scales.size() * sizeof(float));
    zeros_f.write((char *)zeros.data(), zeros.size() * sizeof(WgT));
  }

  void ForwardCPU() {

    // compute golden (slow computation, therefore in CI only for small shapes)
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        c_golden[m * N + n] = bias[n];
        for (int k = 0; k < K; ++k) {
          float x = ryzenai::bfloat16_to_float(a[m * K + k]);
          int g_idx = (k / G);
          float y = (b[k * N + n] - zeros[g_idx * N + n]) *
                    ryzenai::bfloat16_rnd_even(scales[g_idx * N + n]);

          c_golden[m * N + n] +=
              ryzenai::bfloat16_rnd_even(x) * ryzenai::bfloat16_rnd_even(y);
        }
      }
    }
  }
};

} // namespace mladfmatmulbias_helpers
