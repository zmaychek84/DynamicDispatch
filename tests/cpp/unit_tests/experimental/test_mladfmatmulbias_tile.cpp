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

#include <iostream>
#include <torch/torch.h>

#include "ops/ops_common/matrix_formatting.h"
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <ops/mladfmatmulbias/mladfmatmulbias.hpp>

#include "../enable_perf.hpp"

#include "test_common.hpp"

template <typename T>
static void initialize_random(std::vector<T> &vec, size_t size, int data_max,
                              std::string dtype = "int8") {
  auto data_min = -(data_max + 1);
  for (int i = 0; i < size; i++) {
    if (dtype == "bfloat16") {
      vec[i] = ryzenai::rand_bfloat16(float(data_max));
    } else if (dtype == "uint4") {
      vec[i] = ryzenai::rand_uint4(data_max);
    } else if (dtype == "int4") {
      vec[i] = ryzenai::rand_int4(data_max);
    } else if (std::is_same<T, float>::value) {
      vec[i] = (2.0 * (rand() / (float)RAND_MAX) - 1.0) * data_max;
    } else {
      vec[i] = (T)(rand() % (data_max - data_min + 1)) + data_min;
    }
  }
}

template <typename InT = uint16_t, typename WgT = uint8_t,
          typename OuT = uint16_t>
int test_matmul_mladf_tile(int M, int K, int N, bool debug = false,
                           const std::string &a_dtype = "bfloat16",
                           const std::string &b_dtype = "int4",
                           const std::string &c_dtype = "bfloat16",
                           int group_size = 128, bool compare_values = true,
                           std::string op_version = "v1") {

  if (b_dtype == "int4" || b_dtype == "uint4") {
    std::tuple<int, int> a_shape = {M, K};
    std::tuple<int, int> b_shape = {K, N};
    std::vector<InT> a(M * K);
    std::vector<float> bias(N);
    std::vector<float> scales(K * N / group_size);
    std::vector<WgT> b(K * N);
    std::vector<WgT> zeros(K * N / group_size);
    int32_t garbage_value = 0xCDCDCDCD;
    std::vector<OuT> c(M * N, garbage_value);
    // std::vector<float> c_golden(M * N, garbage_value);
    srand(42);
    std::map<std::string, std::any> attr;

    attr["op_version"] = op_version;
    attr["group_size"] = group_size;

    // Select the input data range for activations, weights, scales, and bias
    initialize_random<InT>(a, M * K, 100, "bfloat16");
    initialize_random<WgT>(b, K * N, 7, b_dtype);
    initialize_random<WgT>(zeros, K * N / group_size, 7, b_dtype);
    initialize_random<float>(bias, N, 1);
    initialize_random<float>(scales, K * N / group_size, 1);

    ryzenai::mladfmatmulbias mladfmatmulbias_ =
        ryzenai::mladfmatmulbias<InT, WgT, OuT, OuT>(a_dtype, b_dtype, c_dtype,
                                                     true, attr);

    mladfmatmulbias_.debug(debug);

    size_t Ms = static_cast<size_t>(M);
    size_t Ks = static_cast<size_t>(K);
    size_t Ns = static_cast<size_t>(N);

    std::vector<Tensor> const_Tensor;
    std::vector<size_t> v_shape_vec = {Ms, Ks};
    std::vector<size_t> b_shape_vec = {Ks, Ns};
    std::vector<size_t> size_shape = {0, 0}; // useless here
    const_Tensor = {{b.data(), b_shape_vec, b_dtype},
                    {bias.data(), size_shape, a_dtype},
                    {scales.data(), size_shape, a_dtype},
                    {zeros.data(), b_shape_vec, b_dtype}};

    std::vector<Tensor> input_Tensor;
    std::vector<size_t> a_shape_vec = {Ms, Ks};

    input_Tensor = {{a.data(), a_shape_vec, a_dtype}};

    std::vector<Tensor> output_Tensor;
    std::vector<size_t> c_shape_vec = {Ms, Ns};
    output_Tensor = {{c.data(), c_shape_vec, c_dtype}};

    attr["max_m"] = 4096;
    b_shape_vec = {4096, 4096};
    const_Tensor = {{b.data(), b_shape_vec, b_dtype},
                    {bias.data(), size_shape, a_dtype},
                    {scales.data(), size_shape, a_dtype},
                    {zeros.data(), b_shape_vec, b_dtype}};
    mladfmatmulbias_.initialize_const_params(const_Tensor, attr);
    b_shape_vec = {Ks, Ns};
    const_Tensor = {{b.data(), b_shape_vec, b_dtype},
                    {bias.data(), size_shape, a_dtype},
                    {scales.data(), size_shape, a_dtype},
                    {zeros.data(), b_shape_vec, b_dtype}};
    mladfmatmulbias_.initialize_const_params(const_Tensor, attr);
    mladfmatmulbias_.set_shape(a_shape_vec, b_shape_vec, group_size);
#ifdef UNIT_TEST_PERF
    LOG_THIS("M = " << M << ", K = " << K << ", N = " << N
                    << ", Gs = " << group_size);
    PROFILE_THIS(
        mladfmatmulbias_.execute_internal(input_Tensor, output_Tensor, 1));
#else
    mladfmatmulbias_.execute_internal(input_Tensor, output_Tensor, 1);
#endif
    if (!compare_values) {
      return 0;
    }

    // Compute using pytorch
    torch::Tensor a_f = torch::empty({M, K});
    torch::Tensor b_f = torch::empty({K, N});
    torch::Tensor bias_f = torch::empty({1, N});

    // convert bfloat16 activation to float
    std::vector<float> a_f_vec(M * K);
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        a_f_vec[m * K + k] = ryzenai::bfloat16_rnd_even(
            ryzenai::bfloat16_to_float(a[m * K + k]));
      }
    }
    a_f = torch::from_blob(a_f_vec.data(), {M, K}, torch::kFloat32);

    // set bias
    bias_f = torch::from_blob(bias.data(), {1, N}, torch::kFloat32);

    // dequantize weight
    std::vector<float> b_f_vec(K * N);
    for (int n = 0; n < N; ++n) {
      // bias_f[0][n] = bias[n];
      for (int k = 0; k < K; ++k) {
        int g_idx = (k / group_size);
        b_f_vec[k * N + n] =
            (b[k * N + n] - zeros[g_idx * N + n]) * scales[g_idx * N + n];
      }
    }
    b_f = torch::from_blob(b_f_vec.data(), {K, N}, torch::kFloat32);

    // Calculate MatMulBias
    auto ret = torch::matmul(a_f, b_f);
    ret = ret + bias_f;

    float *c_golden = ret.data_ptr<float>();

    float const EPSILON_MAX =
        7.0; // this is the tolerated max error, normalized by sqrt(K)
    float const EPSILON_MEAN =
        0.8; // this is the tolerated mean error, normalized by sqrt(K)
    float const EPSILON_SCALE =
        b_dtype == "int4"
            ? 2.5
            : 1; // for int4 weights, the quantization deviations are higher
    int err_count = 0;
    float err_max = 0;
    float err_min = 0;
    float err_total = 0;
    float err_mean = 0;

    for (int i = 0; i < c.size(); i++) {
      float err = std::abs(ryzenai::bfloat16_rnd_even(c_golden[i]) -
                           ryzenai::bfloat16_to_float(c[i]));
      if (std::abs(err_max) < std::abs(err)) {
        err_max = err;
      }
      if (i == 0) {
        err_min = err;
      } else if (std::abs(err_min) > std::abs(err)) {
        err_min = err;
      }
      err_total += err;
      if (err > EPSILON_MAX * sqrt(K) * EPSILON_SCALE) {
        err_count++;
        if (err_count < 16) {
          std::cout << "First deviating values:"
                    << "\n";
          std::cout << std::dec << "c[" << i << "]: "
                    << "Err: " << err << ", "
                    << "Expected: " << ryzenai::bfloat16_rnd_even(c_golden[i])
                    << ", "
                    << "Received: " << ryzenai::bfloat16_to_float(c[i]) << "\n";
        }
      }
    }

    err_mean = err_total / c.size();
    printf("err_max: %.2f, target: %.2f\n", err_max,
           EPSILON_MAX * sqrt(K) * EPSILON_SCALE);
    printf("err_mean: %.2f, target: %.2f\n", err_mean,
           EPSILON_MEAN * sqrt(K) * EPSILON_SCALE);

    if (err_count > 0) {
      std::cout << std::dec << std::fixed << std::setprecision(3)
                << 100.0 * err_count / c.size()
                << "\% of the values deviate more than allowed." << std::endl;
    }
    bool max_error_violation =
        std::isnan(err_max) || err_max > EPSILON_MAX * sqrt(K) * EPSILON_SCALE;
    bool mean_error_violation =
        std::isnan(err_mean) ||
        err_mean > EPSILON_MEAN * sqrt(K) * EPSILON_SCALE;
    return max_error_violation || mean_error_violation;
  }
}

// Formal test

TEST(QlinearTile_2Testw3a16, Kernel1) {
  int err_count = test_matmul_mladf_tile<int16_t, int8_t, int16_t>(
      1152, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(QlinearTile_2Testw3a16, Kernel1b) {
  int err_count = test_matmul_mladf_tile<int16_t, int8_t, int16_t>(
      1664, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(QlinearTile_2Testw3a16, Kernel2) {
  int err_count = test_matmul_mladf_tile<int16_t, int8_t, int16_t>(
      1536, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(QlinearTile_2Testw3a16, Kernel2b) {
  int err_count = test_matmul_mladf_tile<int16_t, int8_t, int16_t>(
      384, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(QlinearTile_2Testw3a16, Kernel3) {
  int err_count = test_matmul_mladf_tile<int16_t, int8_t, int16_t>(
      768, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(QlinearTile_2Testw3a16, Kernel3b) {
  int err_count = test_matmul_mladf_tile<int16_t, int8_t, int16_t>(
      4096, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(QlinearTile_2Testw3a16, Kernel4) {
  int err_count = test_matmul_mladf_tile<int16_t, int8_t, int16_t>(
      192, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 32, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(QlinearTile_2Testw3a16, Kernel4b) {
  int err_count = test_matmul_mladf_tile<int16_t, int8_t, int16_t>(
      384, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(QlinearTile_2Testw3a16, Kernel5) {
  int err_count = test_matmul_mladf_tile<int16_t, int8_t, int16_t>(
      576, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 32, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(QlinearTile_2Testw3a16, Kernel5b) {
  int err_count = test_matmul_mladf_tile<int16_t, int8_t, int16_t>(
      320, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(QlinearTile_2Testw3a16, Kernel6) {
  int err_count = test_matmul_mladf_tile<int16_t, int8_t, int16_t>(
      420, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 32, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(QlinearTile_2Testw3a16, Kernel6b) {
  int err_count = test_matmul_mladf_tile<int16_t, int8_t, int16_t>(
      12, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 128, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
