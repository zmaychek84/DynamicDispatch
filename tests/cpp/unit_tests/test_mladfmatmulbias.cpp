/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <iostream>
#include <torch/torch.h>

#include "ops/ops_common/matrix_formatting.h"
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <ops/mladfmatmulbias/mladfmatmulbias.hpp>

#include "enable_perf.hpp"

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
int test_matmul_mladf(int M, int K, int N, bool debug = false,
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
    mladfmatmulbias_.set_shape(a_shape_vec, b_shape_vec, group_size);
    mladfmatmulbias_.initialize_const_params(const_Tensor, attr);
#ifdef UNIT_TEST_PERF
    LOG_THIS("M = " << M << ", K = " << K << ", N = " << N
                    << ", Gs = " << group_size);
    PROFILE_THIS(mladfmatmulbias_.execute(input_Tensor, output_Tensor));
#else

    mladfmatmulbias_.execute(input_Tensor, output_Tensor);
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

// M = 1, updated overlay

TEST(Qlinear_2Testw3a16, Kernel4mladf1_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf1_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf1_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf1_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf2_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf2_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf2_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf2_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf3_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf3_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf3_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf3_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf4_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf4_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf4_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf4_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf5_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf5_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf5_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 22528, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf5_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 22528, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf6_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf6_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf6_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 32768, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf6_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 32768, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// auto gen tests

// (M, K, N) =  1 4096 1024

TEST(Qlinear_2Testw3a16, Kernel4mladf43_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 1024, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf43_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 1024, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf43_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 1024, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf43_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 1024, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1 4096 4096

TEST(Qlinear_2Testw3a16, Kernel4mladf44_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf44_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf44_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf44_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1 4096 6144

TEST(Qlinear_2Testw3a16, Kernel4mladf45_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 6144, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf45_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 6144, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf45_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 6144, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf45_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 6144, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1 4096 11008

TEST(Qlinear_2Testw3a16, Kernel4mladf46_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf46_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf46_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf46_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1 4096 12288

TEST(Qlinear_2Testw3a16, Kernel4mladf47_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf47_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf47_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf47_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1 4096 14336

TEST(Qlinear_2Testw3a16, Kernel4mladf48_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 14336, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf48_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 14336, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf48_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 14336, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf48_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 14336, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1 4096 22016

TEST(Qlinear_2Testw3a16, Kernel4mladf49_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 22016, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf49_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 22016, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf49_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 22016, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf49_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 22016, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1 4096 28672

TEST(Qlinear_2Testw3a16, Kernel4mladf50_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 28672, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf50_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 28672, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf50_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 28672, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf50_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 28672, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1 4096 128256

TEST(Qlinear_2Testw3a16, Kernel4mladf51_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 128256, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf51_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 128256, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1 4096 151936

TEST(Qlinear_2Testw3a16, Kernel4mladf52_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 151936, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf52_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 4096, 151936, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1 11008 4096

TEST(Qlinear_2Testw3a16, Kernel4mladf53_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf53_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf53_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf53_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1 14336 4096

TEST(Qlinear_2Testw3a16, Kernel4mladf54_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 14336, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf54_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 14336, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf54_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 14336, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf54_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1, 14336, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// M=128

TEST(Qlinear_2Testw3a16, Kernel4mladf7) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf7b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf8) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf8b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf9) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf9b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf10) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf10b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf11) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf11b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf12) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf12b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  128 4096 1024

TEST(Qlinear_2Testw3a16, Kernel4mladf55_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 1024, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf55_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 1024, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf55_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 1024, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf55_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 1024, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  128 4096 4096

TEST(Qlinear_2Testw3a16, Kernel4mladf56_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf56_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf56_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf56_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  128 4096 6144

TEST(Qlinear_2Testw3a16, Kernel4mladf57_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 6144, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf57_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 6144, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf57_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 6144, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf57_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 6144, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  128 4096 11008

TEST(Qlinear_2Testw3a16, Kernel4mladf58_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf58_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf58_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf58_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  128 4096 12288

TEST(Qlinear_2Testw3a16, Kernel4mladf59_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf59_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf59_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf59_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  128 4096 14336

TEST(Qlinear_2Testw3a16, Kernel4mladf60_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 14336, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf60_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 14336, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf60_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 14336, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf60_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 14336, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  128 4096 22016

TEST(Qlinear_2Testw3a16, Kernel4mladf61_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 22016, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf61_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 22016, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf61_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 22016, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf61_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 22016, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  128 4096 28672

TEST(Qlinear_2Testw3a16, Kernel4mladf62_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 28672, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf62_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 28672, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf62_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 28672, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf62_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 28672, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  128 4096 128256

TEST(Qlinear_2Testw3a16, Kernel4mladf63_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 128256, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf63_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 128256, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  128 4096 151936

TEST(Qlinear_2Testw3a16, Kernel4mladf64_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 151936, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf64_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 4096, 151936, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  128 11008 4096

TEST(Qlinear_2Testw3a16, Kernel4mladf65_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf65_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf65_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf65_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  128 14336 4096

TEST(Qlinear_2Testw3a16, Kernel4mladf66_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 14336, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf66_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 14336, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf66_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 14336, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf66_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      128, 14336, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// M=256

TEST(Qlinear_2Testw3a16, Kernel4mladf13) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf13b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf14) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf14b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf15) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf15b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf16) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf16b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf17) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf17b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf18) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf18b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  256 4096 1024

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf67_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 1024, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf67_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 1024, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf67_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 1024, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf67_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 1024, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  256 4096 4096

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf68_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf68_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf68_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf68_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  256 4096 6144

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf69_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 6144, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf69_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 6144, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf69_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 6144, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf69_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 6144, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  256 4096 11008

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf70_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf70_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf70_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf70_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  256 4096 12288

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf71_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf71_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf71_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf71_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  256 4096 14336

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf72_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 14336, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf72_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 14336, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf72_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 14336, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf72_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 14336, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  256 4096 22016

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf73_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 22016, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf73_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 22016, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf73_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 22016, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf73_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 22016, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  256 4096 28672

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf74_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 28672, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf74_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 28672, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf74_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 28672, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf74_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 28672, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  256 4096 128256

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf75_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 128256, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  256 4096 151936

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf76_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 4096, 151936, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  256 11008 4096

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf77_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf77_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf77_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf77_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  256 14336 4096

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf78_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 14336, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf78_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 14336, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf78_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 14336, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf78_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      256, 14336, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// M=512

TEST(Qlinear_2Testw3a16, Kernel4mladf19) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf19b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf20) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf20b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf21) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf21b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf22) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf22b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf23) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf23b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf24) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf24b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  512 4096 1024

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf79_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 1024, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf79_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 1024, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf79_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 1024, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf79_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 1024, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  512 4096 4096

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf80_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf80_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf80_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf80_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  512 4096 6144

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf81_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 6144, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf81_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 6144, false, "bfloat16", "uint4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf81_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 6144, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf81_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 6144, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  512 4096 11008

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf82_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf82_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf82_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf82_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  512 4096 12288

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf83_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf83_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf83_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf83_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  512 4096 14336

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf84_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 14336, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf84_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 14336, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf84_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 14336, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf84_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 14336, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  512 4096 22016

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf85_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 22016, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf85_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 22016, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf85_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 22016, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf85_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 22016, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  512 4096 28672

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf86_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 28672, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf86_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 28672, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf86_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 28672, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf86_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 28672, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  512 4096 151936

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf88_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 151936, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf88_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 4096, 151936, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  512 11008 4096

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf89_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf89_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf89_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf89_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  512 14336 4096

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf90_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 14336, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf90_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 14336, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf90_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 14336, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf90_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      512, 14336, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// M=800

TEST(Qlinear_2Testw3a16, Kernel4mladf25) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf25b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf26) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf26b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf27) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf27b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf28) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf28b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf29) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf29b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf30) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf30b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      800, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// M=1024

TEST(Qlinear_2Testw3a16, Kernel4mladf31) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf31b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf32) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf32b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf33) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf33b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf34) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf34b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf35) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf35b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf36) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf36b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1024 4096 1024

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf91_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 1024, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf91_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 1024, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf91_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 1024, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1024 4096 4096

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf92_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf92_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf92_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf92_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1024 4096 6144

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf93_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 6144, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf93_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 6144, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf93_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 6144, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf93_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 6144, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1024 4096 11008

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf94_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf94_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf94_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf94_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1024 4096 12288

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf95_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf95_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf95_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf95_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1024 4096 14336

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf96_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 14336, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf96_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 14336, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf96_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 14336, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1024 4096 22016

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf97_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 22016, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf97_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 22016, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf97_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 22016, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf97_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 22016, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1024 4096 28672

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf98_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 28672, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf98_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 28672, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf98_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 28672, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf98_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 28672, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1024 4096 128256

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf99_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 128256, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1024 4096 151936

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf100_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 4096, 151936, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1024 11008 4096

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf101_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf101_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf101_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf101_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  1024 14336 4096

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf102_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 14336, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf102_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 14336, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf102_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 14336, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf102_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      1024, 14336, 4096, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// M=2048

TEST(Qlinear_2Testw3a16, Kernel4mladf37) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf37b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 11008, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf38) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf38b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf39) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf39b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf40) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf40b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf41) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf41b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 22528, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf42) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 32, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16, Kernel4mladf42b) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 32768, false, "bfloat16", "uint4", "bfloat16", 128, false);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
// (M, K, N) =  2048 4096 1024

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf103_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 1024, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf103_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 1024, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf103_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 1024, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf103_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 1024, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  2048 4096 4096

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf104_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf104_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 4096, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf104_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf104_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 4096, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  2048 4096 6144

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf105_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 6144, false, "bfloat16", "uint4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf105_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 6144, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf105_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 6144, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf105_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 6144, false, "bfloat16", "int4", "bfloat16", 128, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  2048 4096 11008

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf106_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf106_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 11008, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf106_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf106_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 11008, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  2048 4096 12288

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf107_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 12288, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf107_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf107_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 12288, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  2048 4096 14336

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf108_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 14336, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf108_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 14336, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf108_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 14336, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf108_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 14336, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  2048 4096 22016

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf109_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 22016, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf109_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 22016, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf109_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 22016, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  2048 4096 28672

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf110_uint4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 28672, false, "bfloat16", "uint4", "bfloat16", 32, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf110_uint4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 28672, false, "bfloat16", "uint4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf110_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 28672, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf110_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 4096, 28672, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf113_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf113_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 11008, 4096, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// (M, K, N) =  2048 14336 4096

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf114_int4_grp32_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 14336, 4096, false, "bfloat16", "int4", "bfloat16", 32, true, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Qlinear_2Testw3a16_high_time, Kernel4mladf114_int4_grp128_v1) {
  int err_count = test_matmul_mladf<int16_t, int8_t, int16_t>(
      2048, 14336, 4096, false, "bfloat16", "int4", "bfloat16", 128, true,
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
// ------------- TEST END for mladfmatmulbias -------------
