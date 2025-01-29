/*
 Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 Licensed under the MIT License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/gelu/gelu.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_gelu(int M, int N, bool debug = false,
              const std::string &a_dtype = "int16",
              const std::string &b_dtype = "int16",
              const std::string &c_dtype = "int16",
              const std::string &model_name = "mdsqr") {

  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ns};
  std::vector<size_t> aie_out_shape = {Ms, Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> a(M * N);
  std::vector<float> a_dq(M * N);
  std::vector<OutT> cpu_out(M * N);
  std::vector<OutT> aie_out(M * N);
  std::vector<int16_t> qdq_params(QDQparam_size);

  RowMajorMatrix<OutT> cpu_Y(M, N, cpu_out.data());
  RowMajorMatrix<OutT> aie_Y(M, N, aie_out.data());
  RowMajorMatrix<InT> inputMat(M, N, a.data());

#ifdef RANDOM_DATA
  int32_t is_input_uint16 = 0;

  if (a_dtype == "uint16") {
    is_input_uint16 = 1;
  }

  uint16_t in_dq_zero_point = 4250;
  float in_dq_scale = 0.001;

  srand(0xABCD);
  if (is_input_uint16 == 1) {
    initialize_random<InT>(a, M * N, 4600, 4000);
  } else {
    throw std::runtime_error("Gelu not supported datatype.");
  }

  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float in_gold = (inputMat.at(r, c) - in_dq_zero_point) * in_dq_scale;
      cpu_Y.at(r, c) = float_to_bfloat16(gelu_golden(in_gold));
    }
  }

  qdq_params[0] = 0; // for silu
  qdq_params[1] = 0;
  qdq_params[2] = 0; // out_quant_enable
  qdq_params[3] = in_dq_zero_point;
  qdq_params[4] = float_to_bfloat16(in_dq_scale);
  qdq_params[5] = 1; // if 1, enalbe de-quant at input

#endif

  // run aie
  std::map<std::string, std::any> attr;

  if (model_name == "4x4mzdk5") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::gelu gelu_ =
      ryzenai::gelu<InT, WgT, OutT>(a_dtype, b_dtype, c_dtype, false, attr);

  gelu_.debug(debug);
  gelu_.set_params(model_name, a_shape);

  std::vector<Tensor> const_Tensor;

  const_Tensor = {{qdq_params.data(), qdq_params_shape, "int16"}};

  gelu_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", N = " << N);
  PROFILE_THIS(gelu_.execute(input_Tensor, output_Tensor));
#else
  gelu_.execute(input_Tensor, output_Tensor);
#endif

  // compare results
  err_count = check_result_bfloat(cpu_Y, aie_Y, 0.01);

  return err_count;
}

// mzdk5 4x4
TEST(C4mzdk5_GELU_Testa16, Kernel_64_5120) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      64, 5120, false, "uint16", "uint16", "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_GELU_Testa16, Kernel_256_5120) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      256, 5120, false, "uint16", "uint16", "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_GELU_Testa16, Kernel_1024_2560) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      1024, 2560, false, "uint16", "uint16", "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_GELU_Testa16, Kernel_4096_1280) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      4096, 1280, false, "uint16", "uint16", "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
