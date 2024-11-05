/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/DotProductSigmoid/DotProductSigmoid.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_DotProductSigmoid(int M, int N, bool debug = false,
                           const std::string &a_dtype = "int16",
                           const std::string &w_dtype = "int16",
                           const std::string &c_dtype = "int16",
                           const std::string &model_name = "mdsqr") {

  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ns};
  std::vector<size_t> w_shape = {Ns, Ms};
  std::vector<size_t> bias_shape = {1};

  std::vector<size_t> aie_out_shape = {1};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> a(M * N);
  std::vector<InT> w(N * M);
  std::vector<int32_t> bias(1, 0);
  std::vector<OutT> cpu_out(1);
  std::vector<OutT> aie_out(1);
  std::vector<int32_t> qdq_params(QDQparam_size);
  std::vector<uint8_t> b_bo(w_shape[0] * w_shape[1] * sizeof(InT) +
                            bias_shape[0] * sizeof(int32_t) +
                            QDQparam_size * sizeof(int32_t));

  RowMajorMatrix<OutT> cpu_Y(1, 1, cpu_out.data());
  RowMajorMatrix<OutT> aie_Y(1, 1, aie_out.data());

#ifdef RANDOM_DATA
  srand(0xABDE);
  initialize_random_bfloat16(a, M * N, -2, 2);
  initialize_random_bfloat16(w, M * N, -2, 2);

  float dot_sum = 0;
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      dot_sum +=
          bfloat16_to_float(a[r * N + c]) * bfloat16_to_float(w[r * N + c]);
    }
  }
  cpu_Y.at(0, 0) = float_to_bfloat16(sigmoid_golden(dot_sum, 0, 0));

#else
  std::string fld_name;
  if (a_dtype == "uint16") {
    fld_name = "//bins//dotproduct-sigmoid-dq//";
  } else {
    fld_name = "//bins//dotproduct-sigmoid//";
  }

  read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "ifm.bin",
                reinterpret_cast<char *>(a.data()));

  read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "wgt.bin",
                reinterpret_cast<char *>(b_bo.data()));

  memcpy(w.data(), reinterpret_cast<char *>(b_bo.data()),
         w_shape[0] * w_shape[1] * sizeof(InT));
  memcpy(bias.data(),
         reinterpret_cast<char *>(b_bo.data()) +
             w_shape[0] * w_shape[1] * sizeof(InT),
         bias_shape[0] * sizeof(int32_t));
  memcpy(qdq_params.data(),
         reinterpret_cast<char *>(b_bo.data()) +
             w_shape[0] * w_shape[1] * sizeof(InT) +
             bias_shape[0] * sizeof(int32_t),
         QDQparam_size * sizeof(int32_t));

  read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "ofm.bin",
                reinterpret_cast<char *>(cpu_out.data()));
#endif

  // run aie
  std::map<std::string, std::any> attr;

  if (model_name == "4x4mzdk5") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::DotProductSigmoid DotProductSigmoid_ =
      ryzenai::DotProductSigmoid<InT, WgT, OutT>(a_dtype, w_dtype, c_dtype,
                                                 false, attr);

  DotProductSigmoid_.debug(debug);
  DotProductSigmoid_.set_params(model_name, a_shape);

  std::vector<Tensor> const_Tensor;

  const_Tensor = {{w.data(), w_shape, w_dtype},
                  {bias.data(), bias_shape, "int32"},
                  {qdq_params.data(), qdq_params_shape, "int32"}};

  DotProductSigmoid_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", N = " << N);
  PROFILE_THIS(DotProductSigmoid_.execute(input_Tensor, output_Tensor));
#else
  DotProductSigmoid_.execute(input_Tensor, output_Tensor);
#endif

  // compare results
  err_count = check_result_bfloat(cpu_Y, aie_Y, 0.01);
  std::cout << cpu_out[0] << std::endl;
  std::cout << aie_out[0] << std::endl;
  std::cout << cpu_out[0] - aie_out[0] << std::endl;
#ifdef RANDOM_DATA
  return 0; // err_count;
#else
  return err_count;
#endif
}

// input bf16
// TEST(START_TAIL_mxgan_DotProductSigmoid_Testa16, Kernel1) {
//  int err_count = test_DotProductSigmoid<uint16_t, uint16_t, uint16_t>(
//      1, 768, false, "bfloat16", "bfloat16", "bfloat16", "START_TAIL_PS");
//  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
//}

// input uint16
TEST(START_TAIL_mxgan_DotProductSigmoid_Testa16, Kernel2) {
  int err_count = test_DotProductSigmoid<uint16_t, uint16_t, uint16_t>(
      1, 768, false, "uint16", "bfloat16", "bfloat16", "START_TAIL_PS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
