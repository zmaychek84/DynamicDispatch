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

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/sigmoid/sigmoid.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_sigmoid(int M, int N, bool debug = false,
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
  std::vector<OutT> cpu_out(M * N);
  std::vector<OutT> aie_out(M * N);
  std::vector<int32_t> qdq_params(QDQparam_size);
  RowMajorMatrix<OutT> cpu_Y(M, N, cpu_out.data());
  RowMajorMatrix<OutT> aie_Y(M, N, aie_out.data());
  RowMajorMatrix<InT> inputMat(M, N, a.data());

#ifdef RANDOM_DATA
  srand(0xABCD);
  initialize_random_bfloat16(a, M * N, -20, 20);

  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float in_gold = bfloat16_to_float(inputMat.at(r, c));
      float sigmoid_out = 1 / (std::exp(-in_gold) + 1);
      cpu_Y.at(r, c) = float_to_bfloat16(sigmoid_out);
    }
  }
#else
  std::string fld_name;
  fld_name = "//bins//sigmoid_input_dq//";

  read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "ifm.bin",
                reinterpret_cast<char *>(a.data()));

  read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "wgt.bin",
                reinterpret_cast<char *>(qdq_params.data()));

  read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "ofm.bin",
                reinterpret_cast<char *>(cpu_out.data()));
#endif

  // run aie
  std::map<std::string, std::any> attr;

  if (model_name == "4x4mzdk5") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::sigmoid sigmoid_ =
      ryzenai::sigmoid<InT, WgT, OutT>(a_dtype, b_dtype, c_dtype, false, attr);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{qdq_params.data(), qdq_params_shape, "int32"}};
  sigmoid_.debug(debug);
  sigmoid_.set_params(model_name, a_shape);
  sigmoid_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", N = " << N);
  PROFILE_THIS(sigmoid_.execute(input_Tensor, output_Tensor));
#else
  sigmoid_.execute(input_Tensor, output_Tensor);
#endif

  // compare results
  err_count = check_add_result_bfloat16<OutT>(cpu_out, aie_out, a_shape, 0.02);

  return err_count;
}

// mxgan
// TEST(START_TAIL_mxgan_SIGMOID_Testa16, Kernel1) {
//  int err_count = test_sigmoid<uint16_t, uint16_t, uint16_t>(
//      1, 512, false, "bfloat16", "uint16", "bfloat16", "START_TAIL_PS");
//  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
//}

TEST(START_TAIL_mxgan_SIGMOID_Testa16, Kernel2) {
  int err_count = test_sigmoid<uint16_t, uint16_t, uint16_t>(
      1, 512, false, "uint16", "uint16", "bfloat16", "START_TAIL_PS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
