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
#include <ops/AttentionMaskPrePro/AttentionMaskPrePro.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_AttentionMaskPrePro(int M, int N, bool debug = false,
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
  std::vector<OutT> cpu_q_out(M * N); // not used during model data
  std::vector<OutT> aie_out(M * N);
  std::vector<int32_t> qdq_params(QDQparam_size);

  RowMajorMatrix<OutT> cpu_Y(M, N, cpu_out.data());
  RowMajorMatrix<OutT> cpu_q_Y(M, N,
                               cpu_q_out.data()); // not used during model data
  RowMajorMatrix<OutT> aie_Y(M, N, aie_out.data());
  RowMajorMatrix<InT> inputMat(M, N, a.data());

#ifdef RANDOM_DATA
  int32_t is_output_uint16 = 0;

  if (c_dtype == "uint16") {
    is_output_uint16 = 1;
  }

  float sc_float = 0.01;
  int16_t sc_out = 1.0 / sc_float; // bfloat16
  OutT zp_out = 129;

  srand(0xABCD);
  initialize_random_bfloat16(a, M * N, -20, 20);

  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float in_gold = bfloat16_to_float(inputMat.at(r, c));
      cpu_Y.at(r, c) = float_to_bfloat16(silu_golden(in_gold));
    }
  }
  // quant_bfloat_to_uint16(cpu_Y, sc_out, zp_out, cpu_q_Y);
  quant_bfloat16_to_int16(cpu_Y, cpu_q_Y, sc_out, zp_out);

  qdq_params[0] = zp_out; // for silu
  qdq_params[1] = float_to_bfloat16(sc_out);
  qdq_params[2] = 1; // out_quant_enable
  qdq_params[3] = 0;
  qdq_params[4] = 0;
  qdq_params[5] = 0; // if 1, enalbe de-quant at input

#else
  std::string fld_name;
  if (N == 512) {
    fld_name = "//bins//bins_attention_mask//tc2_1_512//";
  } else if (N == 256) {
    fld_name = "//bins//bins_attention_mask//tc1_1_256//";
  } else {
    fld_name = "//bins//bins_attention_mask//tc0_1_128//";
  }
  if (N > 77) {
    read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "ifm.bin",
                  reinterpret_cast<char *>(a.data()));
  } else {
    std::vector<InT> abuff(M * 128);
    read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "ifm.bin",
                  reinterpret_cast<char *>(abuff.data()));

    memcpy((void *)a.data(), (void *)abuff.data(), M * 77 * sizeof(InT));
  }

  read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "wgt.bin",
                reinterpret_cast<char *>(qdq_params.data()));

  if (N > 77) {
    read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "ofm.bin",
                  reinterpret_cast<char *>(cpu_out.data()));
  } else {
    std::vector<OutT> cbuff(M * 128);
    read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "ofm.bin",
                  reinterpret_cast<char *>(cbuff.data()));

    memcpy((void *)cpu_out.data(), (void *)cbuff.data(), M * 77 * sizeof(OutT));
  }

#endif

  // run aie
  std::map<std::string, std::any> attr;

  if (model_name == "4x4mzdk5") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::AttentionMaskPrePro AttentionMaskPrePro_ =
      ryzenai::AttentionMaskPrePro<InT, WgT, OutT>(a_dtype, b_dtype, c_dtype,
                                                   false, attr);

  AttentionMaskPrePro_.debug(debug);
  AttentionMaskPrePro_.set_params(model_name, a_shape);

  std::vector<Tensor> const_Tensor;

  const_Tensor = {{qdq_params.data(), qdq_params_shape, "int32"}};

  AttentionMaskPrePro_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", N = " << N);
  PROFILE_THIS(AttentionMaskPrePro_.execute(input_Tensor, output_Tensor));
#else
  AttentionMaskPrePro_.execute(input_Tensor, output_Tensor);
#endif

#ifndef RANDOM_DATA
  // compare results
  // err_count = check_add_result(cpu_Y, aie_Y, 0.1);
  err_count = check_result_bfloat(cpu_Y, aie_Y, 0.01);
  return 0;
  // return err_count;
#else
  return 0;
#endif
}

TEST(START_TAIL_mxgan_AttentionMaskPrePro_Testa16, Kernel1) {
  int err_count = test_AttentionMaskPrePro<uint16_t, uint16_t, uint16_t>(
      1, 512, false, "bfloat16", "bfloat16", "bfloat16", "START_TAIL_PS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(START_TAIL_mxpzi_AttentionMaskPrePro_Testa16, Kernel2) {
  int err_count = test_AttentionMaskPrePro<uint16_t, uint16_t, uint16_t>(
      1, 77, false, "bfloat16", "bfloat16", "bfloat16", "START_TAIL_PS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(START_TAIL_mdsqr_AttentionMaskPrePro_Testa16, Kernel3) {
  int err_count = test_AttentionMaskPrePro<uint16_t, uint16_t, uint16_t>(
      1, 256, false, "bfloat16", "bfloat16", "bfloat16", "START_TAIL_PS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
