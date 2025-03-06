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
#include <ops/dmacompiler/qdq_mul/qdq_mul.hpp>

#include "test_common.hpp"
// #define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;
template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_qdq_mul(size_t M1, size_t K1, size_t M2, size_t K2, size_t M3,
                 size_t K3, bool debug = false,
                 const std::string &a_dtype = "int16",
                 const std::string &b_dtype = "int8",
                 const std::string &c_dtype = "int32",
                 const std::string &model_name = "mdsqr") {
  int err_count = 0;
  size_t Ms1 = static_cast<size_t>(M1);
  size_t Ks1 = static_cast<size_t>(K1);
  size_t Ms2 = static_cast<size_t>(M2);
  size_t Ks2 = static_cast<size_t>(K2);
  size_t Ms3 = static_cast<size_t>(M3);
  size_t Ks3 = static_cast<size_t>(K3);

  std::vector<size_t> a_shape = {Ms1, Ks1};
  std::vector<size_t> b_shape = {Ms2, Ks2};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> ab((M1 * K1) + (M2 * K2)); // only used with model data
  std::vector<InT> a(M1 * K1);
  std::vector<WgT> b(M2 * K2);
  std::vector<OuT> cpu_out(M3 * K3);
  std::vector<OuT> cpu_q_out(M3 * K3);
  std::vector<OuT> aie_out(M3 * K3);
  std::vector<int32_t> qdq_params(QDQparam_size);
  OutMatrix<OuT, 1, 1> aie_Y(M3, K3, aie_out.data());
  OutMatrix<OuT, 1, 1> cpu_Y(M3, K3, cpu_out.data());
  OutMatrix<OuT, 1, 1> cpu_Q_Y(M3, K3, cpu_q_out.data());

#if 0
  srand(0xABCD);
  initialize_random<InT>(a, M * K, 16, 0);
  initialize_random<WgT>(b, M * K, 16, 0);

  int32_t matA_zero_point = 2;
  float matA_scale = 2.0;
  int32_t is_matA_uint16 = 1; // is_matA_uint16 = 0, input a is bf16

  int32_t matB_zero_point = 2;
  float matB_scale = 2.0;

  // quantize output for uint16 output
  float matC_scale = 0.1;
  int16_t sc_out = float2bfloat(1.0 / matC_scale); // bfloat16;
  OuT matC_zero_point = 4451;
  int32_t is_matC_uint16 = 0; // is_matC_uint16 = 1, output c is uint16

  if (a_dtype == "uint16" || a_dtype == "bfloat16") {
    initialize_random<InT>(a, M * K, 4600, 4300);
    initialize_random<WgT>(b, M * K, 1200, 800);
    matA_zero_point = 4451;
    matA_scale = 0.001;
    matB_zero_point = 1000;
    matB_scale = 0.0002;
  }

  if (a_dtype == "bfloat16") {
    is_matA_uint16 = 0;
  }

  if (c_dtype == "uint16") {
    is_matC_uint16 = 1;
  }
#endif

#ifdef RANDOM_DATA
  qdq_params[0] = float_to_bfloat16(matA_scale);
  qdq_params[1] = matA_zero_point;
  qdq_params[2] = float_to_bfloat16(matB_scale);
  qdq_params[3] = matB_zero_point;
  qdq_params[4] = (int32_t)sc_out;
  qdq_params[5] = (int32_t)matC_zero_point;
  qdq_params[6] = is_matA_uint16;
  qdq_params[7] = is_matC_uint16;

  std::vector<OuT> a_dq(M * K);
  std::vector<OuT> b_dq(M * K);

  dequant_to_bfloat(a, a_dq, matA_zero_point, matA_scale);
  dequant_to_bfloat(b, b_dq, matB_zero_point, matB_scale);

  if (is_matA_uint16 == 0) { // matA is bf16
    memcpy((void *)a.data(), (void *)a_dq.data(), M * K * sizeof(OuT));
  }

  // compute golden
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < K; c++) {
      cpu_out.at(r * K + c) =
          float_to_bfloat16(bfloat16_to_float(a_dq.at(r * K + c)) +
                            bfloat16_to_float(b_dq.at(r * K + c)));
    }
  }

  if (c_dtype == "uint16") {
    quant_bfloat_to_uint16(cpu_Y, sc_out, matC_zero_point, cpu_Q_Y);
    // q_bfloat2uint16(Out, float2bfloat(matC_scale), zp_out, cpu_Y);
  }

#else

  std::string fld_name =
      OpInterface::get_dd_base_dir() + "//qdq_mul_binfiles//";

  read_bin_file(fld_name + "ifm1.bin", (char *)a.data());
  read_bin_file(fld_name + "ifm2.bin", (char *)b.data());
  read_bin_file(fld_name + "ofm.bin", (char *)cpu_out.data());
  read_bin_file(fld_name + "wgt.bin", (char *)qdq_params.data());
#endif

  std::map<std::string, std::any> attr;

  if (model_name.find("4x4") != std::string::npos) {
    attr["design_param"] = std::vector<string>{"4x4"};
    attr["input_shape_1"] = std::vector<size_t>{1, Ms1, Ks1};
    attr["input_shape_2"] = std::vector<size_t>{1, Ms2, Ks2};
    attr["output_shape"] = std::vector<size_t>{1, Ms3, Ks3};
  }
  ryzenai::qdq_mul qdq_mul_ =
      ryzenai::qdq_mul<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype, false, attr);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{qdq_params.data(), qdq_params_shape, "int32"}};

  std::vector<Tensor> input_Tensor;
  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor b_T = {b.data(), a_shape, b_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(b_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  qdq_mul_.debug(debug);
  qdq_mul_.set_params(model_name, a_shape);
  qdq_mul_.initialize_const_params(const_Tensor);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M3 << ", K = " << K3);
  PROFILE_THIS(qdq_mul_.execute(input_Tensor, output_Tensor));
#else
  qdq_mul_.execute(input_Tensor, output_Tensor);
#endif

  err_count = check_add_result(cpu_Y, aie_Y, 0.01);
  return err_count;
}

// Currently tests runs with model data, yet to do for RANDOM data.
#if 0
TEST(PSW_QDQ_MUL_Testa16a16, Kernel_uint16_uint16_uint16_64_768_64_1_64_768) {
  int err_count = test_qdq_mul<uint16_t, uint16_t, uint16_t>(
      64, 768, 64, 1, 64, 768, false, "uint16", "uint16", "uint16", "4x4PSW1.0");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
#endif
