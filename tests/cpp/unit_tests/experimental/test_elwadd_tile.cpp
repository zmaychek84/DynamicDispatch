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

#include "../enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/experimental/elwadd_tile.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;
template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_elwadd_tile(size_t M, size_t K, bool debug = false,
                     const std::string &a_dtype = "int16",
                     const std::string &b_dtype = "int8",
                     const std::string &c_dtype = "int32",
                     const std::string &model_name = "mdsqr") {
  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);

  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ms, Ks};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> ab(2 * (M * K)); // only used with model data
  std::vector<InT> a(M * K);
  std::vector<WgT> b(M * K);
  std::vector<OuT> cpu_out(M * K);
  std::vector<OuT> cpu_q_out(M * K);
  std::vector<OuT> aie_out(M * K);
  std::vector<int32_t> qdq_params(QDQparam_size);
  OutMatrix<OuT, 1, 1> aie_Y(M, K, aie_out.data());
  OutMatrix<OuT, 1, 1> cpu_Y(M, K, cpu_out.data());
  OutMatrix<OuT, 1, 1> cpu_Q_Y(M, K, cpu_q_out.data());

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
#if 0
  std::string fld_name = "//bin_files//m3uec_add0"; // 3136x128
  std::vector<uint32_t> aint(M * K);
  read_data_file<uint32_t>(OpInterface::get_dd_base_dir() + fld_name +
                               "//add_in0.txt",
                           (uint32_t *)aint.data());
  for (int r = 0; r < M * K; r++) {
    a[r] = (InT)(aint[r]);
  }

  std::vector<uint32_t> bint(M * K);
  read_data_file<uint32_t>(OpInterface::get_dd_base_dir() + fld_name +
                               "//add_in1.txt",
                           (uint32_t *)bint.data());
  for (int r = 0; r < M * K; r++) {
    b[r] = (InT)(bint[r]);
  }

  read_data_file<float>(OpInterface::get_dd_base_dir() + fld_name +
                            "//in0_scale.txt",
                        (float *)&matA_scale);

  read_data_file<uint32_t>(OpInterface::get_dd_base_dir() + fld_name +
                               "//in0_zp.txt",
                           (uint32_t *)&matA_zero_point);

  read_data_file<float>(OpInterface::get_dd_base_dir() + fld_name +
                            "//in1_scale.txt",
                        (float *)&matB_scale);

  read_data_file<uint32_t>(OpInterface::get_dd_base_dir() + fld_name +
                               "//in1_zp.txt",
                           (uint32_t *)&matB_zero_point);

  qdq_params[0] = float_to_bfloat16(matA_scale);
  qdq_params[1] = matA_zero_point;
  qdq_params[2] = float_to_bfloat16(matB_scale);
  qdq_params[3] = matB_zero_point;

  std::vector<float> outint(M * K);
  read_data_file<float>(OpInterface::get_dd_base_dir() + fld_name +
                            "//add_out.txt",
                        (float *)outint.data());
  for (int r = 0; r < K * M; r++) {
    cpu_out[r] = float_to_bfloat16(outint[r]);
  }
#else
#if 0
  std::string fld_name = OpInterface::get_dd_base_dir() + "//elwadd_mdsqrv1.1//";
  read_bin_file(fld_name + "ifm1.bin", (char *)a.data());
  read_bin_file(fld_name + "ifm2.bin", (char *)b.data());
  read_bin_file(fld_name + "ofm.bin", (char *)cpu_out.data());
  read_bin_file(fld_name + "wgt2.bin", (char *)qdq_params.data());
#else
  size_t a_size = M * K * sizeof(InT);
  std::string fld_name =
      OpInterface::get_dd_base_dir() + "//elwadd_mdsqrv1.1//";
  read_bin_file(fld_name + "ifm.bin", (char *)ab.data());
  memcpy(a.data(), ab.data(), a_size);
  memcpy(b.data(), (char *)(ab.data() + a_size), a_size);

  read_bin_file(fld_name + "ofm.bin", (char *)cpu_out.data());
  read_bin_file(fld_name + "wgt.bin", (char *)qdq_params.data());

  // kernel always expects zero point followed by scale.
  // but for elwadd and elwmul, software interface is opposite.
  // First scale and then zero point.
  // So, software gives as scale, zero point.
  // Kernel swaps so that data is given to kernel in opposite way.
  // When loading XRT data extra swap is needed in test.
  auto temp = qdq_params[0];
  qdq_params[0] = qdq_params[1];
  qdq_params[1] = temp;
  temp = qdq_params[2];
  qdq_params[2] = qdq_params[3];
  qdq_params[3] = temp;
  temp = qdq_params[4];
  qdq_params[4] = qdq_params[5];
  qdq_params[5] = temp;
#endif
#endif

#endif
  std::map<std::string, std::any> attr;

  if (model_name == "4x4mzdk5") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::elw_add_tile elwadd_ = ryzenai::elw_add_tile<InT, WgT, OuT>(
      a_dtype, b_dtype, c_dtype, false, attr);

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

  elwadd_.debug(debug);
  elwadd_.set_params(model_name, a_shape);
  elwadd_.initialize_const_params(const_Tensor);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(elwadd_.execute(input_Tensor, output_Tensor));
#else
  elwadd_.execute(input_Tensor, output_Tensor);
#endif
  if (c_dtype == "uint16") {
    err_count = check_add_result(cpu_Q_Y, aie_Y, 0.01);
  } else {
    err_count = check_add_result_bfloat16<OuT>(cpu_out, aie_out, a_shape, 0.01);
  }
  return err_count;
}

TEST(mdsqrv1_0_ELWADDTILE_Testa8w8, Kernel1) {
  int err_count = test_elwadd_tile<uint8_t, uint8_t, uint16_t>(
      2048, 768, false, "uint8", "uint8", "bfloat16", "mdsqr");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mdsqrv1_0_ELWADDTILE_Testa8w8, Kernel2) {
  int err_count = test_elwadd_tile<uint8_t, uint8_t, uint16_t>(
      512, 1536, false, "uint8", "uint8", "bfloat16", "mdsqr");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mdsqrv1_0_ELWADDTILE_Testa8w8, Kernel3) {
  int err_count = test_elwadd_tile<uint8_t, uint8_t, uint16_t>(
      1024, 1536, false, "uint8", "uint8", "bfloat16", "mdsqr");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mdsqrv1_0_ELWADDTILE_Testa8w8, Kernel4) {
  int err_count = test_elwadd_tile<uint8_t, uint8_t, uint16_t>(
      1, 786432, false, "uint8", "uint8", "bfloat16", "mdsqr");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mxpzi_ELWADDTILE_Testa16w8, Kernel1) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      1024, 768, false, "uint16", "uint16", "bfloat16", "mxpzi");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mxpzi_ELWADDTILE_Testa16w8, Kernel2) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      512, 1536, false, "uint16", "uint16", "bfloat16", "mxpzi");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mxpzi_ELWADDTILE_Testa16w8, Kernel3) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      1024, 1536, false, "uint16", "uint16", "bfloat16", "mxpzi");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mxpzi_ELWADDTILE_Testa16w8, Kernel4) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      1, 786432, false, "uint16", "uint16", "bfloat16", "mxpzi");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_ELWADDTILE_Testa16w8, Kernel1) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      98, 1024, false, "uint16", "uint16", "bfloat16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_ELWADDTILE_Testa16w8, Kernel2) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      49, 3072, false, "uint16", "uint16", "bfloat16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_ELWADDTILE_Testa16w8, Kernel3) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      392, 512, false, "uint16", "uint16", "bfloat16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_ELWADDTILE_Testa16w8, Kernel4) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      392, 1024, false, "uint16", "uint16", "bfloat16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_ELWADDTILE_Testa16w8, Kernel5) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      1568, 256, false, "uint16", "uint16", "bfloat16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_ELWADDTILE_Testa16w8, Kernel6) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      1, 401408, false, "uint16", "uint16", "bfloat16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_ELWADDTILE_Testa16w8, Kernel1) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      128, 1280, false, "bfloat16", "uint16", "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_ELWADDTILE_Testa16w8, Kernel2) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      128, 2560, false, "bfloat16", "uint16", "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_ELWADDTILE_Testa16w8, Kernel3) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      512, 1280, false, "bfloat16", "uint16", "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_ELWADDTILE_Testa16w8, Kernel4) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      2048, 640, false, "bfloat16", "uint16", "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_ELWADDTILE_Testa16w8, Kernel5) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      2048, 1280, false, "bfloat16", "uint16", "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_ELWADDTILE_Testa16w8, Kernel6) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      512, 1280, false, "uint16", "uint16", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_ELWADDTILE_Testa16w8, Kernel7) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      2048, 640, false, "uint16", "uint16", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_ELWADDTILE_Testa16w8, Kernel8) {
  int err_count = test_elwadd_tile<uint16_t, uint16_t, uint16_t>(
      1, 655360, false, "uint16", "uint16", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
