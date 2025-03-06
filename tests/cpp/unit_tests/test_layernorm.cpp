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
#include "ops/ops_common/lrn_matrix.hpp"
#include <ops/layernorm/layernorm.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
#ifndef RANDOM_DATA
#define BF16_WEIGHT_BIAS
#endif
using namespace lrn_matrix;
using namespace std;

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_lrn(int M, int N, bool debug = false,
             const std::string &a_dtype = "int16",
             const std::string &b_dtype = "int16",
             const std::string &c_dtype = "int16",
             const std::string &model_name = "mdsqr") {

  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ns};
  std::vector<size_t> gamma_shape = {Ns};
  std::vector<size_t> beta_shape = {Ns};
  std::vector<size_t> aie_out_shape = {Ms, Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> a(M * N);
  std::vector<float> gamma(N); // for CPU calculation
  std::vector<float> beta(N);  // for CPU calculation
  std::vector<WgT> aie_gamma(N);
  std::vector<WgT> aie_beta(N);
  std::vector<OutT> cpu_out(M * N);
  std::vector<OutT> aie_out(M * N);
  std::vector<int32_t> qdq_params(QDQparam_size);

  std::vector<WgT> b(2 * N);
  BiasVector<WgT, 1> bias(N, b.data());
  ActMatrix<OutT, 1, 1> cpu_Y(M, N, cpu_out.data());
#ifdef RANDOM_DATA
  int32_t is_input_uint16 = 0;

  if (a_dtype == "uint16") {
    is_input_uint16 = 1;
  }

  srand(0xABCD);
  if (is_input_uint16 == 1) {
    initialize_random<InT>(a, M * N, 40000, 100);
  } else {
    initialize_random_bfloat16(a, M * N, -20, 20);
  }
  initialize_random_bfloat16(b, 2 * N, -1, 1);
  // init_random_bias(bias, -2, 2); // float to bfloat16

  for (int c = 0; c < N; c++) {
    gamma[c] = (bfloat2float(bias.gamma(c)));
    beta[c] = (bfloat2float(bias.beta(c)));
    aie_gamma[c] = bias.gamma(c);
    aie_beta[c] = bias.beta(c);
  }

  // quantize output
  float sc_float = 0.1;
  int16_t sc_out = float2bfloat(1.0 / sc_float); // bfloat16
  OutT zp_out = 129;
  float sc_in = 0.03;
  InT zp_in = 128;

  qdq_params[0] = (int32_t)sc_out;
  qdq_params[1] = (int32_t)zp_out;
  // for mxgan, user needs to set it based on Q datatype
  qdq_params[lrn_isint16_idx] = 1; // for mzdk5, this is enable quant at output
  qdq_params[3] = float2bfloat(sc_in);
  qdq_params[4] = (is_input_uint16 == 0) ? 0 : zp_in;
  qdq_params[5] = is_input_uint16; // if 1, enalbe de-quant at input

  std::vector<std::vector<InT>> In(M);
  std::vector<std::vector<float>> Out(M);

  if (is_input_uint16 == 1) {
    std::vector<uint16_t> a_dq(M * N);
    dequant_to_bfloat(a, a_dq, zp_in, sc_in);
    for (int r = 0; r < M; r++) {
      for (int c = 0; c < N; c++) {
        In[r].push_back(a_dq[r * N + c]);
      }
    }
  } else {
    // initialize golden inputs
    for (int r = 0; r < M; r++) {
      for (int c = 0; c < N; c++) {
        In[r].push_back(a[r * N + c]);
      }
    }
  }

  // compute golden
  compute_lrn_bfloat16(In, gamma, beta, Out);

  if (c_dtype == "uint16") {
    q_bfloat2uint16(Out, float2bfloat(sc_float), zp_out, cpu_Y);
  } else {
    q_bfloat2uint8(Out, float2bfloat(sc_float), zp_out, cpu_Y);
  }
#else
  int32_t is_input_uint16 = 0;

  if (a_dtype == "uint16") {
    is_input_uint16 = 1;
  }

  std::string data_folder = OpInterface::get_dd_base_dir() + "//.." +
                            "//bins//PSW//lrn//layer2_lrn2_dd//";

#ifdef BF16_WEIGHT_BIAS
  std::string wgt_filename = data_folder + "weight_bf16.txt";
  std::string bias_filename = data_folder + "bias_bf16.txt";
#else
  std::string wgt_filename = data_folder + "weight_uint8.txt";
  std::string bias_filename = data_folder + "bias_int32.txt";
  std::string wgt_scale_filename = data_folder + "weight_scale_float32.txt";
  std::string wgt_zp_filename = data_folder + "weight_zp_uint8.txt";
  std::string bias_scale_filename = data_folder + "bias_scale_float32.txt";
  std::string bias_zp_filename = data_folder + "bias_zp_int32.txt";
#endif

  std::string ifm_filename = data_folder + "ifm_uint16.txt";
  std::string ofm_filename = data_folder + "ofm_uint16.txt";
  std::string sc_in_filename = data_folder + "sc_in_float32.txt";
  std::string zp_in_filename = data_folder + "zp_in_uint16.txt";
  std::string sc_out_filename = data_folder + "sc_out_float32.txt";
  std::string zp_out_filename = data_folder + "zp_out_uint16.txt";

  std::vector<uint32_t> aint(M * N);
  read_data_file<uint32_t>(ifm_filename, (uint32_t *)aint.data());

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a[i * N + j] = (uint16_t)aint[i * N + j];
    }
  }

#ifdef BF16_WEIGHT_BIAS
  read_data_file<InT>(wgt_filename, (InT *)aie_gamma.data());
  read_data_file<InT>(bias_filename, (InT *)aie_beta.data());
#else
  std::vector<uint32_t> weight(N);
  std::vector<int32_t> bias_in(N);
  uint16_t wgt_zp;
  int32_t bias_zp;
  float wgt_scale, bias_scale;

  read_data_file<float>(wgt_scale_filename, (float *)&wgt_scale);
  read_data_file<uint16_t>(wgt_zp_filename, (uint16_t *)&wgt_zp);
  read_data_file<uint32_t>(wgt_filename, (uint32_t *)weight.data());
  read_data_file<float>(bias_scale_filename, (float *)&bias_scale);
  read_data_file<int32_t>(bias_zp_filename, (int32_t *)&bias_zp);
  read_data_file<int32_t>(bias_filename, (int32_t *)bias_in.data());

  for (int i = 0; i < N; i++) {
    float out_dq = (weight[i] - wgt_zp) * wgt_scale;
    gamma[i] = out_dq;
    aie_gamma[i] = float2bfloat(out_dq);
    out_dq = ((int32_t)bias_in[i] - bias_zp) * bias_scale;
    beta[i] = out_dq;
    aie_beta[i] = float2bfloat(out_dq);
  }
#endif

  float sc_in, sc_float = 0;
  uint16_t zp_in, zp_out = 0;
  read_data_file<float>(sc_in_filename, (float *)&sc_in);
  read_data_file<uint16_t>(zp_in_filename, (uint16_t *)&zp_in);
  read_data_file<float>(sc_out_filename, (float *)&sc_float);
  read_data_file<uint16_t>(zp_out_filename, (uint16_t *)&zp_out);

  int16_t sc_out = float2bfloat(1.0 / sc_float); // bfloat16
  qdq_params[0] = sc_out;
  qdq_params[1] = zp_out;
  qdq_params[2] = 1; // for mzdk5, this is enable quant at output
  qdq_params[3] = float2bfloat(sc_in);
  qdq_params[4] = (is_input_uint16 == 0) ? 0 : zp_in;
  qdq_params[5] = is_input_uint16; // if 1, enalbe de-quant at input

#if 0 // No need for model data. All this is related to CPU calculation
  if (is_input_uint16 == 0) { // a is bfloat
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float out_dq = (float)((uint16_t)aint[i * N + j] - zp_in) * sc_in;
        a[i * N + j] = float2bfloat(out_dq);
      }
    }
  }

  std::vector<std::vector<InT>> In(M);
  std::vector<std::vector<float>> Out(M);

  if (is_input_uint16 == 1) {
    std::vector<uint16_t> a_dq(M * N);
    dequant_to_bfloat(a, a_dq, zp_in, sc_in);
    for (int r = 0; r < M; r++) {
      for (int c = 0; c < N; c++) {
        In[r].push_back(a_dq[r * N + c]);
      }
    }
  } else {
    // initialize golden inputs
    for (int r = 0; r < M; r++) {
      for (int c = 0; c < N; c++) {
        In[r].push_back(a[r * N + c]);
      }
    }
  }

  // compute golden
  compute_lrn_bfloat16(In, gamma, beta, Out);

  if (c_dtype == "uint16") {
    q_bfloat2uint16(Out, float2bfloat(sc_float), zp_out, cpu_Y);
  } else {
    q_bfloat2uint8(Out, float2bfloat(sc_float), zp_out, cpu_Y);
  }
#endif

  read_data_file<uint16_t>(ofm_filename, (uint16_t *)cpu_out.data());
#endif
  // run aie
  std::map<std::string, std::any> attr;

  if (model_name == "4x4PSW1.0") {
    attr["input_shape"] = std::vector<int>{1, M, N};
  }

  if (model_name.find("4x4") != std::string::npos) {
    attr["design_param"] = std::vector<string>{"4x4"};
  }

  ryzenai::layernorm layernorm_ = ryzenai::layernorm<InT, WgT, OutT>(
      a_dtype, b_dtype, c_dtype, false, attr);

  layernorm_.debug(debug);
  layernorm_.set_params(model_name, a_shape);

  std::vector<Tensor> const_Tensor;

  const_Tensor = {{aie_gamma.data(), gamma_shape, b_dtype},
                  {aie_beta.data(), beta_shape, b_dtype},
                  {qdq_params.data(), qdq_params_shape, "int32"}};

  layernorm_.initialize_const_params(const_Tensor, attr);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", N = " << N);
  PROFILE_THIS(layernorm_.execute(input_Tensor, output_Tensor));
#else
  layernorm_.execute(input_Tensor, output_Tensor);
#endif

  // compare results
  int max_error = 0;
  int error_limit = 183;
  float L2_norm = 0;
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      int32_t diff = std::abs(aie_out[r * N + c] - cpu_Y.at(r, c));
      L2_norm += ((float)diff * (float)diff);
      if (diff > error_limit) {
        // std::cout << "ERROR: Y[" << r << ", " << c << "]: "
        //           << "Expected: " << (int)cpu_Y.at(r, c) << ", "
        //           << "Received: " << (int)aie_out[r * N + c] << ", "
        //           << "Diff: " << diff << "\n";
        err_count++;
      }
      max_error = (diff > max_error) ? diff : max_error;
    }
  }
  L2_norm = sqrt(L2_norm);
  std::cout << "L2_norm : " << L2_norm << std::endl;
  std::cout << "Maximum Difference : " << max_error << std::endl;
  // LOG_THIS("Maximum Difference : " << max_error);

  if (max_error <= error_limit) {
    err_count = 0;
  }

  return err_count;
}

TEST(mdsqrv1_0_LRN_Testa8w8, Kernel1) {
  int err_count = test_lrn<int16_t, int16_t, uint8_t>(
      512, 768, false, "bfloat16", "int16", "uint8", "mdsqr");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mxpzi_LRN_Testa16w8, Kernel1) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      128, 768, false, "bfloat16", "int16", "uint16", "mxpzi");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mxganv1_1_LRN_Testa16w8, Kernel1) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      512, 768, false, "bfloat16", "int16", "uint16", "mxgan");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mxganv1_2_LRN_Testa16w8, Kernel1) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      512, 768, false, "bfloat16", "int16", "uint16", "mxganv1.2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// m3uec : use the actual shape from ONNX
TEST(m3uec_LRN_Testa16w8, Kernel1) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      3136, 128, false, "bfloat16", "int16", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_LRN_Testa16w8, Kernel2) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      784, 256, false, "bfloat16", "int16", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_LRN_Testa16w8, Kernel3) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      196, 512, false, "bfloat16", "int16", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_LRN_Testa16w8, Kernel4) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      49, 1024, false, "bfloat16", "int16", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_LRN_Testa16w8, Kernel5) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      1, 1024, false, "bfloat16", "int16", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_LRN_Testa16w8, Kernel6) {
  int err_count = test_lrn<uint16_t, int16_t, uint16_t>(
      3136, 128, false, "uint16", "int16", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_LRN_Testa16w8, Kernel7) {
  int err_count = test_lrn<uint16_t, int16_t, uint16_t>(
      784, 256, false, "uint16", "int16", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_LRN_Testa16w8, Kernel8) {
  int err_count = test_lrn<uint16_t, int16_t, uint16_t>(
      196, 512, false, "uint16", "int16", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_LRN_Testa16w8, Kernel9) {
  int err_count = test_lrn<uint16_t, int16_t, uint16_t>(
      49, 1024, false, "uint16", "int16", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_LRN_Testa16w8, Kernel10) {
  int err_count = test_lrn<uint16_t, int16_t, uint16_t>(
      1, 1024, false, "uint16", "int16", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m7h4xjg_LRN_Testa16w8, Kernel1) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      77, 1024, false, "bfloat16", "int16", "uint16", "m7h4xjg");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_LRN_Testa16w8, Kernel_bf16_uint16_64_1280) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      64, 1280, false, "bfloat16", "int16", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_LRN_Testa16w8, Kernel_bf16_uint16_256_1280) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      256, 1280, false, "bfloat16", "int16", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_LRN_Testa16w8, Kernel_bf16_uint16_1024_640) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      1024, 640, false, "bfloat16", "int16", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_LRN_Testa16w8, Kernel_bf16_uint16_4096_320) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      4096, 320, false, "bfloat16", "int16", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_LRN_Testa16w8, Kernel_uint16_uint16_64_1280) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      64, 1280, false, "uint16", "int16", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_LRN_Testa16w8, Kernel_uint16_uint16_256_1280) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      256, 1280, false, "uint16", "int16", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_LRN_Testa16w8, Kernel_uint16_uint16_1024_640) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      1024, 640, false, "uint16", "int16", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_LRN_Testa16w8, Kernel_uint16_uint16_4096_320) {
  int err_count = test_lrn<int16_t, int16_t, uint16_t>(
      4096, 320, false, "uint16", "int16", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mdsqrv1_1_LRN_Testabf16, Kernel1) {
  int err_count = test_lrn<int16_t, int16_t, uint8_t>(
      256, 768, false, "bfloat16", "int16", "uint8", "mdsqrv1.1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSW_LRN_Testa16w8,
     Kernel_act_uint16_wgt_bf16_bias_bf16_out_uint16_64_768) {
  int err_count = test_lrn<uint16_t, int16_t, uint16_t>(
      64, 768, false, "uint16", "uint16", "uint16", "4x4PSW1.0");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
