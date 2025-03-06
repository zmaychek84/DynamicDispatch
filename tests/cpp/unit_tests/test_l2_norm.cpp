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
#include <math.h>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/l2_norm/l2_norm.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename T> static float lpnorm(std::vector<T> const &v, int P) {
  if (v.empty()) {
    return 0;
  }
  auto const count = static_cast<float>(v.size());
  float sum = 0;
  for (int i = 0; i < count; i++) {
    if (P == 1) {
      sum += v[i];
    } else if (P == 2) {
      sum += v[i] * v[i];
    } else {
      printf("Unsupported P val");
    }
  }

  return sum;
}

template <typename in_el_type = int16_t, typename out_el_type = int8_t>
static void compute_lpnorm_bfloat16(std::vector<std::vector<in_el_type>> &In,
                                    std::vector<std::vector<out_el_type>> &Out,
                                    int P) {
  int num_rows = In.size();
  int num_cols = In[0].size();
  int a_idx;

  assert(num_rows > 0);
  assert(num_cols > 0);
  assert(P > 0 && P < 3);

  for (int r = 0; r < num_rows; r++) {

    std::vector<float> inp_row;
    for (in_el_type i : In[r]) {
      inp_row.push_back(bfloat2float(i));
    }

    // compute lpnorm for each row
    float norm = lpnorm<float>(inp_row, P);
    // printf("CPU Norm %d : %f \n", r, norm);
    //  compute op1 and op2
    for (int c = 0; c < num_cols; c++) {
      float op1;
      if (P == 1) {
        op1 = 1 / norm;
      } else {
        op1 = 1 / sqrt(norm);
      }
      Out[r].push_back(inp_row[c] * op1);
    }
  }
}

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_l2_norm(int M, int N, bool debug = false,
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
  std::vector<InT> a_dq(M * N);
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

  int32_t is_input_uint16 = 0;
  if (a_dtype == "uint16") {
    is_input_uint16 = 1;
  }

  InT zp_in = 13334;
  float sc_in = 0.0015;
  float sc_float = 0.01;
  float sc_out = 1.0 / sc_float; // bfloat16
  OutT zp_out = 32510;

  std::vector<std::vector<InT>> In(M);
  std::vector<std::vector<float>> Out(M);

  srand(0xABCD);
  if (a_dtype == "uint16") {
    initialize_random<InT>(a, a.size(), 26600, 0);
    dequant_to_bfloat(a, a_dq, (int)zp_in,
                      bfloat16_to_float(float_to_bfloat16(sc_in)));
    for (int r = 0; r < M; r++) {
      for (int c = 0; c < N; c++) {
        In[r].push_back(a_dq[r * N + c]);
      }
    }
  } else if (a_dtype == "bfloat16") {
    initialize_random_bfloat16(a, M * N, -20, 20);
    for (int r = 0; r < M; r++) {
      for (int c = 0; c < N; c++) {
        In[r].push_back(a[r * N + c]);
      }
    }
  } else {
    throw std::runtime_error("a_dtype is not valid.");
  }

  compute_lpnorm_bfloat16(In, Out, 2);

  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      cpu_Y.at(r, c) = float_to_bfloat16(Out[r][c]);
    }
  }
  if (is_output_uint16 == 1) {
    quant_bfloat16_to_int16(cpu_Y, cpu_q_Y, sc_out, zp_out);
  }

  qdq_params[0] = float_to_bfloat16(sc_out);
  qdq_params[1] = zp_out;
  qdq_params[2] = is_output_uint16; // out_quant_enable
  qdq_params[3] = float_to_bfloat16(sc_in);
  qdq_params[4] = zp_in;
  qdq_params[5] = is_input_uint16; // if 1, enalbe de-quant at input

#else
  std::string fld_name;
  fld_name = "//bins//l2_norm//";

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
  } else if (model_name == "4x4PSU") {
    attr["design_param"] = std::vector<string>{"4x4PSU"};
  } else if (model_name == "8x4PSU") {
    attr["design_param"] = std::vector<string>{"8x4PSU"};
  } else if (model_name == "8x4HFDS") {
    attr["design_param"] = std::vector<string>{"8x4HFDS"};
  }
  ryzenai::l2_norm l2_norm_ =
      ryzenai::l2_norm<InT, WgT, OutT>(a_dtype, b_dtype, c_dtype, false, attr);

  l2_norm_.debug(debug);
  l2_norm_.set_params(model_name, a_shape);

  std::vector<Tensor> const_Tensor;

  const_Tensor = {{qdq_params.data(), qdq_params_shape, "int32"}};

  l2_norm_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", N = " << N);
  PROFILE_THIS(l2_norm_.execute(input_Tensor, output_Tensor));
#else
  l2_norm_.execute(input_Tensor, output_Tensor);
#endif

  // compare results
  if (is_output_uint16) {
    err_count = check_result_uint16(cpu_q_Y, aie_Y, 2);
  } else {
    err_count = check_result_bfloat(cpu_Y, aie_Y, 0.01);
  }
  return err_count;
}

// mzdk5 4x2
TEST(START_TAIL_m3uec_L2Norm_Testa16, Kernel1) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      1, 768, false, "bfloat16", "bfloat16", "bfloat16", "START_TAIL_PS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(START_TAIL_mxpzi_L2Norm_Testa16, Kernel1) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      1, 768, false, "bfloat16", "bfloat16", "bfloat16", "START_TAIL_PS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
// PSUv1.2 4x4
// PSU1
TEST(PSU4x4_L2Norm_Testa16, Kernel1) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      1, 3072, false, "uint16", "uint16", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(PSU4x4_L2Norm_Testa16, Kernel2) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      1, 3072, false, "bfloat16", "uint16", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
// PSU0
TEST(PSU4x4_L2Norm_Testa16, Kernel3) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      64, 3072, false, "uint16", "uint16", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(PSU4x4_L2Norm_Testa16, Kernel4) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      64, 3072, false, "bfloat16", "uint16", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSUv1.2 8x4
// PSU1
TEST(PSU8x4_L2Norm_Testa16, Kernel1) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      1, 3072, false, "uint16", "uint16", "uint16", "8x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(PSU8x4_L2Norm_Testa16, Kernel2) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      1, 3072, false, "bfloat16", "uint16", "uint16", "8x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
// PSU0
TEST(PSU8x4_L2Norm_Testa16, Kernel3) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      64, 3072, false, "uint16", "uint16", "uint16", "8x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(PSU8x4_L2Norm_Testa16, Kernel4) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      64, 3072, false, "bfloat16", "uint16", "uint16", "8x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// HFDS 8x4
// HFDS0
TEST(HFDS8x4_L2Norm_Testa16, Kernel_64_1536_UINT) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      64, 1536, false, "uint16", "uint16", "uint16", "8x4HFDS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(HFDS8x4_L2Norm_Testa16, Kernel_64_1536_BF16) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      64, 1536, false, "bfloat16", "uint16", "uint16", "8x4HFDS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// HFDS1
TEST(HFDS8x4_L2Norm_Testa16, Kernel_1_1536_UINT) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      1, 1536, false, "uint16", "uint16", "uint16", "8x4HFDS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(HFDS8x4_L2Norm_Testa16, Kernel_1_1536_BF16) {
  int err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
      1, 1536, false, "bfloat16", "uint16", "uint16", "8x4HFDS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
