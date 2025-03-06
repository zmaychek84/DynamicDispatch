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
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/quant/quant.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename InT = uint16_t, typename OutT = uint16_t>
int test_quant(int M, int N, bool debug = false,
               const std::string &a_dtype = "bfloat16",
               const std::string &c_dtype = "uint16",
               const std::string &model_name = "4x4mzdk5") {

  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> data_shape = {Ms, Ns};
  std::vector<size_t> aie_out_shape = {Ms, Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> data(M * N);
  std::vector<OutT> cpu_data_dq(M * N);
  std::vector<OutT> aie_data_dq(M * N);
  std::vector<int32_t> qdq_params(QDQparam_size);

  RowMajorMatrix<OutT> input_mat(
      M, N, data.data()); // just used for generating golden data
  RowMajorMatrix<OutT> cpu_mat(
      M, N, cpu_data_dq.data()); // just used for generating golden data

#ifdef RANDOM_DATA
  srand(0xABCD);
  lrn_matrix::initialize_random_bfloat16(data, M * N, -1.3, 1.3);

  float scale = 0.00065;
  uint16_t zp = 57424;
  qdq_params[0] = zp;
  qdq_params[1] = float_to_bfloat16(1 / scale);

  quant(input_mat, cpu_mat, scale, zp, "uint16");
#else
  // std::string data_folder = OpInterface::get_dd_base_dir() + "//QUANT_" +
  std::string data_folder = OpInterface::get_dd_base_dir() +
                            "//..//Q_DeQ_shapes//Quant_" + std::to_string(M) +
                            "_" + std::to_string(N) + "//";

  std::string ifm_filename = data_folder + "ifm.bin";
  std::string ofm_filename = data_folder + "ofm.bin";
  std::string qdq_filename = data_folder + "wgt.bin";

  read_bin_file(ifm_filename, (char *)data.data());
  read_bin_file(ofm_filename, (char *)cpu_data_dq.data());
  read_bin_file(qdq_filename, (char *)qdq_params.data());
#endif

  // run aie
  std::map<std::string, std::any> attr;

  if (model_name == "4x4mzdk5") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::quant quant_ =
      ryzenai::quant<InT, OutT>(a_dtype, c_dtype, false, attr);

  quant_.debug(debug);
  quant_.set_params(model_name, data_shape);

  std::vector<Tensor> const_Tensor;

  const_Tensor = {{qdq_params.data(), qdq_params_shape, "int16"}};

  quant_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{data.data(), data_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_data_dq.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", N = " << N);
  PROFILE_THIS(quant_.execute(input_Tensor, output_Tensor));
#else
  quant_.execute(input_Tensor, output_Tensor);
#endif

  // compare results
  int max_error = 0;
  int error_limit = 10;
  float L2_norm = 0;
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      int32_t diff = std::abs(aie_data_dq[r * N + c] - cpu_data_dq[r * N + c]);
      L2_norm += ((float)diff * (float)diff);
      if (diff > error_limit) {
        std::cout << "ERROR: Y[" << r << ", " << c << "]: "
                  << "Expected: " << (int)cpu_data_dq[r * N + c] << ", "
                  << "Received: " << (int)aie_data_dq[r * N + c] << ", "
                  << "Diff: " << diff << "\n";
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

#if 0
TEST(C4mzdk5_QUANT_Testa16, Kernel1) {
  int err_count = test_quant<uint16_t, uint16_t>(64, 5120, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel2) {
  int err_count = test_quant<uint16_t, uint16_t>(64, 64, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel3) {
  int err_count = test_quant<uint16_t, uint16_t>(64, 1280, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel4) {
  int err_count = test_quant<uint16_t, uint16_t>(64, 2560, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel5) {
  int err_count = test_quant<uint16_t, uint16_t>(256, 5120, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel6) {
  int err_count = test_quant<uint16_t, uint16_t>(256, 64, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel7) {
  int err_count = test_quant<uint16_t, uint16_t>(256, 640, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel8) {
  int err_count = test_quant<uint16_t, uint16_t>(256, 1280, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel9) {
  int err_count = test_quant<uint16_t, uint16_t>(256, 1920, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel10) {
  int err_count = test_quant<uint16_t, uint16_t>(256, 2560, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel11) {
  int err_count = test_quant<uint16_t, uint16_t>(1024, 2560, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel12) {
  int err_count = test_quant<uint16_t, uint16_t>(1024, 64, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel13) {
  int err_count = test_quant<uint16_t, uint16_t>(1024, 320, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel14) {
  int err_count = test_quant<uint16_t, uint16_t>(1024, 640, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
#endif

TEST(C4mzdk5_QUANT_Testa16, Kernel15) {
  int err_count = test_quant<uint16_t, uint16_t>(1024, 960, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

#if 0
TEST(C4mzdk5_QUANT_Testa16, Kernel16) {
  int err_count = test_quant<uint16_t, uint16_t>(1024, 1280, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel17) {
  int err_count = test_quant<uint16_t, uint16_t>(1024, 1920, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel18) {
  int err_count = test_quant<uint16_t, uint16_t>(4096, 1280, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel19) {
  int err_count = test_quant<uint16_t, uint16_t>(4096, 64, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel20) {
  int err_count = test_quant<uint16_t, uint16_t>(4096, 320, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_QUANT_Testa16, Kernel21) {
  int err_count = test_quant<uint16_t, uint16_t>(4096, 640, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
#endif

TEST(C4mzdk5_QUANT_Testa16, Kernel22) {
  int err_count = test_quant<uint16_t, uint16_t>(4096, 960, false, "bfloat16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
