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
#include <ops/dequant/dequant.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename InT = uint16_t, typename OutT = uint16_t>
int test_dequant(int M, int N, bool debug = false,
                 const std::string &a_dtype = "uint16",
                 const std::string &c_dtype = "bfloat16",
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

  RowMajorMatrix<OutT> cpu_mat(
      M, N, cpu_data_dq.data()); // just used for results comparison
  RowMajorMatrix<OutT> aie_mat(
      M, N, aie_data_dq.data()); // just used for results comparison

#ifdef RANDOM_DATA
  srand(0xABCD);
  initialize_random<InT>(data, M * N, 65535, 0);

  float scale = 1.0;
  qdq_params[0] = 0;
  qdq_params[1] = float_to_bfloat16(scale);

  dequant_to_bfloat(data, cpu_data_dq, qdq_params[0], scale);
#else
  // std::string data_folder = OpInterface::get_dd_base_dir() + "//DEQUANT_" +
  std::string data_folder = OpInterface::get_dd_base_dir() +
                            "//..//Q_DeQ_shapes//DeQuant_" + std::to_string(M) +
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
  ryzenai::dequant dequant_ =
      ryzenai::dequant<InT, OutT>(a_dtype, c_dtype, false, attr);

  dequant_.debug(debug);
  dequant_.set_params(model_name, data_shape);

  std::vector<Tensor> const_Tensor;

  const_Tensor = {{qdq_params.data(), qdq_params_shape, "int32"}};

  dequant_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{data.data(), data_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_data_dq.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", N = " << N);
  PROFILE_THIS(dequant_.execute(input_Tensor, output_Tensor));
#else
  dequant_.execute(input_Tensor, output_Tensor);
#endif

  // compare results
  // err_count = check_add_result(cpu_mat, aie_mat, 0.1);

  float L2_norm = 0;
  float max_error = 0;
  float error_limit = 0.1;
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float diff = std::abs(bfloat2float(cpu_mat.at(r, c)) -
                            bfloat2float(aie_data_dq[r * N + c]));
      L2_norm += (diff * diff);
      if (diff > error_limit || std::isnan(diff)) {
        std::cout << "ERROR: Y[" << r << ", " << c << "]: "
                  << "Expected: " << bfloat2float(cpu_mat.at(r, c)) << ","
                  << "Received: " << bfloat2float(aie_data_dq[r * N + c]) << ","
                  << "Diff: " << diff << "\n";
        err_count++;
      }
      max_error = (diff > max_error) ? diff : max_error;
    }
  }
  if (max_error <= error_limit) {
    err_count = 0;
  }
  LOG_THIS("Maximum Difference : " << max_error);
  float RMSE = std::sqrt(L2_norm / (M * N));
  L2_norm = sqrt(L2_norm);
  std::cout << "Max Error is " << max_error << std::endl;
  std::cout << "L2_norm is " << L2_norm << std::endl;
  std::cout << "Root Mean square Error = " << RMSE << std::endl;

  return err_count;
}
#if 0
TEST(C4mzdk5_DEQUANT_Testa16, Kernel1) {
  int err_count = test_dequant<uint16_t, uint16_t>(64, 64, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_DEQUANT_Testa16, Kernel2) {
  int err_count = test_dequant<uint16_t, uint16_t>(64, 1280, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_DEQUANT_Testa16, Kernel3) {
  int err_count = test_dequant<uint16_t, uint16_t>(64, 10240, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_DEQUANT_Testa16, Kernel4) {
  int err_count = test_dequant<uint16_t, uint16_t>(256, 64, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_DEQUANT_Testa16, Kernel5) {
  int err_count = test_dequant<uint16_t, uint16_t>(256, 640, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_DEQUANT_Testa16, Kernel6) {
  int err_count = test_dequant<uint16_t, uint16_t>(256, 1280, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_DEQUANT_Testa16, Kernel7) {
  int err_count = test_dequant<uint16_t, uint16_t>(256, 10240, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_DEQUANT_Testa16, Kernel8) {
  int err_count = test_dequant<uint16_t, uint16_t>(1024, 64, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
#endif

TEST(C4mzdk5_DEQUANT_Testa16, Kernel9) {
  int err_count = test_dequant<uint16_t, uint16_t>(1024, 320, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_DEQUANT_Testa16, Kernel10) {
  int err_count = test_dequant<uint16_t, uint16_t>(1024, 640, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
#if 0
TEST(C4mzdk5_DEQUANT_Testa16, Kernel11) {
  int err_count = test_dequant<uint16_t, uint16_t>(1024, 1280, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_DEQUANT_Testa16, Kernel12) {
  int err_count = test_dequant<uint16_t, uint16_t>(1024, 5120, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_DEQUANT_Testa16, Kernel13) {
  int err_count = test_dequant<uint16_t, uint16_t>(4096, 64, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
#endif

TEST(C4mzdk5_DEQUANT_Testa16, Kernel14) {
  int err_count = test_dequant<uint16_t, uint16_t>(4096, 320, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_DEQUANT_Testa16, Kernel15) {
  int err_count = test_dequant<uint16_t, uint16_t>(4096, 640, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
#if 0
TEST(C4mzdk5_DEQUANT_Testa16, Kernel16) {
  int err_count = test_dequant<uint16_t, uint16_t>(4096, 2560, false, "uint16",
                                                   "bfloat16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
#endif
