/*
 Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 Licensed under the MIT License.
 */

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <tuple>

#include "mladfsoftmax_helpers.hpp"
#include <ops/mladfrmsnorm/mladfrmsnorm.hpp>

#include "enable_perf.hpp"
#include "test_common.hpp"
#include <stdexcept>

void goldenRmsNorm(size_t M, size_t K, uint16_t *a, uint16_t *wts,
                   uint16_t *g_ref) {
  float r;
  float rm;
  float rms;
  for (int j = 0; j < M; j++) {
    r = 0.0;
    rm = 0.0;
    rms = 0.0;
    for (int i = 0; i < K; i++) {
      r = r + ryzenai::bfloat16_to_float(a[j * K + i]) *
                  ryzenai::bfloat16_to_float(a[j * K + i]);
    }
    rm = r / float(K);
    rms = (sqrt(rm) + 0.000001);
    for (int i = 0; i < K; i++) {
      float res = ryzenai::bfloat16_to_float(wts[i]) *
                  ryzenai::bfloat16_to_float(a[j * K + i]) / rms;
      g_ref[j * K + i] = ryzenai::float_to_bfloat16(res);
    }
  }
}

template <typename InT = uint16_t, typename WtsT = uint16_t,
          typename OuT = uint16_t>
int test_mladfrmsnormRand(size_t M, size_t K, bool debug = false,
                          const std::string &a_dtype = "bfloat16",
                          const std::string &b_dtype = "bfloat16",
                          const std::string &c_dtype = "bfloat16",
                          const std::string &model_name = "LLAMA2",
                          const std::string &op_version = "v1") {
  int err_count = 0;
  std::vector<size_t> a_shape = {M, K};
  std::vector<size_t> wts_shape = {K};
  std::vector<InT> a(M * K);
  std::vector<InT> wts(K);

  // compute aie
  std::vector<OuT> aie_out(M * K, garbage_value);
  std::vector<OuT> golden_ref(M * K, garbage_value);
  std::map<std::string, std::any> attr;
  std::vector<int> size_matmul_M{1, 128, 256, 512, 1024, 2048};
  std::vector<std::vector<int>> shape_list;
  for (auto M : size_matmul_M) {
    shape_list.push_back({M, (int)K});
  }
  attr["op_version"] = op_version;
  attr["shapes"] = shape_list;

  ryzenai::rms_norm mladfrmsnorm_ =
      ryzenai::rms_norm<InT, WtsT, OuT>(a_dtype, true, attr);

  std::vector<Tensor> const_Tensor;
  std::vector<Tensor> input_Tensor;
  struct Tensor in_T = {a.data(), a_shape, a_dtype};
  struct Tensor wts_T = {wts.data(), wts_shape, a_dtype};
  struct Tensor ou_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(in_T);
  input_Tensor.push_back(wts_T);
  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(ou_T);
  uint16_t *ip = (uint16_t *)a.data();
  uint16_t *wt = (uint16_t *)wts.data();
  uint16_t *gr = (uint16_t *)golden_ref.data();
  dd::initialize_random_bfloat16(a, 15.0);
  dd::initialize_random_bfloat16(wts, 23.0);
  goldenRmsNorm(M, K, ip, wt, gr);

  mladfrmsnorm_.debug(debug);
  mladfrmsnorm_.initialize_const_params(const_Tensor);
#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(mladfrmsnorm_.execute(input_Tensor, output_Tensor));
#else
  mladfrmsnorm_.execute(input_Tensor, output_Tensor);
#endif
  uint16_t *ct = (uint16_t *)aie_out.data();
  err_count =
      dd::count_errors_bfloat16vsbfloat16(golden_ref, aie_out, a_shape, 0.51);
  return err_count;
}

template <typename InT = uint16_t, typename WtsT = uint16_t,
          typename OuT = uint16_t>
int test_mladfrmsnorm(size_t M, size_t K, bool debug = false,
                      const std::string &a_dtype = "bfloat16",
                      const std::string &b_dtype = "bfloat16",
                      const std::string &c_dtype = "bfloat16",
                      const std::string &model_name = "LLAMA2",
                      const std::string &op_version = "v0") {
  int err_count = 0;
  std::vector<size_t> a_shape = {M, K};
  std::vector<size_t> wts_shape = {K};
  std::vector<InT> a(M * K);
  std::vector<InT> wts(K);

  // compute aie
  std::vector<OuT> aie_out(M * K, garbage_value);
  std::vector<OuT> reference_out(M * K);

  // Using golden teste vectors from MLLIB
  // https://gitenterprise.xilinx.com/AIELibs/mllib/tree/716e81ac7bf6fd135c86d54eb51435c6a1a3f403/internal/examples/rmsnorm_2x4x4/data
  std::string data_path_prefix = OpInterface::get_dd_base_dir() + "\\" +
                                 "tests" + "\\" + "cpp" + "\\" + "unit_tests" +
                                 "\\" + "testDataMladf" + "\\" +
                                 "llama2_2x4x4_mladfrmsnorm_2048_4096" + "\\";
  std::string a_bin_path = data_path_prefix + "ifm32.bin";
  std::string wts_bin_path = data_path_prefix + "wts32.bin";
  std::string ofm_bin_path = data_path_prefix + "ofm32.bin";

  mladfsoftmax_helpers::read_bin_to_vector(a_bin_path, a);
  mladfsoftmax_helpers::read_bin_to_vector(wts_bin_path, wts);
  mladfsoftmax_helpers::read_bin_to_vector(ofm_bin_path, reference_out);

  std::map<std::string, std::any> attr;
  attr["op_version"] = op_version;
  ryzenai::rms_norm mladfrmsnorm_ =
      ryzenai::rms_norm<InT, WtsT, OuT>(a_dtype, true, attr);

  std::vector<Tensor> const_Tensor;
  std::vector<Tensor> input_Tensor;

  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor wts_T = {wts.data(), wts_shape, a_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(wts_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  mladfrmsnorm_.debug(debug);
  mladfrmsnorm_.initialize_const_params(const_Tensor);
#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(mladfrmsnorm_.execute(input_Tensor, output_Tensor));
#else
  mladfrmsnorm_.execute(input_Tensor, output_Tensor);
#endif

  err_count = dd::count_errors_bfloat16vsbfloat16(
      reference_out, aie_out, a_shape, mladfrmsnorm_.EPSILON);

  return err_count;
}

// v1
TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel2048x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      2048, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel1024x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      1024, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel512x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      512, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel256x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      256, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel128x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      128, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel32x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      32, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel1x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      1, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel384x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      384, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel640x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      640, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel768x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      768, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel1152x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      1152, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel1280x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      1280, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel1408x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      1408, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel1536x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      1536, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel1664x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      1664, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel1792x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      1792, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel1920x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      1920, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// tiling
TEST(LLAMA2_MLADFRMSNORM_RAND_Testa16, Kernel4096x4096_v1) {
  int err_count = test_mladfrmsnormRand<uint16_t, uint16_t, uint16_t>(
      4096, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
