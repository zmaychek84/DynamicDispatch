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
#include <tuple>

#include "../src/ops/ops_common/matmul_matrix.hpp"
#include <ops/mladfadd/mladfadd.hpp>
#include <stdexcept>

#include "enable_perf.hpp"

#include "test_common.hpp"

using namespace matmul_matrix;
template <typename LhsT = int16_t, typename RhsT = int16_t,
          typename OuT = int16_t>
int test_mladfadd(size_t M, size_t K, bool debug = false,
                  const std::string &a_dtype = "bfloat16",
                  const std::string &b_dtype = "bfloat16",
                  const std::string &c_dtype = "bfloat16",
                  const std::string &model_name = "LLAMA2",
                  const std::string &op_version = "v1") {
  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);

  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ms, Ks};

  std::vector<LhsT> a(M * K);
  std::vector<LhsT> b(M * K);
  std::vector<float> cpu_out(M * K);
  std::vector<OuT> aie_out(M * K, garbage_value);

  dd::initialize_random_bfloat16(a, 40);
  dd::initialize_random_bfloat16(b, 40);

  // compute golden
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < K; c++) {
      cpu_out.at(r * K + c) = bfloat16_to_float(a.at(r * K + c)) +
                              bfloat16_to_float(b.at(r * K + c));
    }
  }
  std::map<std::string, std::any> attr;
  std::vector<int> size_matmul_M{1, 128, 256, 512, 1024, 2048};
  std::vector<std::vector<int>> shape_list;
  for (auto M : size_matmul_M) {
    shape_list.push_back({M, (int)K});
  }
  attr["op_version"] = op_version;
  // attr["shapes"] = shape_list;
  ryzenai::mladf_add mladfadd_ =
      ryzenai::mladf_add<LhsT, RhsT, OuT>(a_dtype, true, attr);

  std::vector<Tensor> const_Tensor;
  std::vector<Tensor> input_Tensor;

  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor b_T = {b.data(), a_shape, a_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(b_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  mladfadd_.debug(debug);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(mladfadd_.execute(input_Tensor, output_Tensor));
#else
  mladfadd_.execute(input_Tensor, output_Tensor);
#endif

  err_count = dd::count_errors_floatvsbfloat16(cpu_out, aie_out, a_shape, 4);

  return err_count;
}

// v1

TEST(LLAMA2_MLADFADD_Testa16, Kernel4096x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      4096, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel1x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      1, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel128x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      128, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel256x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      256, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel384x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      384, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel512x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      512, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel640x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      640, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel768x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      768, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel896x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      896, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel1024x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      1024, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel1152x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      1152, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel1280x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      1280, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel1408x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      1408, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel1536x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      1536, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel1664x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      1664, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel1792x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      1792, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel1920x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      1920, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFADD_Testa16, Test2047) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      2047, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFADD_Testa16, Test3071) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      3071, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFADD_Testa16, Kernel2048x4096_v1) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      2048, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFADD_Testa16, Kernel300) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      300, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFADD_Testa16, Kernel600) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      600, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFADD_Testa16, Kernel767) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      767, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFADD_Testa16, Kernel850) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      850, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// v2

TEST(LLAMA2_MLADFADD_Testa16, Kernel1x4096_v2) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      1, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_MLADFADD_Testa16, Kernel128x4096_v2) {
  int err_count = test_mladfadd<uint16_t, uint16_t, uint16_t>(
      128, 4096, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
