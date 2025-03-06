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

#include "enable_perf.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include "test_common.hpp"
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <ops/bmm/bmm.hpp>
#include <vector>

#include "bmm_helpers.hpp"

#define RANDOM_DATA

using namespace ryzenai;
using namespace matmul_matrix;

enum TRANS { NONE = 0, WTS = 1, OFM = 2 };

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_bmm(int M, int K, int N, int B = 32, bool debug = false,
             const std::string &a_dtype = "bfloat16",
             const std::string &b_dtype = "bfloat16",
             const std::string &c_dtype = "bfloat16",
             const std::string &model_name = "BMM", TRANS trans = TRANS::NONE,
             const std::string &op_version = "v1") {
  int BM = M * B;
  int err_count = 0;

  size_t BMs = static_cast<size_t>(BM);
  size_t Bs = static_cast<size_t>(B);
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);

  std::vector<size_t> a_shape = {Bs, Ms, Ks};
  std::vector<size_t> b_shape = {Bs, Ks, Ns};
  std::vector<size_t> aie_out_shape = {Bs, Ms, Ns};
  std::vector<InT> a(BM * K);
  std::vector<WgT> b(B * K * N);
  std::vector<uint16_t> cpu_out(BM * N);
  std::vector<OuT> aie_out(BM * N, garbage_value);
  RowMajorMatrix<InT> X(BM, K, a.data());
  RowMajorMatrix<WgT> *W;
  RowMajorMatrix<uint16_t> cpu_Y(BM, N, cpu_out.data());
  RowMajorMatrix<OuT> aie_Y(BM, N, aie_out.data());

  srand(0xABCD);
  dd::initialize_random_bfloat16(a, 1.5);
  dd::initialize_random_bfloat16(b, 1.5);
  bool bmm_transpose_flag = trans == TRANS::WTS;
  std::map<std::string, std::any> attr;
  attr["op_version"] = op_version;
  ryzenai::bmm bmm_ = ryzenai::bmm<InT, WgT, OuT>(
      a_dtype, b_dtype, c_dtype, false, bmm_transpose_flag, attr);
  bmm_.debug(debug);
  bmm_.set_params(model_name, a_shape, b_shape);
  std::vector<Tensor> const_Tensor;
  const_Tensor = {{b.data(), b_shape, b_dtype}};
  bmm_.initialize_const_params(const_Tensor);
  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};
  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
  PROFILE_THIS({ bmm_.execute(input_Tensor, output_Tensor); });
#else
  bmm_.execute(input_Tensor, output_Tensor);
#endif

  if (trans == TRANS::OFM) {
    RowMajorMatrix<InT> XX(B * M, K, a.data());
    RowMajorMatrix<WgT> WW(B * K, N, b.data());
    RowMajorMatrix<uint16_t> cpu_YY(M * B, N, cpu_out.data());
    bmm_helpers::cpu_bmm2<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
                          RowMajorMatrix<OuT>>(XX, WW, cpu_YY, B);
  } else {
    for (int i = 0; i < B; i++) {
      RowMajorMatrix<InT> XX(M, K, a.data() + i * M * K);
      RowMajorMatrix<InT> *WW;
      if (trans == TRANS::WTS) {
        WW = new RowMajorMatrix<WgT>(N, K, b.data() + i * K * N);
      } else {
        WW = new RowMajorMatrix<WgT>(K, N, b.data() + i * K * N);
      }
      RowMajorMatrix<uint16_t> cpu_YY(M, N, cpu_out.data() + i * M * N);
      bmm_helpers::cpu_bmm<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
                           RowMajorMatrix<OuT>>(XX, *WW, cpu_YY,
                                                bmm_transpose_flag);
      std::cout << ".";
    }
    std::cout << std::endl;
  }
  err_count =
      check_add_result_bfloat16<OuT>(cpu_out, aie_out, aie_out_shape, 1.0);
  return err_count;
}

// BMM a16w16 - v1 with wts transpose
TEST(BMM_Testa16w16_2048_128_2048_Transpose, Kernel1_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      2048, 128, 2048, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMM_Testa16w16_1024_128_1024_Transpose, Kernel2_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      1024, 128, 1024, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMM_Testa16w16_512_128_512_Transpose, Kernel3_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      512, 128, 512, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMM_Testa16w16_256_128_256_Transpose, Kernel4_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      256, 128, 256, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMM_Testa16w16_128_128_128_Transpose, Kernel4_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      128, 128, 128, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// BMM a16w16 - v1 with output transpose
TEST(BMM_Testa16w16_2048_2048_128, Kernel5_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      2048, 2048, 128, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMM_Testa16w16_1024_1024_128, Kernel6_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      1024, 1024, 128, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMM_Testa16w16_512_512_128, Kernel7_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      512, 512, 128, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMM_Testa16w16_256_256_128, Kernel8_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      256, 256, 128, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMM_Testa16w16_128_128_128, Kernel8_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      128, 128, 128, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// new shapes

TEST(BMM_Testa16w16_128_256_128, Kernel8_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      128, 256, 128, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMM_Testa16w16_128_384_128, Kernel8_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      128, 384, 128, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMM_Testa16w16_128_512_128, Kernel8_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      128, 512, 128, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMM_Testa16w16_128_640_128, Kernel8_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      128, 640, 128, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMM_Testa16w16_128_768_128, Kernel8_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      128, 768, 128, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMM_Testa16w16_128_896_128, Kernel8_v1) {
  int err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
      128, 896, 128, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
