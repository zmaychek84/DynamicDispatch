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

#include "bmm_helpers.hpp"
#include "enable_perf.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include "test_common.hpp"
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <ops/bmm/bmm.hpp>
#include <vector>

#define RANDOM_DATA
using namespace ryzenai;
using namespace matmul_matrix;

enum TRANS { NONE = 0, WTS = 1, OFM = 2 };

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_bmmd(int M, int K, int N, int B0, int B1, bool debug = false,
              const std::string &a_dtype = "bfloat16",
              const std::string &b_dtype = "bfloat16",
              const std::string &c_dtype = "bfloat16",
              const std::string &model_name = "BMM", TRANS trans = TRANS::NONE,
              const std::string &op_version = "v1") {
  int BM = M * B0;
  int err_count = 0;

  size_t BMs = static_cast<size_t>(BM);
  size_t B0s = static_cast<size_t>(B0);
  size_t B1s = static_cast<size_t>(B1);

  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);

  std::vector<size_t> a_shape = {B0s, Ms, Ks};
  std::vector<size_t> b_shape = {B1s, Ks, Ns};
  std::vector<size_t> aie_out_shape = {B0s, Ms, Ns};
  std::vector<InT> a(BM * K);
  std::vector<WgT> b(B1 * K * N);
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
    RowMajorMatrix<InT> XX(B0 * M, K, a.data());
    RowMajorMatrix<WgT> WW(B1 * K, N, b.data());
    RowMajorMatrix<uint16_t> cpu_YY(M * B0, N, cpu_out.data());
    bmm_helpers::cpu_bmmb2<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
                           RowMajorMatrix<OuT>>(XX, WW, cpu_YY, B0, B1);
  } else {
    for (int i = 0; i < B0; i++) {
      RowMajorMatrix<InT> XX(M, K, a.data() + i * M * K);
      RowMajorMatrix<InT> *WW;
      int dv = B0 / B1;
      if (trans == TRANS::WTS) {
        WW = new RowMajorMatrix<WgT>(N, K, b.data() + (i / dv) * K * N);
      } else {
        WW = new RowMajorMatrix<WgT>(K, N, b.data() + (i / dv) * K * N);
      }
      RowMajorMatrix<uint16_t> cpu_YY(M, N, cpu_out.data() + i * M * N);
      bmm_helpers::cpu_bmmb1<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
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

TEST(BMMD_Testa16w16_32_1_1024_8_1024_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 1024, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_1152_8_1152_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 1152, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_1280_8_1280_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 1280, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_1408_8_1408_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 1408, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_1536_8_1536_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 1536, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_2048_8_2048_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 2048, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_256_8_256_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 256, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_384_8_384_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 384, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_512_8_512_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 512, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_640_8_640_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 640, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_768_8_768_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 768, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_896_8_896_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 896, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_128_3072_32_3072_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      128, 3072, 128, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_3072_3072_32_3072_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      3072, 3072, 128, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_128_128_32_128_3072_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      128, 128, 3072, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_3072_128_32_128_3072_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      3072, 128, 3072, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
// psu1 bmm1
TEST(BMMD_Testa16w16_32_1_96_32_96_128_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 96, 128, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_96_32_96_256_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 96, 256, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_96_32_96_384_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 96, 384, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_96_32_96_512_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 96, 512, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_96_32_96_640_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 96, 640, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_96_32_96_768_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 96, 768, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_96_32_96_896_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 96, 896, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_96_32_96_1024_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 96, 1024, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// psu1 bmm2
TEST(BMMD_Testa16w16_32_1_128_32_128_96, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 96, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_256_32_256_96, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 256, 96, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_384_32_384_96, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 384, 96, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_512_32_512_96, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 512, 96, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_640_32_640_96, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 640, 96, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_768_32_768_96, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 768, 96, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_896_32_896_96, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 896, 96, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_1024_32_1024_96, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 1024, 96, 32, 32, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_2176_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 2176, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_2304_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 2304, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_2432_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 2432, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_2560_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 2560, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_2688_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 2688, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_2816_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 2816, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_2944_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 2944, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_3072_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 3072, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_3200_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 3200, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_3328_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 3328, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_3456_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 3456, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_3584_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 3584, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_3712_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 3712, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_3840_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 3840, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_3968_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 3968, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_4096_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 4096, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// gemma2 bmm1
TEST(BMMD_Testa16w16_8_64_256_4_256_64_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      64, 256, 64, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_8_128_256_4_256_128_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      128, 256, 128, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_8_256_256_4_256_256_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      256, 256, 256, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_8_512_256_4_256_512_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      512, 256, 512, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_8_1024_256_4_256_1024_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1024, 256, 1024, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_8_2048_256_4_256_2048_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      2048, 256, 2048, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_8_3072_256_4_256_3072_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      3072, 256, 3072, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// gemma2 bmm2
TEST(BMMD_Testa16w16_8_64_64_4_64_256, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      64, 64, 256, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_8_128_128_4_128_256, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      128, 128, 256, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_8_256_256_4_256_256, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      256, 256, 256, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_8_512_512_4_512_256, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      512, 512, 256, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_8_1024_1024_4_1024_256, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1024, 1024, 256, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_8_2048_2048_4_2048_256, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      2048, 2048, 256, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_8_3072_3072_4_3072_256, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      3072, 3072, 256, 8, 4, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// bmm2
TEST(BMMD_Testa16w16_32_1_2176_8_2176_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 2176, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_2304_8_2304_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 2304, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_2432_8_2432_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 2432, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_2560_8_2560_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 2560, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_2688_8_2688_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 2688, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_2816_8_2816_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 2816, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_2944_8_2944_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 2944, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_3072_8_3072_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 3072, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_3200_8_3200_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 3200, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_3328_8_3328_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 3328, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_3456_8_3456_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 3456, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_3584_8_3584_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 3584, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_3712_8_3712_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 3712, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_3840_8_3840_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 3840, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_3968_8_3968_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 3968, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_4096_8_4096_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 4096, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_128_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_256_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 256, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_384_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 384, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_512_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 512, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_640_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 640, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_768_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 768, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_896_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 896, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_1024_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 1024, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_1152_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 1152, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_1280_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 1280, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_1408_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 1408, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_1536_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 1536, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_1664_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 1664, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_1792_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 1792, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_1920_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 1920, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_128_8_128_2048_Transpose, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 128, 2048, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM1",
      TRANS::WTS, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_1664_8_1664_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 1664, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_1792_8_1792_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 1792, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(BMMD_Testa16w16_32_1_1920_8_1920_128, Kernel8_v1) {
  int err_count = test_bmmd<uint16_t, uint16_t, uint16_t>(
      1, 1920, 128, 32, 8, false, "bfloat16", "bfloat16", "bfloat16", "BMM2",
      TRANS::OFM, "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
