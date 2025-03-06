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

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <tuple>

#include <ops/maskedsoftmax/maskedsoftmax.hpp>

#include <stdexcept>

#include "enable_perf.hpp"

#include "maskedsoftmax_helpers.hpp"
#include "test_common.hpp"

template <typename InT = uint16_t, typename MaskT = uint16_t,
          typename OuT = uint16_t>
int test_maskedsoftmax(size_t B, size_t M, size_t K, size_t H,
                       bool debug = false,
                       const std::string &a_dtype = "bfloat16",
                       const std::string &b_dtype = "bfloat16",
                       const std::string &c_dtype = "bfloat16",
                       const std::string &model_name = "LLAMA2",
                       const std::string &op_version = "v1") {
  int err_count = 0;

  std::vector<size_t> a_shape = {B, M, K};
  std::vector<size_t> mask_shape = {1, M, K};

  std::vector<InT> a(B * M * K);
  // Range taken from
  // https://gitenterprise.xilinx.com/AIELibs/mllib/blob/dev/internal/models/python/restructured/operators/Transformers/SoftMax.py#L348
  dd::initialize_random_bfloat16(a, 5);

  std::vector<InT> mask(M * K, ryzenai::float_to_bfloat16(
                                   -std::numeric_limits<float>::infinity()));
  // zero out lower triangluar to use a casual mask
  dd::initialize_lowertriangular(mask, M, K, ryzenai::float_to_bfloat16(0.0));

  // compute golden
  std::vector<float> cpu_float = maskedsoftmax_helpers::golden_maskedsoftmax(
      {B, M, K}, a, mask,
      ryzenai::masked_softmax<uint16_t, uint16_t,
                              uint16_t>::DEFAULT_PREMASK_SCALE);

  // compute aie
  std::vector<OuT> aie_out(B * M * K,
                           /*garbage_value*/ ryzenai::float_to_bfloat16(1.0));

  std::map<std::string, std::any> attr;
  attr["op_version"] = op_version;
  attr["headsize"] = static_cast<int>(H);

  ryzenai::masked_softmax maskedsoftmax_ =
      ryzenai::masked_softmax<InT, MaskT, OuT>(a_dtype, true, attr);

  std::vector<Tensor> const_Tensor;
  std::vector<Tensor> input_Tensor;

  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor mask_T = {mask.data(), mask_shape, a_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(mask_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  maskedsoftmax_.debug(debug);
  maskedsoftmax_.initialize_const_params(const_Tensor);
  maskedsoftmax_.set_params(model_name, a_shape);
#ifdef UNIT_TEST_PERF
  LOG_THIS("B = " << B << ", M = " << M << ", K = " << K);
  PROFILE_THIS(maskedsoftmax_.execute(input_Tensor, output_Tensor));
#else
  maskedsoftmax_.execute(input_Tensor, output_Tensor);
#endif

  err_count = dd::count_errors_floatvsbfloat16(cpu_float, aie_out, a_shape,
                                               maskedsoftmax_.EPSILON);

  return err_count;
}

// v1

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x2048x2048_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 2048, 2048, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1024x1024_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1024, 1024, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x512x512_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 512, 512, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x256x256_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 256, 256, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x128x128_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 128, 128, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PHI_MASKEDSOFTMAX_Testa16, Kernel32x2048x2048_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 2048, 2048, 96, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PHI_MASKEDSOFTMAX_Testa16, Kernel32x1024x1024_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1024, 1024, 96, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PHI_MASKEDSOFTMAX_Testa16, Kernel32x512x512_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 512, 512, 96, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PHI_MASKEDSOFTMAX_Testa16, Kernel32x256x256_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 256, 256, 96, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PHI_MASKEDSOFTMAX_Testa16, Kernel32x128x128_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 128, 128, 96, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x2176_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 2176, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x2304_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 2304, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x2432_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 2432, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x2560_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 2560, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x2688_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 2688, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x2816_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 2816, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x2944_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 2944, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x3072_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 3072, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x3200_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 3200, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x3328_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 3328, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x3456_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 3456, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x3584_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 3584, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x3712_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 3712, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x3840_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 3840, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x3968_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 3968, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x4096_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 4096, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PHI_MASKEDSOFTMAX_Testa16, Kernel32x1x128_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 128, 96, false, "bfloat16", "bfloat16", "bfloat16", "PHI", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PHI_MASKEDSOFTMAX_Testa16, Kernel32x1x256_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 256, 96, false, "bfloat16", "bfloat16", "bfloat16", "PHI", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PHI_MASKEDSOFTMAX_Testa16, Kernel32x1x384_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 384, 96, false, "bfloat16", "bfloat16", "bfloat16", "PHI", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PHI_MASKEDSOFTMAX_Testa16, Kernel32x1x512_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 512, 96, false, "bfloat16", "bfloat16", "bfloat16", "PHI", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PHI_MASKEDSOFTMAX_Testa16, Kernel32x1x640_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 640, 96, false, "bfloat16", "bfloat16", "bfloat16", "PHI", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PHI_MASKEDSOFTMAX_Testa16, Kernel32x1x768_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 768, 96, false, "bfloat16", "bfloat16", "bfloat16", "PHI", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PHI_MASKEDSOFTMAX_Testa16, Kernel32x1x896_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 896, 96, false, "bfloat16", "bfloat16", "bfloat16", "PHI", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PHI_MASKEDSOFTMAX_Testa16, Kernel32x1x1024_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 1024, 96, false, "bfloat16", "bfloat16", "bfloat16", "PHI", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x128_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 128, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x256_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 256, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x384_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 384, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x512_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 512, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x640_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 640, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x768_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 768, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x896_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 896, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x1024_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 1024, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x1152_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 1152, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x1280_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 1280, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x1408_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 1408, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x1536_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 1536, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x1664_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 1664, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x1792_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 1792, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x1920_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 1920, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x1x2048_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 1, 2048, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// v2
TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x2048x2048_v2) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 2048, 2048, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel32x128x128_v2) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      32, 128, 128, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel12x64x64_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      12, 64, 64, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel12x128x128_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      12, 128, 128, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel28x64x64_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      28, 64, 64, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel28x128x128_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      28, 128, 128, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel8x128x128_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      8, 128, 128, 256, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MASKEDSOFTMAX_Testa16, Kernel8x64x64_v1) {
  int err_count = test_maskedsoftmax<uint16_t, uint16_t, uint16_t>(
      8, 64, 64, 256, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2",
      "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
