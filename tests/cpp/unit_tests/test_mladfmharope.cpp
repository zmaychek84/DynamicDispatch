/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <ops/mladfmharope/mladfmharope.hpp>
#include <tuple>

#include "enable_perf.hpp"
#include "test_common.hpp"
#include <stdexcept>

void goldenRope(size_t B, size_t M, size_t K, std::vector<float> *a_float,
                std::vector<float> *wts_float, std::vector<float> *cpu_float) {
  for (int h = 0; h < B; h++) {
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < K / 2; i++) {
        float x0 = a_float->at(h * M * K + j * K + i);
        float x1 = a_float->at(h * M * K + j * K + i + K / 2);
        float w0 = wts_float->at(j * K + i);
        float w1 = wts_float->at(M * K + j * K + i);
        cpu_float->at(h * M * K + j * K + i) = x0 * w0 - x1 * w1;
        cpu_float->at(h * M * K + j * K + i + K / 2) = x1 * w0 + x0 * w1;
      }
    }
  }
}

void goldenRopeT(size_t B, size_t M, size_t K, std::vector<float> *a_float,
                 std::vector<float> *wts_float, std::vector<float> *cpu_float) {
  for (int h = 0; h < B; h++) {
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < K / 2; i++) {
        float x0 = a_float->at(h * K + j * B * K + i);
        float x1 = a_float->at(h * K + j * B * K + i + K / 2);
        float w0 = wts_float->at(i + j * K);
        float w1 = wts_float->at(M * K + i + j * K);
        cpu_float->at(h * M * K + j * K + i) = x0 * w0 - x1 * w1;
        cpu_float->at(h * M * K + j * K + i + K / 2) = x1 * w0 + x0 * w1;
      }
    }
  }
}

void goldenRopeA(size_t B, size_t M, size_t K, std::vector<float> *a_float,
                 std::vector<float> *wts_float, std::vector<float> *cpu_float) {
  for (int h = 0; h < B; h++) {
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < K / 2; i++) {
        float x0 = a_float->at(h * M * K + j * K + i);
        float x1 = a_float->at(h * M * K + j * K + i + K / 2);
        float w0 = wts_float->at(i + (j / B) * K + h * K * M / B);
        float w1 = wts_float->at(M * K + i + (j / B) * K + h * K * M / B);
        cpu_float->at(h * M * K + j * K + i) = x0 * w0 - x1 * w1;
        cpu_float->at(h * M * K + j * K + i + K / 2) = x1 * w0 + x0 * w1;
      }
    }
  }
}

void goldenRopeG1(size_t B, size_t M, size_t K, std::vector<float> *a_float,
                  std::vector<float> *wts_float,
                  std::vector<float> *cpu_float) {
  for (int h = 0; h < B; h++) {
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < K / 4; i++) {
        float x0 = a_float->at(h * M * K + j * K + (2 * i));
        float x1 = a_float->at(h * M * K + j * K + (2 * i + 1));
        float x2 = a_float->at(h * M * K + j * K + (2 * i) + K / 2);
        float x3 = a_float->at(h * M * K + j * K + (2 * i + 1) + K / 2);

        float w0 = wts_float->at(j * K + (2 * i));
        float w1 = wts_float->at(j * K + (2 * i + 1));
        float w2 = wts_float->at(j * K + K / 2 + (2 * i));
        float w3 = wts_float->at(j * K + K / 2 + (2 * i + 1));

        cpu_float->at(h * M * K + j * K + (2 * i)) = x0 * w0 + x1 * w2;
        cpu_float->at(h * M * K + j * K + (2 * i + 1)) = x1 * w1 + x0 * w3;
        cpu_float->at(h * M * K + j * K + (2 * i) + K / 2) = x2;
        cpu_float->at(h * M * K + j * K + (2 * i + 1) + K / 2) = x3;
      }
    }
  }
}

template <typename InT = uint16_t, typename TrigT = uint16_t,
          typename OuT = uint16_t>
int test_mladfmharope(size_t B, size_t M, size_t K, bool debug = false,
                      const std::string &a_dtype = "bfloat16",
                      const std::string &b_dtype = "bfloat16",
                      const std::string &c_dtype = "bfloat16",
                      const std::string &model_name = "LLAMA2",
                      const std::string &op_version = "v1",
                      const std::string &transpose = "") {
  int err_count = 0;

  std::vector<size_t> a_shape = {B, M, K};
  std::vector<size_t> trig_shape = {2, M, K};
  std::vector<float> a_float(B * M * K, 0.0f);
  std::vector<float> trig_float(2 * M * K, 0.0f);
  std::vector<float> cpu_float(B * M * K, 0.0f);

  // init random data
  for (int i = 0; i < B * M * K; i++) {
    a_float.at(i) = 2.0f * (rand() / 65536.0f) - 1.0f;
  }
  for (int i = 0; i < 2 * M * K; i++) {
    trig_float.at(i) = 2.0f * (rand() / 65536.0f) - 1.0f;
  }

  std::vector<InT> a(B * M * K, ryzenai::float_to_bfloat16(1.0f));
  std::vector<InT> trig(2 * M * K, ryzenai::float_to_bfloat16(1.0f));

  for (int i = 0; i < a_float.size(); ++i) {
    a.at(i) = ryzenai::float_to_bfloat16(a_float.at(i));
  }
  for (int i = 0; i < trig_float.size(); ++i) {
    trig.at(i) = ryzenai::float_to_bfloat16(trig_float.at(i));
  }
  if (transpose == "input") {
    goldenRopeT(B, M, K, &a_float, &trig_float, &cpu_float);
  } else if (transpose == "all") {
    goldenRopeA(B, M, K, &a_float, &trig_float, &cpu_float);
  } else if (transpose == "glm") {
    goldenRopeG1(B, M, K, &a_float, &trig_float, &cpu_float);
  } else {
    goldenRope(B, M, K, &a_float, &trig_float, &cpu_float);
  }
  // compute aie
  std::vector<OuT> aie_out(B * M * K, garbage_value);
  std::map<std::string, std::any> attr;
  attr["op_version"] = op_version;
  attr["transpose"] = transpose;
  ryzenai::mha_rope mladfmharope_ =
      ryzenai::mha_rope<InT, TrigT, OuT>(a_dtype, true, attr);
  std::vector<Tensor> input_Tensor;
  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor trig_T = {trig.data(), trig_shape, a_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(trig_T);
  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);
  mladfmharope_.debug(debug);
  mladfmharope_.set_params("rope", a_shape);
#ifdef UNIT_TEST_PERF
  LOG_THIS("B = " << B << ", M = " << M << ", K = " << K);
  PROFILE_THIS(mladfmharope_.execute(input_Tensor, output_Tensor));
#else
  mladfmharope_.execute(input_Tensor, output_Tensor);
#endif
  mladfmharope_.EPSILON = 0.011;
  err_count = dd::count_errors_floatvsbfloat16(cpu_float, aie_out, a_shape,
                                               mladfmharope_.EPSILON);
  return err_count;
}

TEST(GLM_MLADFMHAROPE_Testa16, Kernel32x1x128_v1_glm) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 1, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "glm");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x4096x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 4096, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x2048x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 2048, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x1280x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 1280, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x1152x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 1152, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x1024x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 1024, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x896x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 896, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x768x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 768, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x640x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 640, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x512x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 512, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x384x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 384, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x256x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 256, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x128x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 128, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x8x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 8, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x1x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 1, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x4096x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 4096, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x384x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 384, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x512x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 512, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x128x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 128, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x8x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 8, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x1x128_v1) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 1, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x4096x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 4096, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x2048x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 2048, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x1920x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 1920, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x1792x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 1792, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x1024x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 1024, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x896x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 896, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x8x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 8, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x128x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 128, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x1x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 1, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x4096x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 4096, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x2048x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 2048, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x1920x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 1920, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x1792x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 1792, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x1024x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 1024, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x896x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 896, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x8x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 8, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x128x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 128, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x1x128_v1_transpose) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 1, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "input");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x2048x128_v1_all) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 2048, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "all");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x1920x128_v1_all) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 1920, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "all");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x1792x128_v1_all) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 1792, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "all");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x1024x128_v1_all) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 1024, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "all");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x896x128_v1_all) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 896, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "all");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel32x128x128_v1_all) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      32, 128, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "all");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x8x128_v1_all) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 8, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "all");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x768x128_v1_all) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 768, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "all");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x512x128_v1_all) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 512, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "all");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x896x128_v1_all) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 896, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "all");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_MLADFMHAROPE_Testa16, Kernel8x128x128_v1_all) {
  int err_count = test_mladfmharope<uint16_t, uint16_t, uint16_t>(
      8, 128, 128, false, "bfloat16", "bfloat16", "bfloat16", "LLAMA2", "v1",
      "all");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
