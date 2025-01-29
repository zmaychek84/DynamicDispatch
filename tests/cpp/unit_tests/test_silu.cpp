/*
 Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 Licensed under the MIT License.
 */

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

#include "../src/ops/ops_common/matmul_matrix.hpp"
#include <ops/silu/silu.hpp>

#include "enable_perf.hpp"

#include "test_common.hpp"

namespace {
// Function to count the number of lines in the file
size_t countLines(const std::string &filename) {
  std::ifstream file(filename);
  size_t lineCount = 0;
  std::string line;
  while (std::getline(file, line)) {
    ++lineCount;
  }
  return lineCount;
}
// Function to load hex values from a file into a vector
bool loadHexValues(const std::string &filename,
                   std::vector<uint16_t> &hexValues, float force_value) {
  size_t lineCount = countLines(filename);
  hexValues.resize(lineCount * 2); // Each line contains 2 hex values
  bool do_once = false;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file." << std::endl;
    return false;
  }

  std::string line;
  size_t index = 0;
  while (std::getline(file, line)) {
    if (line.length() != 8) {
      std::cerr << "Invalid line length: " << line << std::endl;
      continue;
    }

    std::string highStr = line.substr(0, 4);
    std::string lowStr = line.substr(4, 4);

    uint16_t highValue;
    uint16_t lowValue;

    std::stringstream highConverter;
    std::stringstream lowConverter;

    highConverter << std::hex << highStr;
    highConverter >> highValue;

    lowConverter << std::hex << lowStr;
    lowConverter >> lowValue;
    if (!do_once) {
      std::cout << " highValue " << ryzenai::bfloat16_to_float(highValue)
                << std::endl;
      printf("highValue %f, %d \n", ryzenai::bfloat16_to_float(highValue),
             highValue);
      std::cout << " lowValue " << ryzenai::bfloat16_to_float(lowValue)
                << std::endl;
      printf("lowValue %f, %d \n", ryzenai::bfloat16_to_float(lowValue),
             highValue);
      do_once = true;
    }

    hexValues.at(index++) = ryzenai::float_to_bfloat16(
        force_value); //(highValue==0&&force_value)?
                      // float_to_bfloat16(6.0f):highValue;
    hexValues.at(index++) = ryzenai::float_to_bfloat16(
        force_value); //(lowValue==0&&force_value)?
                      // float_to_bfloat16(6.0f):lowValue;
  }

  file.close();
  return true;
}
} // namespace

template <typename InT = uint16_t, typename OuT = uint16_t>
int test_silu(size_t M, size_t K, bool debug = false,
              const std::string &a_dtype = "bfloat16",
              const std::string &c_dtype = "bfloat16",
              const std::string &model_name = "LLAMA2",
              const std::string &op_version = "v1",
              const bool use_reference_data = false) {
  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);

  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ms, Ks};

  std::vector<InT> a(M * K);
  std::vector<InT> b(M * K);
  std::vector<float> cpu_float(M * K);
  std::vector<OuT> aie_out(M * K, garbage_value);

  dd::initialize_random_bfloat16(a, 42);
  // compute golden
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < K; c++) {
      float x = ryzenai::bfloat16_to_float(a.at(r * K + c));
      float sigmoid = 1 / (std::exp(-x) + 1);
      float fSilu = x * sigmoid;
      cpu_float.at(r * K + c) = fSilu;
    }
  }

  std::map<std::string, std::any> attr;
  std::vector<int> size_matmul_M{1, 128, 256, 512, 1024, 2048};
  std::vector<std::vector<int>> shape_list;
  for (auto m : size_matmul_M) {
    if (K <= 14336) {
      shape_list.push_back({m, (int)K});
    } else {
      shape_list.push_back({m, 14336});
    }
  }
  attr["op_version"] = op_version;
  // attr["shapes"] = shape_list;

  ryzenai::silu silu_ = ryzenai::silu<InT, OuT>(a_dtype, true, attr);

  std::vector<Tensor> const_Tensor;

  std::vector<Tensor> input_Tensor;
  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  silu_.debug(debug);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(silu_.execute(input_Tensor, output_Tensor));
#else
  silu_.execute(input_Tensor, output_Tensor);
#endif
  err_count = dd::count_errors_floatvsbfloat16(cpu_float, aie_out, a_shape,
                                               silu_.EPSILON);
  return err_count;
}

// v1
TEST(LLAMA2_SILU_Testa16, Kernel2048x11008_v1) {
  int err_count = test_silu<uint16_t, uint16_t>(2048, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1024x11008_v1) {
  int err_count = test_silu<uint16_t, uint16_t>(1024, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel512x11008_v1) {
  int err_count = test_silu<uint16_t, uint16_t>(512, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel256x11008_v1) {
  int err_count = test_silu<uint16_t, uint16_t>(256, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel128x11008_v1) {
  int err_count = test_silu<uint16_t, uint16_t>(128, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1x11008_v1) {
  int err_count = test_silu<uint16_t, uint16_t>(1, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// new shapes

TEST(LLAMA2_SILU_Testa16, Kernel384x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(384, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel640x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(640, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel768x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(768, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel896x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(896, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1152x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(1152, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1280x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(1280, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1408x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(1408, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1536x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(1536, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1664x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(1664, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1792x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(1792, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1920x11008) {
  int err_count = test_silu<uint16_t, uint16_t>(1920, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// new shapes V1

TEST(LLAMA2_SILU_Testa16, Kernel384x11008_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(384, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel640x11008_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(640, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel768x11008_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(768, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel896x11008_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(896, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1152x11008_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1152, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1280x11008_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1280, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1408x11008_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1408, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1536x11008_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1536, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1664x11008_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1664, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1792x11008_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1792, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1920x11008_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1920, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SiLU shapes for X14336

TEST(LLAMA2_SILU_Testa16, Kernel2048x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(2048, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1024x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(1024, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel512x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(512, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel256x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(256, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel128x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(128, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(1, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// v1
TEST(LLAMA2_SILU_Testa16, Kernel2048x14336_v1) {
  int err_count = test_silu<uint16_t, uint16_t>(2048, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1024x14336_v1) {
  int err_count = test_silu<uint16_t, uint16_t>(1024, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel512x14336_v1) {
  int err_count = test_silu<uint16_t, uint16_t>(512, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel256x14336_v1) {
  int err_count = test_silu<uint16_t, uint16_t>(256, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel128x14336_v1) {
  int err_count = test_silu<uint16_t, uint16_t>(128, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1x14336_v1) {
  int err_count = test_silu<uint16_t, uint16_t>(1, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// new shapes

TEST(LLAMA2_SILU_Testa16, Kernel384x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(384, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel640x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(640, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel768x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(768, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel896x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(896, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1152x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(1152, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1280x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(1280, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1408x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(1408, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1536x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(1536, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1664x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(1664, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1792x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(1792, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1920x14336) {
  int err_count = test_silu<uint16_t, uint16_t>(1920, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// new shapes V1

TEST(LLAMA2_SILU_Testa16, Kernel384x14336_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(384, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel640x14336_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(640, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel768x14336_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(768, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel896x14336_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(896, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1152x14336_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1152, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1280x14336_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1280, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1408x14336_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1408, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1536x14336_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1536, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1664x14336_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1664, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1792x14336_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1792, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LLAMA2_SILU_Testa16, Kernel1920x14336_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(1920, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// tiling
TEST(LLAMA2_SILU_Testa16, Kernel4096x11008_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(4096, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_SILU_Testa16, Kernel300) {
  int err_count = test_silu<uint16_t, uint16_t>(300, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_SILU_Testa16, Kernel600) {
  int err_count = test_silu<uint16_t, uint16_t>(600, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_SILU_Testa16, Kernel767) {
  int err_count = test_silu<uint16_t, uint16_t>(767, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_SILU_Testa16, Kernel850) {
  int err_count = test_silu<uint16_t, uint16_t>(850, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_SILU_Testa16, Kernel2047) {
  int err_count = test_silu<uint16_t, uint16_t>(2047, 11008, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_SILU_Testa16, Kernel2048x20000_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(2048, 20000, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_SILU_Testa16, Kernel4096x14336_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(4096, 14336, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(LLAMA2_SILU_Testa16, Kernel4096x28672_V1) {
  int err_count = test_silu<uint16_t, uint16_t>(4096, 28672, false, "bfloat16",
                                                "bfloat16", "LLAMA2", "v1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
