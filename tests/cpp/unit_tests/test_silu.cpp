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

TEST(LLAMA2_SILU_V1, AutoRunAllTxnShapes) {
  // Create an instance of the silu operator that will discover supported
  // shapes.
  using SiluOp = ryzenai::silu<uint16_t, uint16_t>;
  SiluOp shapeFinderOp("bfloat16", true, std::map<std::string, std::any>());

  // Retrieve the discovered shapes.
  auto shapes = shapeFinderOp.get_supported_shapes();

  // Optionally, build and apply a skip set if some shapes need to be excluded.
  // auto skipSet = buildSkipSet_silu();
  // shapes.erase(
  //     std::remove_if(shapes.begin(), shapes.end(), [&](const auto &s) {
  //       std::string key = shapeToKey(std::get<0>(s), std::get<1>(s));
  //       return skipSet.find(key) != skipSet.end();
  //     }),
  //     shapes.end());

  // Loop over each discovered shape and run the test.
  for (const auto &s : shapes) {
    int M = std::get<0>(s);
    int K = std::get<1>(s);

    int err_count = test_silu<uint16_t, uint16_t>(M, K,
                                                  /*debug=*/false,
                                                  /*a_dtype=*/"bfloat16",
                                                  /*c_dtype=*/"bfloat16",
                                                  /*model_name=*/"LLAMA2",
                                                  /*op_version=*/"v1",
                                                  /*use_reference_data=*/false);

    EXPECT_EQ(err_count, 0) << "[test_silu] Error count = " << err_count
                            << " for shape M=" << M << ", K=" << K;
  }
}
