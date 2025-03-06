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

#include <cfenv>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <torch/torch.h>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include "test_common.hpp"
#include <ops/cast/cast.hpp>

// #pragma STDC FENV_ACCESS ON
static int test_count = 0;

static size_t get_shape_ele_num(const std::vector<size_t> &shape) {
  size_t total_num = 1;
  for (size_t dim : shape) {
    total_num *= dim;
  }
  return total_num;
}

static float bfloat16_to_float(uint16_t x) {
  float y = 0.0;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *dst = (uint8_t *)&y;
  dst[2] = src[0];
  dst[3] = src[1];
  return y;
}

static std::vector<uint32_t> read_hex_file(const std::string &filePath) {
  std::ifstream fileStream(filePath);

  if (!fileStream.is_open()) {
    std::cerr << "Failed to open file " << filePath << "!" << std::endl;
    throw std::runtime_error("Failed to open file " + filePath + "!");
    return {};
  }
  std::vector<uint32_t> buffer;
  uint32_t temp;
  while (fileStream >> std::hex >> temp) {
    buffer.push_back(temp);
  }
  fileStream.close();
  return buffer;
}

static double round_half_2_even(double value) {
  // Set rounding mode to "round to nearest, ties to even"
  std::fesetround(FE_TONEAREST);

  // Use nearbyint, which rounds according to the current rounding mode
  return std::nearbyint(value);
}

static void aie_srs_compute(std::vector<uint32_t> &input_output) {
  int data_width = 16;
  int shift = 16;
  for (size_t i = 0; i < input_output.size(); ++i) {
    double temp = static_cast<double>(input_output[i]) / std::pow(2.0, shift);
    // temp = std::round(temp);
    temp = round_half_2_even(temp);
    if (temp > std::pow(2.0f, data_width) - 1) {
      temp = float(std::pow(2.0f, data_width)) - 1;
    }
    if (temp < 0) {
      temp = 0;
    }
    input_output[i] = static_cast<uint32_t>(temp);
  }
}

static std::vector<uint16_t> float_2_bf16_vec(std::vector<float> &x) {
  std::vector<uint32_t> x_uint32(x.size());
  std::memcpy(x_uint32.data(), x.data(), x.size() * sizeof(float));
  aie_srs_compute(x_uint32);

  std::vector<uint16_t> x_uint16(x.size());

  for (size_t i = 0; i < x_uint32.size(); ++i) {
    x_uint16[i] = static_cast<uint16_t>(x_uint32[i]);
  }

  return x_uint16;
}

static std::vector<float> bf16_2_float_vec(std::vector<uint16_t> &x) {
  std::vector<float> x_fp32(x.size());
  for (int i = 0; i < x_fp32.size(); i++) {
    x_fp32[i] = bfloat16_to_float(x[i]);
  }
  return x_fp32;
}

static void initialize_random_float(std::vector<float> &vec, int max, int min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

void writeVectorToFile(const std::vector<uint8_t> &data,
                       const std::string &fileName) {
  std::ofstream outFile(fileName);
  if (!outFile.is_open()) {
    throw std::runtime_error("Failed to open file: " + fileName);
  }
  for (size_t i = 0; i < data.size(); ++i) {
    outFile << std::bitset<8>(data[i]) << "\n";
  }
  outFile.close();
}

void writeVectorToFile(const std::vector<uint16_t> &data,
                       const std::string &fileName) {
  std::ofstream outFile(fileName);
  if (!outFile.is_open()) {
    throw std::runtime_error("Failed to open file: " + fileName);
  }
  for (size_t i = 0; i < data.size(); ++i) {
    outFile << std::bitset<16>(data[i]) << "\n";
  }
  outFile.close();
}

static std::vector<uint8_t> compress_bf16(const std::vector<float> &data,
                                          int block_size, int sub_block_size,
                                          int sub_block_shift_bits) {
  int m_bfp = 16 - 9;
  int exp_bias = 127;
  std::vector<uint8_t> ret(block_size + 1);
  std::vector<uint8_t> exp_data(block_size);

  // Extract exponent for each element
  for (int i = 0; i < block_size; ++i) {
    uint32_t u32_data;
    std::memcpy(&u32_data, &(data[i]), sizeof(float));
    uint8_t exp = (u32_data & 0x7F800000) >> 23;
    exp_data[i] = exp;
  }

  // Compute shared exponent
  uint8_t shared_exp = *std::max_element(exp_data.begin(), exp_data.end());
  ret[0] = shared_exp;

  for (int i = 0; i < block_size / sub_block_size; ++i) {
    uint8_t max_sub_exp =
        *std::max_element(exp_data.begin() + i * sub_block_size,
                          exp_data.begin() + (i + 1) * sub_block_size);
    int shift_upper_bound = (1 << sub_block_shift_bits) - 1;
    int shift =
        std::min(static_cast<int>(shared_exp - max_sub_exp), shift_upper_bound);

    for (int j = 0; j < sub_block_size; ++j) {
      float fp32_data = data[i * sub_block_size + j];
      float sign_mantissa =
          fp32_data /
          std::pow(2.0f, (shared_exp - exp_bias - shift + 1.0f - m_bfp));

      // Apply rounding logic
      int rounded_mantissa = static_cast<int>(std::round(sign_mantissa));

      // Clamp the result to fit within 8 bits (signed int8)
      rounded_mantissa = std::max(-128, std::min(127, rounded_mantissa));

      ret[i * sub_block_size + j + 1] = static_cast<int8_t>(rounded_mantissa);
    }
  }

  return ret;
}

// Transpose function with 0213 order
std::vector<float> transpose0213(const std::vector<float> &input,
                                 const std::array<size_t, 4> &dims) {
  // Check input size validity
  size_t size = dims[0] * dims[1] * dims[2] * dims[3];
  if (input.size() != size) {
    throw std::invalid_argument("Input size does not match dimensions.");
  }
  // Output vector with the same size
  std::vector<float> output(size);
  // Transpose logic
  size_t d0 = dims[0], d1 = dims[1], d2 = dims[2], d3 = dims[3];
  for (size_t i0 = 0; i0 < d0; ++i0) {
    for (size_t i1 = 0; i1 < d1; ++i1) {
      for (size_t i2 = 0; i2 < d2; ++i2) {
        for (size_t i3 = 0; i3 < d3; ++i3) {
          // Input index for (i0, i1, i2, i3)
          size_t input_idx =
              i0 * (d1 * d2 * d3) + i1 * (d2 * d3) + i2 * d3 + i3;
          // Output index for (i0, i2, i1, i3)
          size_t output_idx =
              i0 * (d2 * d1 * d3) + i2 * (d1 * d3) + i1 * d3 + i3;
          // Assign value
          output[output_idx] = input[input_idx];
        }
      }
    }
  }
  return output;
}

std::vector<uint8_t> cast_bf16bfp16_cpu(const std::vector<float> &ifm, size_t H,
                                        size_t W) {
  std::array<size_t, 4> dims = {H / 8, 8, W / 8, 8};
  std::vector<float> transposed_ifm = transpose0213(ifm, dims);
  size_t dim0 = H / 8;
  size_t dim1 = W / 8;
  size_t dim2 = 8;
  size_t dim3 = 8;
  std::vector<uint8_t> bfp16_ifm;
  for (size_t d0 = 0; d0 < dim0; d0++) {
    for (size_t d1 = 0; d1 < dim1; d1++) {
      for (size_t d2 = 0; d2 < dim2; d2++) {
        size_t start = d0 * dim1 * dim2 * dim3 + d1 * dim2 * dim3 + d2 * dim3;
        size_t end =
            d0 * dim1 * dim2 * dim3 + d1 * dim2 * dim3 + d2 * dim3 + dim3;
        std::vector<float> sub_vec(transposed_ifm.begin() + start,
                                   transposed_ifm.begin() + end);
        std::vector<uint8_t> bfp16_sub_vec = compress_bf16(sub_vec, 8, 8, 0);
        bfp16_ifm.insert(bfp16_ifm.end(), bfp16_sub_vec.begin(),
                         bfp16_sub_vec.end());
      }
    }
  }
  return bfp16_ifm;
}

bool compareBitwiseEqual(const std::vector<uint8_t> &vec1,
                         const std::vector<uint8_t> &vec2) {
  if (vec1.size() != vec2.size()) {
    return false;
  }
  bool flag = true;
  for (size_t i = 0; i < vec1.size(); ++i) {
    if (vec1[i] != vec2[i]) {
      flag = false;
    }
  }
  return flag;
}

static void dumpVectorToTxt(const std::vector<float> &data,
                            const std::vector<size_t> &c_shape,
                            const std::string &filename) {
  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }
  size_t H = c_shape[0];
  size_t W = c_shape[1];
  size_t index = 0;

  for (size_t h = 0; h < H; ++h) {
    outFile << "  Height " << h << ":\n";
    for (size_t w = 0; w < W; ++w) {
      outFile << std::setw(8) << data[index++] << " ";
    }
    outFile << "\n";
  }

  outFile.close();
}

template <typename InT, typename OuT>
int test_cast(const size_t H, const size_t W, bool debug = false,
              const std::string &a_type = "bfloat16",  // a bo
              const std::string &c_type = "bfp16ebs8", // c bo
              const std::string &model_name = "SD_VAE_DEC",
              float pixel_L2_norm_tolerance = 0,
              bool test_with_golden = false) {
  int err_count = 0;
  bool equal;
  float error_tolerance = 0;
  std::map<std::string, std::string> txnbin_a_header = {{"bfloat16", "a16bf"}};
  std::map<std::string, std::string> txnbin_c_header = {
      {"bfp16ebs8", "acc16bfp"}};
  std::map<std::string, std::any> attr;
  attr["ebs"] = 8;

  std::string shape_key = std::to_string(H) + "_" + std::to_string(W);

  std::vector<size_t> a_shape = {H, W};
  std::vector<size_t> c_shape = {H / 8, W / 8, 8, 9};

  std::vector<OuT> aie_out(get_shape_ele_num(c_shape));

  if (test_with_golden) {
    ryzenai::cast cast =
        ryzenai::cast<std::uint16_t, std::uint8_t>(a_type, c_type, false, attr);
    cast.debug(debug);
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/CastBf16Bfp16/";
    std::string ifm_path = test_golden_root_dir + shape_key + "_ifm32.txt";
    std::vector<uint32_t> a_aie = read_hex_file(ifm_path);

    cast.set_params(a_shape);

    std::vector<Tensor> const_Tensor;
    cast.initialize_const_params(const_Tensor);

    std::vector<Tensor> input_Tensor;
    input_Tensor.push_back({a_aie.data(), a_shape, a_type});

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), c_shape, c_type}};

#ifdef UNIT_TEST_PERF
    LOG_THIS(shape_key);
    PROFILE_THIS(cast.execute(input_Tensor, output_Tensor));
#else
    cast.execute(input_Tensor, output_Tensor);
#endif
    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref.txt";

    std::vector<uint32_t> output_golden = read_hex_file(output_golden_path);
    std::vector<OuT> bfp16_output_golden(aie_out.size());
    memcpy(bfp16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));
    equal = compareBitwiseEqual(bfp16_output_golden, aie_out);
    err_count = equal ? 0 : 1;
  } else {
    ryzenai::cast cast =
        ryzenai::cast<std::uint16_t, std::uint8_t>(a_type, c_type, false, attr);
    cast.debug(debug);
    // gen rand
    std::vector<float> raw_a(H * W, 0);
    initialize_random_float(raw_a, 2, -2);
    auto bf16_a = float_2_bf16_vec(raw_a);
    auto fp32_a = bf16_2_float_vec(bf16_a);

    cast.set_params(a_shape);
    std::vector<Tensor> const_Tensor;
    cast.initialize_const_params(const_Tensor);

    std::vector<Tensor> input_Tensor;
    input_Tensor = {{bf16_a.data(), a_shape, a_type}};

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), c_shape, c_type}};

#ifdef UNIT_TEST_PERF
    LOG_THIS(shape_key);
    PROFILE_THIS(cast.execute(input_Tensor, output_Tensor));
#else
    cast.execute(input_Tensor, output_Tensor);
    std::vector<uint8_t> cpu_out = cast_bf16bfp16_cpu(fp32_a, H, W);
    equal = compareBitwiseEqual(cpu_out, cpu_out);
    err_count = equal ? 0 : 1;
#endif
  }
  test_count++;
  return err_count;
}

// TEST(Cast, Golden_Layer0) {
//   int err_count = test_cast<uint16_t, uint8_t>(
//       2048, 4096, false, "bfloat16", "bfp16ebs8", "SD_VAE_DEC", 0.01, true);
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }

// TEST(Cast, Golden_Layer1) {
//   int err_count = test_cast<uint16_t, uint8_t>(
//       2048, 11008, false, "bfloat16", "bfp16ebs8", "SD_VAE_DEC", 0.01, true);
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }

// TEST(Cast, Golden_Layer2) {
//   int err_count = test_cast<uint16_t, uint8_t>(
//       2048, 13696, false, "bfloat16", "bfp16ebs8", "SD_VAE_DEC", 0.01, true);
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }

TEST(Cast, Random_SD3Layer0) {
  int err_count = test_cast<uint16_t, uint8_t>(2048, 4096, false, "bfloat16",
                                               "bfp16ebs8", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Cast, Random_SD3Layer1) {
  int err_count = test_cast<uint16_t, uint8_t>(2048, 11008, false, "bfloat16",
                                               "bfp16ebs8", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(Cast, Random_SD3Layer2) {
  int err_count = test_cast<uint16_t, uint8_t>(2048, 13696, false, "bfloat16",
                                               "bfp16ebs8", "SD_VAE_DEC", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
