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
#include <ops/sd/concat.hpp>

// #pragma STDC FENV_ACCESS ON
inline float bfloat16_to_float(uint16_t x) {
  float y = 0.0;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *dst = (uint8_t *)&y;
  dst[2] = src[0];
  dst[3] = src[1];
  return y;
}

inline int get_shape_ele_num(const std::vector<int> &shape) {
  int total_num = 1;
  for (int dim : shape) {
    total_num *= dim;
  }
  return total_num;
}

static std::vector<uint32_t> read_hex_file(const std::string &filePath) {
  std::ifstream fileStream(filePath);

  if (!fileStream.is_open()) {
    std::cerr << "Failed to open file " << filePath << "!" << std::endl;
    throw std::runtime_error("Failed to open file " + filePath + "!");
    return {};
  }

  std::cout << "Opened file " << filePath << " for reading hex data!"
            << std::endl;

  std::vector<uint32_t> buffer;
  uint32_t temp;

  while (fileStream >> std::hex >> temp) {
    buffer.push_back(temp);
  }

  fileStream.close();
  return buffer;
}

template <typename T>
int sd_concat_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
                           float error_tolerance = 0.01f,
                           float pixel_L2_norm_tolerance = 0.01f) {
  int fail = 0;
  float max_diff = 0;
  float L2_norm = 0;
  int err_count = 0;
  for (int i = 0; i < cpu_Y.size(); ++i) {
    float diff = std::abs(bfloat16_to_float(cpu_Y.at(i)) -
                          bfloat16_to_float(aie_Y.at(i)));
    L2_norm += ((float)diff * (float)diff);
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > error_tolerance) {
      // if (err_count < 100) {
      //   std::cout << "ERROR: Y[" << i << "]: "
      //             << "Expected: " << bfloat16_to_float(cpu_Y.at(i)) << ","
      //             << "Received: " << bfloat16_to_float(aie_Y.at(i))
      //             << "\n";
      // }
      fail = 1;
      err_count++;
    }
  }
  L2_norm = std::sqrt(L2_norm);
  auto pixel_L2_norm = L2_norm / cpu_Y.size();
  std::cout << "max_diff is " << max_diff << std::endl;
  std::cout << "L2_norm is " << L2_norm << std::endl;
  std::cout << "pixel L2_norm is " << pixel_L2_norm << std::endl;
  std::cout << "pixel_L2_norm_tolerance is " << pixel_L2_norm_tolerance
            << std::endl;
  if (err_count > 0 && pixel_L2_norm < pixel_L2_norm_tolerance) {
    std::cout << "deem err_count as zero due to low pixel_L2_norm" << std::endl;
    err_count = 0;
  }
  std::cout << "err_count is " << err_count << std::endl;
  return err_count;
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

inline void initialize_random_float(std::vector<float> &vec, float max,
                                    float min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

std::string vectorToString(const std::vector<int> &vec) {
  std::string result;
  for (size_t i = 0; i < vec.size(); ++i) {
    result += std::to_string(vec[i]);
    if (i != vec.size() - 1) {
      result += ", "; // 用逗号分隔
    }
  }
  return result;
}

template <typename InT, typename OuT>
int test_sd_concat(const std::vector<int> &a_shape,
                   const std::vector<int> &b_shape, bool debug = false,
                   const std::string &a_type = "bfloat16", // a bo
                   const std::string &c_type = "bfloat16", // c bo
                   const std::string &model_name = "SD_VAE_DEC",
                   float pixel_L2_norm_tolerance = 0.01f,
                   bool test_with_golden = false, int axis = -1) {
  int quantize_err_count = 0;
  int unquantize_err_count = 0;
  float error_tolerance = 0.01f;
  std::map<std::string, std::string> txnbin_a_header = {
      {"bfloat16", "a16bf"}, {"bfp16ebs8", "a16bfp"}};
  std::map<std::string, std::string> txnbin_b_header = {
      {"bfloat16", "w16bf"}, {"bfp16ebs8", "w16bfp"}};
  std::map<std::string, std::string> txnbin_acc_header = {
      {"bfloat16", "acc16bf"}};
  std::string b_type = a_type;
  std::vector<int> c_shape;
  c_shape = a_shape;
  if (axis >= 0) {
    c_shape[axis] = a_shape[axis] + b_shape[axis];
  } else {
    c_shape[c_shape.size() + axis] =
        a_shape[a_shape.size() + axis] + b_shape[b_shape.size() + axis];
  }
  std::cout << "a_shape " << vectorToString(a_shape) << std::endl;
  std::cout << "b_shape " << vectorToString(b_shape) << std::endl;
  std::cout << "c_shape " << vectorToString(c_shape) << std::endl;
  std::vector<OuT> aie_out(get_shape_ele_num(c_shape));
  std::map<std::string, std::any> attr;
  attr["a_shape"] = a_shape;
  attr["b_shape"] = b_shape;
  attr["c_shape"] = c_shape;

  std::vector<size_t> a_size_t_shape;
  std::transform(a_shape.begin(), a_shape.end(),
                 std::back_inserter(a_size_t_shape),
                 [](int val) { return static_cast<size_t>(val); });
  std::vector<size_t> b_size_t_shape;
  std::transform(b_shape.begin(), b_shape.end(),
                 std::back_inserter(b_size_t_shape),
                 [](int val) { return static_cast<size_t>(val); });
  std::vector<size_t> c_size_t_shape;
  std::transform(c_shape.begin(), c_shape.end(),
                 std::back_inserter(c_size_t_shape),
                 [](int val) { return static_cast<size_t>(val); });
  std::string shape_key = txnbin_a_header.at(a_type) +
                          txnbin_b_header.at(a_type) +
                          txnbin_acc_header.at(c_type);
  for (int i = 0; i < a_shape.size(); i++) {
    shape_key += "_" + std::to_string(a_shape[i]);
  }
  shape_key += "__";
  for (int i = 0; i < b_shape.size(); i++) {
    shape_key += "_" + std::to_string(b_shape[i]);
  }
  std::string xclbin = sd_get_xclbin(model_name);
  std::string pdi_name =
      xclbin.empty() ? "DPU" : sd_get_pdi(xclbin, "SDConcat");
  std::cerr << "xclbin: " << xclbin << " pdi_name: " << pdi_name << std::endl;
  if (test_with_golden) {
    ryzenai::sd::concat sd_concat =
        ryzenai::sd::concat<std::uint16_t, std::uint16_t>(a_type, c_type, false,
                                                          attr);
    sd_concat.debug(debug);
    sd_concat.set_params(xclbin, pdi_name, a_shape, b_shape, axis);
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/sd_vae_dec_concat/";
    std::string ifm_path = test_golden_root_dir + shape_key + "_ifm32.txt";
    std::vector<uint32_t> a_aie = read_hex_file(ifm_path);

    std::string wts_path = test_golden_root_dir + shape_key + "_wts32.txt";
    std::vector<uint32_t> b_aie = read_hex_file(wts_path);
    std::vector<Tensor> const_Tensor;
    std::vector<Tensor> input_Tensor;
    input_Tensor.push_back({a_aie.data(), a_size_t_shape, a_type});
    input_Tensor.push_back({b_aie.data(), b_size_t_shape, b_type});
    sd_concat.initialize_const_params(const_Tensor);

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), c_size_t_shape, c_type}};

#ifdef UNIT_TEST_PERF
    LOG_THIS(shape_key);
    PROFILE_THIS(sd_concat.execute(input_Tensor, output_Tensor));
#else
    sd_concat.execute(input_Tensor, output_Tensor);
#endif

    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref.txt";

    std::vector<uint32_t> output_golden = read_hex_file(output_golden_path);
    std::vector<OuT> bf16_output_golden(aie_out.size());

    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));
    quantize_err_count = sd_concat_check_result<OuT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  } else {
    ryzenai::sd::concat sd_concat =
        ryzenai::sd::concat<std::uint16_t, std::uint16_t>(a_type, c_type, false,
                                                          attr);
    sd_concat.debug(debug);
    sd_concat.set_params(xclbin, pdi_name, a_shape, b_shape, axis);
    // gen rand
    std::vector<float> raw_a(get_shape_ele_num(a_shape), 0);
    initialize_random_float(raw_a, 2, -2);
    auto bf16_a = float_2_bf16_vec(raw_a);
    std::vector<float> raw_b(get_shape_ele_num(b_shape), 0);
    initialize_random_float(raw_b, 2, -2);
    auto bf16_b = float_2_bf16_vec(raw_b);

    std::vector<Tensor> const_Tensor;
    std::vector<Tensor> input_Tensor;

    input_Tensor.push_back({bf16_a.data(), a_size_t_shape, a_type});
    input_Tensor.push_back({bf16_b.data(), b_size_t_shape, b_type});
    sd_concat.initialize_const_params(const_Tensor);

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), c_size_t_shape, c_type}};
#ifdef UNIT_TEST_PERF
    LOG_THIS(shape_key);
    PROFILE_THIS(sd_concat.execute(input_Tensor, output_Tensor));
#else
    sd_concat.execute(input_Tensor, output_Tensor);
#endif
    std::vector<int64_t> a_int64_shape;
    for (int val : a_shape) {
      a_int64_shape.push_back(static_cast<int64_t>(val));
    }

    std::vector<int64_t> b_int64_shape;
    for (int val : b_shape) {
      b_int64_shape.push_back(static_cast<int64_t>(val));
    }
    auto tensor_a =
        torch::from_blob(raw_a.data(), a_int64_shape, torch::kFloat);
    auto tensor_b =
        torch::from_blob(raw_b.data(), b_int64_shape, torch::kFloat);
    auto tensor_c = torch::cat({tensor_a, tensor_b}, axis);

    std::vector<float> torch_c_buffer(tensor_c.numel());
    std::memcpy(torch_c_buffer.data(), tensor_c.data_ptr<float>(),
                tensor_c.numel() * sizeof(float));
    std::vector<OuT> torch_c_bf16 = float_2_bf16_vec(torch_c_buffer);
    quantize_err_count = sd_concat_check_result<OuT>(
        torch_c_bf16, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  }
  return quantize_err_count;
}

// Golden unittest
TEST(SD_CONCAT_Test, Golden_ConcatLayer6) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 64, 64, 320}, {2, 64, 64, 320}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Golden_ConcatLayer7) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 64, 64, 640}, {2, 64, 64, 320}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Golden_ConcatLayer8) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 8, 8, 1280}, {2, 8, 8, 1280}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Golden_ConcatLayer9) {
  int err_count =
      test_sd_concat<uint16_t, uint16_t>({2, 160}, {2, 160}, false, "bfloat16",
                                         "bfloat16", "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Golden_ConcatLayer1) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 16, 16, 1280}, {2, 16, 16, 1280}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Golden_ConcatLayer2) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 16, 16, 1280}, {2, 16, 16, 640}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Golden_ConcatLayer3) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 32, 32, 1280}, {2, 32, 32, 640}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Golden_ConcatLayer4) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 32, 32, 640}, {2, 32, 32, 320}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Golden_ConcatLayer5) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 32, 32, 640}, {2, 32, 32, 640}, false, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SD3.0
TEST(SD_CONCAT_Test, Golden_SD3ConcatLayer1) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 1024, 1536}, {2, 154, 1536}, true, "bfloat16", "bfloat16",
      "SD_VAE_DEC", 0.01f, true, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Golden_SD3_DIT1024_Layer1) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 4096, 1536}, {2, 154, 1536}, true, "bfloat16", "bfloat16",
      "SD3_DIT1024", 0.01f, true, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Random unittest

// SD3.0
// 512
TEST(SD_CONCAT_Test, Random_SD3_DIT512_1) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 1024, 1536}, {2, 154, 1536}, false, "bfloat16", "bfloat16",
      "SD3_DIT512", 0.01f, false, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Random_SD3_DIT512_2) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 1024, 1536}, {2, 160, 1536}, false, "bfloat16", "bfloat16",
      "SD3_DIT512", 0.01f, false, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// 1024
TEST(SD_CONCAT_Test, Random_SD3_DIT1024_Layer1) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 4096, 1536}, {2, 154, 1536}, false, "bfloat16", "bfloat16",
      "SD3_DIT1024", 0.01f, false, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Random_SD3_DIT1024_Layer2) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 4096, 1536}, {2, 160, 1536}, false, "bfloat16", "bfloat16",
      "SD3_DIT1024", 0.01f, false, 1);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Random_SD15_UNET_2) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 16, 16, 1280}, {2, 16, 16, 1280}, false, "bfloat16", "bfloat16",
      "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Random_SD15_UNET_3) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 16, 16, 1280}, {2, 16, 16, 640}, false, "bfloat16", "bfloat16",
      "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Random_SD15_UNET_4) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 32, 32, 1280}, {2, 32, 32, 640}, false, "bfloat16", "bfloat16",
      "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Random_SD15_UNET_6) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 32, 32, 640}, {2, 32, 32, 320}, false, "bfloat16", "bfloat16",
      "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Random_SD15_UNET_5) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 32, 32, 640}, {2, 32, 32, 640}, false, "bfloat16", "bfloat16",
      "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Random_SD15_UNET_8) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 64, 64, 320}, {2, 64, 64, 320}, false, "bfloat16", "bfloat16",
      "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Random_SD15_UNET_7) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 64, 64, 640}, {2, 64, 64, 320}, false, "bfloat16", "bfloat16",
      "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Random_SD15_UNET_1) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 8, 8, 1280}, {2, 8, 8, 1280}, false, "bfloat16", "bfloat16",
      "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_CONCAT_Test, Random_SD15_UNET_9) {
  int err_count = test_sd_concat<uint16_t, uint16_t>(
      {2, 160}, {2, 160}, false, "bfloat16", "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
