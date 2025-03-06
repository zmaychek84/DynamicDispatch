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
#include <ops/flat/rms_add.hpp>

// #pragma STDC FENV_ACCESS ON
static float bfloat16_to_float(uint16_t x) {
  float y = 0.0;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *dst = (uint8_t *)&y;
  dst[2] = src[0];
  dst[3] = src[1];
  return y;
}

static int get_shape_ele_num(const std::vector<int> &shape) {
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
int flat_rms_add_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
                              float error_tolerance = 0.01,
                              float pixel_L2_norm_tolerance = 0.01) {
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

// Helper function to reinterpret float as uint32_t
static uint32_t float_as_uint(float v) {
  union {
    float f;
    uint32_t i;
  } u;
  u.f = v;
  return u.i;
}

// Get the exponent of the floating-point number in IEEE 754 format
static int get_exponent_cpu(float v) {
  uint32_t uint_v = float_as_uint(v);
  return (uint_v & 0x7f800000) >> 23;
}

// Python-like rounding function
static float py3_round(float x) {
  float x_floor = std::floor(x);
  float diff = x - x_floor;

  if (diff > 0.5) {
    return x_floor + 1;
  } else if (diff == 0.5) {
    return (static_cast<int>(x_floor) % 2 == 0) ? x_floor : (x_floor + 1);
  } else {
    return x_floor;
  }
}

static void initialize_random_float(std::vector<float> &vec, float max,
                                    float min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_flat_rms_add(const std::vector<int> &a_shape,
                      const std::vector<int> &b_shape, bool debug = false,
                      const std::string &a_type = "bfloat16", // a bo
                      const std::string &b_type = "bfloat16", // b bo
                      const std::string &c_type = "bfloat16", // c bo
                      const std::string &model_name = "Flat",
                      float pixel_L2_norm_tolerance = 0.01,
                      bool test_with_golden = false) {
  int quantize_err_count = 0;
  int unquantize_err_count = 0;
  float error_tolerance = 0.01;
  std::map<std::string, std::string> txnbin_a_header = {
      {"bfloat16", "a16bf"}, {"bfp16ebs8", "a16bfp"}};
  std::map<std::string, std::string> txnbin_b_header = {
      {"float", "w16bf"}, {"bfloat16", "w16bf"}, {"bfp16ebs8", "w16bfp"}};
  std::map<std::string, std::string> txnbin_acc_header = {
      {"bfloat16", "acc16bf"}};
  std::vector<int> c_shape;
  if (a_shape.size() == b_shape.size()) {
    int a_size = get_shape_ele_num(a_shape);
    int b_size = get_shape_ele_num(b_shape);
    c_shape = a_size > b_size ? a_shape : b_shape;
  } else {
    if (a_shape.size() > b_shape.size()) {
      c_shape = a_shape;
    } else {
      c_shape = b_shape;
    }
  }

  std::vector<OuT> aie_out1(get_shape_ele_num(c_shape));
  std::vector<OuT> aie_out2(get_shape_ele_num(c_shape));
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
  std::string shape_key;
  if (test_with_golden) {
    ryzenai::flat::rms_add flat_rms_add =
        ryzenai::flat::rms_add<std::uint16_t, std::uint16_t, std::uint16_t>(
            a_type, b_type, c_type, false, attr);
    flat_rms_add.debug(debug);
    flat_rms_add.set_params(model_name, a_shape, b_shape);
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/flat_rms_add/";
    shape_key = "";
    for (int i = 0; i < a_shape.size(); i++) {
      shape_key += std::to_string(a_shape[i]) + "_";
    }
    shape_key += "_";
    for (int i = 0; i < b_shape.size(); i++) {
      shape_key += "_" + std::to_string(b_shape[i]);
    }
    std::string ifm_path = test_golden_root_dir + shape_key + "_ifm32.txt";
    std::vector<uint32_t> a_aie = read_hex_file(ifm_path);

    std::string wts_path = test_golden_root_dir + shape_key + "_wts32.txt";
    std::vector<uint32_t> b_aie = read_hex_file(wts_path);
    std::vector<Tensor> const_Tensor;
    std::vector<Tensor> input_Tensor;
    input_Tensor.push_back({a_aie.data(), a_size_t_shape, a_type});

    if (flat_rms_add.is_bias_cal()) {
      const_Tensor.push_back({b_aie.data(), b_size_t_shape, b_type});
    } else {
      input_Tensor.push_back({b_aie.data(), b_size_t_shape, b_type});
    }
    flat_rms_add.initialize_const_params(const_Tensor);

    std::vector<Tensor> output_Tensor;
    output_Tensor.push_back({aie_out1.data(), c_size_t_shape, c_type});
    output_Tensor.push_back({aie_out2.data(), c_size_t_shape, c_type});

#ifdef UNIT_TEST_PERF
    LOG_THIS(shape_key);
    PROFILE_THIS(flat_rms_add.execute(input_Tensor, output_Tensor));
#else
    flat_rms_add.execute(input_Tensor, output_Tensor);
#endif
    std::string output1_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_add1d_ref.txt";
    std::string output2_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_rmsnorm_ref.txt";

    std::vector<uint32_t> output1_golden = read_hex_file(output1_golden_path);
    std::vector<OuT> bf16_output1_golden(aie_out1.size());
    std::vector<uint32_t> output2_golden = read_hex_file(output2_golden_path);
    std::vector<OuT> bf16_output2_golden(aie_out2.size());
    memcpy(bf16_output1_golden.data(), output1_golden.data(),
           output1_golden.size() * sizeof(uint32_t));
    memcpy(bf16_output2_golden.data(), output2_golden.data(),
           output2_golden.size() * sizeof(uint32_t));
    auto quantize_err_1_count = flat_rms_add_check_result<OuT>(
        bf16_output1_golden, aie_out1, error_tolerance,
        pixel_L2_norm_tolerance);

    std::cout << "################ RMS Output ################" << std::endl;
    quantize_err_count = flat_rms_add_check_result<OuT>(
        bf16_output2_golden, aie_out2, error_tolerance,
        pixel_L2_norm_tolerance);
  } else {
    ryzenai::flat::rms_add flat_rms_add =
        ryzenai::flat::rms_add<std::uint16_t, std::uint16_t, std::uint16_t>(
            a_type, b_type, c_type, false, attr);
    flat_rms_add.debug(debug);
    flat_rms_add.set_params(model_name, a_shape, b_shape);
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

    if (flat_rms_add.is_bias_cal()) {
      const_Tensor.push_back({bf16_b.data(), b_size_t_shape, b_type});
    } else {
      input_Tensor.push_back({bf16_b.data(), b_size_t_shape, b_type});
    }
    flat_rms_add.initialize_const_params(const_Tensor);

    std::vector<Tensor> output_Tensor;
    output_Tensor.push_back({aie_out1.data(), c_size_t_shape, c_type});
    output_Tensor.push_back({aie_out2.data(), c_size_t_shape, c_type});

#ifdef UNIT_TEST_PERF
    LOG_THIS(shape_key);
    PROFILE_THIS(flat_rms_add.execute(input_Tensor, output_Tensor));
#else
    flat_rms_add.execute(input_Tensor, output_Tensor);
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
    auto tensor_add = tensor_a + tensor_b;
    std::vector<float> torch_add_buffer(tensor_add.numel());
    std::memcpy(torch_add_buffer.data(), tensor_add.data_ptr<float>(),
                tensor_add.numel() * sizeof(float));
    auto torch_add_bf16 = float_2_bf16_vec(torch_add_buffer);
    std::cout << "################ Add Output ################" << std::endl;
    quantize_err_count = flat_rms_add_check_result<OuT>(
        torch_add_bf16, aie_out1, error_tolerance, pixel_L2_norm_tolerance);

    std::cout << "################ RMS Output ################" << std::endl;
    torch::Tensor rms =
        torch::sqrt(torch::mean(tensor_add.pow(2), -1, true) + 1e-6);
    torch::Tensor torch_rms = tensor_add / rms;
    std::vector<float> torch_rms_buffer(torch_rms.numel());
    std::memcpy(torch_rms_buffer.data(), torch_rms.data_ptr<float>(),
                torch_rms.numel() * sizeof(float));
    auto torch_rms_bf16 = float_2_bf16_vec(torch_rms_buffer);
    quantize_err_count = flat_rms_add_check_result<OuT>(
        torch_rms_bf16, aie_out2, error_tolerance, pixel_L2_norm_tolerance);
  }
  return quantize_err_count;
}

// Golden unittest
TEST(FLAT_RMS_ADD_Test, Golden_RMS_ADD_Shape1) {
  int err_count = test_flat_rms_add<uint16_t, uint16_t, uint16_t>(
      {1, 3072}, {1, 3072}, false, "bfloat16", "bfloat16", "bfloat16",
      "Flat_RMS_ADD", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Random unittest
TEST(FLAT_RMS_ADD_Test, Random_RMS_ADD_Shape1) {
  int err_count = test_flat_rms_add<uint16_t, uint16_t, uint16_t>(
      {1, 3072}, {1, 3072}, false, "bfloat16", "bfloat16", "bfloat16",
      "Flat_RMS_ADD", 0.01);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
