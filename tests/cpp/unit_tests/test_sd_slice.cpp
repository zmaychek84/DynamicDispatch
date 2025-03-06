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
#include <ops/sd/slice.hpp>

// #pragma STDC FENV_ACCESS ON
inline float bfloat16_to_float(uint16_t x) {
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
int sd_slice_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
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

static double round_half_to_even(double value) {
  // Set rounding mode to "round to nearest, ties to even"
  std::fesetround(FE_TONEAREST);

  // Use nearbyint, which rounds according to the current rounding mode
  return std::nearbyint(value);
}

static void aie_srs(std::vector<uint32_t> &input_output) {
  int data_width = 16;
  int shift = 16;
  for (size_t i = 0; i < input_output.size(); ++i) {
    double temp = static_cast<double>(input_output[i]) / std::pow(2.0, shift);
    // temp = std::round(temp);
    temp = round_half_to_even(temp);
    if (temp > std::pow(2.0f, data_width) - 1) {
      temp = float(std::pow(2.0f, data_width)) - 1;
    }
    if (temp < 0) {
      temp = 0;
    }
    input_output[i] = static_cast<uint32_t>(temp);
  }
}

static std::vector<uint16_t> float2bf16_vec(std::vector<float> &x) {
  std::vector<uint32_t> x_uint32(x.size());
  std::memcpy(x_uint32.data(), x.data(), x.size() * sizeof(float));
  aie_srs(x_uint32);
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

static void initialize_random_float(std::vector<float> &vec, int max, int min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

static std::string get_instr_key(std::string prefix,
                                 const std::vector<size_t> &shape) {
  std::ostringstream oss;
  oss << prefix;
  for (const auto &dim : shape) {
    oss << "_" << dim;
  }
  return oss.str();
}

template <typename InT = uint16_t, typename OuT = uint16_t>
int test_sd_slice(std::vector<size_t> in_shape, std::vector<size_t> out_shape,
                  std::vector<int64_t> slice_attr, bool debug = false,
                  const std::string &ifm_type = "bfloat16", // a bo
                  const std::string &out_type = "bfloat16", // c bo
                  const std::string &model_name = "SD_VAE_UNet",
                  float pixel_L2_norm_tolerance = 0.01f,
                  bool test_with_golden = false) {
  int err_count = 0;
  float error_tolerance = 0.01f;
  std::map<std::string, std::string> txnbin_a_header = {{"bfloat16", "a16bf"}};
  std::map<std::string, std::string> txnbin_acc_header = {
      {"bfloat16", "acc16bf"}};
  std::vector<size_t> a_shape = in_shape;
  std::vector<size_t> aie_out_shape = out_shape;
  std::vector<size_t> dimensions;
  dimensions.insert(dimensions.end(), a_shape.begin(), a_shape.end());
  dimensions.insert(dimensions.end(), aie_out_shape.begin(),
                    aie_out_shape.end());
  size_t a_size = std::accumulate(a_shape.begin(), a_shape.end(), size_t(1),
                                  std::multiplies<size_t>());
  size_t c_size = std::accumulate(aie_out_shape.begin(), aie_out_shape.end(),
                                  size_t(1), std::multiplies<size_t>());
  std::vector<OuT> aie_out(c_size);
  int wgt_size = 128;
  std::map<std::string, std::any> attr;
  std::vector<int> a_shape_int(a_shape.begin(), a_shape.end());
  std::vector<int> c_shape_int(aie_out_shape.begin(), aie_out_shape.end());
  attr["input_shape"] = a_shape_int;
  attr["output_shape"] = c_shape_int;
  std::string xclbin = sd_get_xclbin(model_name);
  std::string pdi_name = xclbin.empty() ? "DPU" : sd_get_pdi(xclbin, "SDSlice");
  std::cerr << "xclbin: " << xclbin << " pdi_name: " << pdi_name << std::endl;
  if (test_with_golden) {

    ryzenai::sd::slice sd_slice =
        ryzenai::sd::slice<std::uint16_t, std::uint16_t>(ifm_type, out_type,
                                                         false, attr);
    sd_slice.debug(debug);
    sd_slice.set_params(xclbin, pdi_name);
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/sd3_slice/";

    std::string prefix =
        txnbin_a_header.at(ifm_type) + txnbin_acc_header.at(out_type);
    std::string shape_key = get_instr_key(prefix, dimensions);
    std::string ifm_path = test_golden_root_dir + shape_key + "_ifm32.txt";
    std::vector<uint32_t> a_aie = read_hex_file(ifm_path);

    std::vector<Tensor> const_Tensor;
    std::vector<uint32_t> dummy_wts(32, 0); // 128 bytes 0
    const_Tensor.push_back({dummy_wts.data(), {32}, "uint32"});
    sd_slice.initialize_const_params(const_Tensor);

    std::vector<Tensor> input_Tensor;

    input_Tensor = {{a_aie.data(), a_shape, ifm_type}};

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, out_type}};

#ifdef UNIT_TEST_PERF
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < a_shape.size(); ++i) {
      oss << a_shape[i];
      if (i != a_shape.size() - 1) {
        oss << ", ";
      }
    }
    oss << "]";
    LOG_THIS("input shape = " << oss.str());
    PROFILE_THIS(sd_slice.execute(input_Tensor, output_Tensor));
#else
    sd_slice.execute(input_Tensor, output_Tensor);
#endif

    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref.txt";

    std::vector<uint32_t> output_golden = read_hex_file(output_golden_path);
    std::vector<OuT> bf16_output_golden(aie_out.size());
    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));
    err_count = sd_slice_check_result<OuT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  } else {
    ryzenai::sd::slice sd_slice =
        ryzenai::sd::slice<std::uint16_t, std::uint16_t>(ifm_type, out_type,
                                                         false, attr);
    sd_slice.debug(debug);
    sd_slice.set_params(xclbin, pdi_name);
    std::vector<float> raw_ifms(a_size, 0);
    initialize_random_float(raw_ifms, 2, -2);
    auto bf16_ifms = float2bf16_vec(raw_ifms);
    std::vector<size_t> in_shape = a_shape;
    std::vector<Tensor> const_Tensor;
    std::vector<uint32_t> dummy_wts(32, 0); // 128 bytes 0
    const_Tensor.push_back({dummy_wts.data(), {32}, "uint32"});
    sd_slice.initialize_const_params(const_Tensor);

    std::vector<Tensor> input_Tensor;
    input_Tensor = {{bf16_ifms.data(), a_shape, ifm_type}};
    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, out_type}};

#ifdef UNIT_TEST_PERF
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < a_shape.size(); ++i) {
      oss << a_shape[i];
      if (i != a_shape.size() - 1) {
        oss << ", ";
      }
    }
    oss << "]";
    LOG_THIS("input shape = " << oss.str());
    PROFILE_THIS(sd_slice.execute(input_Tensor, output_Tensor));
#else
    sd_slice.execute(input_Tensor, output_Tensor);
#endif
    std::vector<OuT> bf16_ofm(c_size);
    std::vector<int64_t> torch_shape(in_shape.begin(), in_shape.end());
    auto torch_input_tensor =
        torch::from_blob(raw_ifms.data(), torch_shape, torch::kFloat);
    // Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step)
    auto ret = torch_input_tensor.slice(slice_attr.at(0), slice_attr.at(1),
                                        slice_attr.at(2), slice_attr.at(3));

    float *c_golden = ret.data_ptr<float>();
    // convert res from torch into bf16
    uint32_t *c_golden_u = reinterpret_cast<uint32_t *>(c_golden);
    for (size_t i = 0; i < c_size; ++i) {
      bf16_ofm[i] = c_golden_u[i] >> 16;
    }
    err_count = sd_slice_check_result<OuT>(bf16_ofm, aie_out, error_tolerance,
                                           pixel_L2_norm_tolerance);
    std::cout << "out err_count " << err_count << std::endl;
  }
  return err_count;
}

// golden test
TEST(SD_SLICE_Test, Golden_KernelShape1) {
  std::vector<size_t> in_shape = {2, 1178, 1536};
  std::vector<size_t> out_shape = {2, 154, 1536};
  // not used for golden test
  std::vector<int64_t> slice_attr = {0, 0, 0, 0};
  int err_count = test_sd_slice<uint16_t, uint16_t>(
      in_shape, out_shape, slice_attr, false, "bfloat16", "bfloat16",
      "SD_MMDIT", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SLICE_Test, Golden_KernelShape2) {
  std::vector<size_t> in_shape = {2, 1178, 1536};
  std::vector<size_t> out_shape = {2, 1024, 1536};
  // not used for golden test
  std::vector<int64_t> slice_attr = {0, 0, 0, 0};
  int err_count = test_sd_slice<uint16_t, uint16_t>(
      in_shape, out_shape, slice_attr, false, "bfloat16", "bfloat16",
      "SD_MMDIT", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SLICE_Test, Golden_KernelShape3) {
  std::vector<size_t> in_shape = {2, 4250, 1536};
  std::vector<size_t> out_shape = {2, 4096, 1536};
  // not used for golden test
  std::vector<int64_t> slice_attr = {0, 0, 0, 0};
  int err_count = test_sd_slice<uint16_t, uint16_t>(
      in_shape, out_shape, slice_attr, false, "bfloat16", "bfloat16",
      "SD_MMDIT", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SLICE_Test, Golden_KernelShape4) {
  std::vector<size_t> in_shape = {2, 4250, 1536};
  std::vector<size_t> out_shape = {2, 154, 1536};
  // not used for golden test
  std::vector<int64_t> slice_attr = {0, 0, 0, 0};
  int err_count = test_sd_slice<uint16_t, uint16_t>(
      in_shape, out_shape, slice_attr, false, "bfloat16", "bfloat16",
      "SD_MMDIT", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// random test
//  SD3.0
// 512
TEST(SD_SLICE_Test, Random_SD3_DIT512_1) {
  std::vector<size_t> in_shape = {2, 1178, 1536};
  std::vector<size_t> out_shape = {2, 1024, 1536};
  // dim, start, end,  step
  std::vector<int64_t> slice_attr = {1, 0, 1024, 1};
  int err_count =
      test_sd_slice<uint16_t, uint16_t>(in_shape, out_shape, slice_attr, false,
                                        "bfloat16", "bfloat16", "SD3_DIT512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SLICE_Test, Random_SD3_DIT512_2) {
  std::vector<size_t> in_shape = {2, 1178, 1536};
  std::vector<size_t> out_shape = {2, 154, 1536};
  // dim, start, end,  step
  std::vector<int64_t> slice_attr = {1, 1024, 9223372036854775807, 1};
  int err_count =
      test_sd_slice<uint16_t, uint16_t>(in_shape, out_shape, slice_attr, false,
                                        "bfloat16", "bfloat16", "SD3_DIT512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SLICE_Test, Random_SD3_DIT512_3) {
  std::vector<size_t> in_shape = {2, 1184, 1536};
  std::vector<size_t> out_shape = {2, 160, 1536};
  // dim, start, end,  step
  std::vector<int64_t> slice_attr = {1, 1024, 9223372036854775807, 1};
  int err_count =
      test_sd_slice<uint16_t, uint16_t>(in_shape, out_shape, slice_attr, false,
                                        "bfloat16", "bfloat16", "SD3_DIT512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SLICE_Test, Random_SD3_DIT512_4) {
  std::vector<size_t> in_shape = {2, 1184, 1536};
  std::vector<size_t> out_shape = {2, 1024, 1536};
  // dim, start, end,  step
  std::vector<int64_t> slice_attr = {1, 0, 1024, 1};
  int err_count =
      test_sd_slice<uint16_t, uint16_t>(in_shape, out_shape, slice_attr, false,
                                        "bfloat16", "bfloat16", "SD3_DIT512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// 1024
TEST(SD_SLICE_Test, Random_SD3_DIT1024_1) {
  std::vector<size_t> in_shape = {2, 4250, 1536};
  std::vector<size_t> out_shape = {2, 154, 1536};
  // dim, start, end,  step
  std::vector<int64_t> slice_attr = {1, 4096, 9223372036854775807, 1};
  int err_count =
      test_sd_slice<uint16_t, uint16_t>(in_shape, out_shape, slice_attr, false,
                                        "bfloat16", "bfloat16", "SD3_DIT1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SLICE_Test, Random_SD3_DIT1024_2) {
  std::vector<size_t> in_shape = {2, 4250, 1536};
  std::vector<size_t> out_shape = {2, 4096, 1536};
  // dim, start, end,  step
  std::vector<int64_t> slice_attr = {1, 0, 4096, 1};
  int err_count =
      test_sd_slice<uint16_t, uint16_t>(in_shape, out_shape, slice_attr, false,
                                        "bfloat16", "bfloat16", "SD3_DIT1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SLICE_Test, Random_SD3_DIT1024_3) {
  std::vector<size_t> in_shape = {2, 4256, 1536};
  std::vector<size_t> out_shape = {2, 160, 1536};
  // dim, start, end,  step
  std::vector<int64_t> slice_attr = {1, 4096, 9223372036854775807, 1};
  int err_count =
      test_sd_slice<uint16_t, uint16_t>(in_shape, out_shape, slice_attr, false,
                                        "bfloat16", "bfloat16", "SD3_DIT1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SLICE_Test, Random_SD3_DIT1024_4) {
  std::vector<size_t> in_shape = {2, 4256, 1536};
  std::vector<size_t> out_shape = {2, 4096, 1536};
  // dim, start, end,  step
  std::vector<int64_t> slice_attr = {1, 0, 4096, 1};
  int err_count =
      test_sd_slice<uint16_t, uint16_t>(in_shape, out_shape, slice_attr, false,
                                        "bfloat16", "bfloat16", "SD3_DIT1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
