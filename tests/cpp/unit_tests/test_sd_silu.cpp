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
#include <ops/sd/silu.hpp>

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
int sd_silu_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
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

static void initialize_random_float(std::vector<float> &vec, float max,
                                    float min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

std::string get_instr_key(std::string prefix,
                          const std::vector<size_t> &shape) {
  std::ostringstream oss;
  oss << prefix;
  for (const auto &dim : shape) {
    oss << "_" << dim;
  }
  return oss.str();
}

template <typename InT = uint16_t, typename OuT = uint16_t>
int test_sd_silu(std::vector<size_t> shape, bool debug = false,
                 const std::string &ifm_type = "bfloat16", // a bo
                 const std::string &out_type = "bfloat16", // c bo
                 const std::string &model_name = "SD_VAE_UNet",
                 float pixel_L2_norm_tolerance = 0.01,
                 bool test_with_golden = false) {
  int err_count = 0;
  float error_tolerance = 0.01f;
  std::map<std::string, std::string> txnbin_a_header = {{"bfloat16", "a16bf"}};
  std::map<std::string, std::string> txnbin_acc_header = {
      {"bfloat16", "acc16bf"}};
  std::vector<size_t> a_shape = shape;
  std::vector<size_t> aie_out_shape = shape;
  size_t a_size = std::accumulate(a_shape.begin(), a_shape.end(), size_t(1),
                                  std::multiplies<size_t>());
  std::vector<OuT> aie_out(a_size);
  int wgt_size = 128;
  std::map<std::string, std::any> attr;
  std::vector<int> a_shape_int(a_shape.begin(), a_shape.end());
  attr["input_shape"] = a_shape_int;
  attr["output_shape"] = a_shape_int;
  std::string xclbin = sd_get_xclbin(model_name);
  std::string pdi_name = xclbin.empty() ? "DPU" : sd_get_pdi(xclbin, "SDSilu");
  std::cerr << "xclbin: " << xclbin << " pdi_name: " << pdi_name << std::endl;
  if (test_with_golden) {

    ryzenai::sd::silu sd_silu = ryzenai::sd::silu<std::uint16_t, std::uint16_t>(
        ifm_type, out_type, false, attr);
    sd_silu.debug(debug);
    sd_silu.set_params(xclbin, pdi_name, a_shape);
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/sd_vae_dec_silu/";

    std::string prefix =
        txnbin_a_header.at(ifm_type) + txnbin_acc_header.at(out_type);
    std::string shape_key = get_instr_key(prefix, a_shape);
    std::string ifm_path = test_golden_root_dir + shape_key + "_ifm32.txt";
    std::vector<uint32_t> a_aie = read_hex_file(ifm_path);

    std::vector<Tensor> const_Tensor;
    std::vector<uint32_t> dummy_wts(32, 0); // 128 bytes 0
    const_Tensor.push_back({dummy_wts.data(), {32}, "uint32"});
    sd_silu.initialize_const_params(const_Tensor);

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
    PROFILE_THIS(sd_silu.execute(input_Tensor, output_Tensor));
#else
    sd_silu.execute(input_Tensor, output_Tensor);
#endif

    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref.txt";

    std::vector<uint32_t> output_golden = read_hex_file(output_golden_path);
    std::vector<OuT> bf16_output_golden(aie_out.size());
    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));
    err_count = sd_silu_check_result<OuT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  } else {
    ryzenai::sd::silu sd_silu = ryzenai::sd::silu<std::uint16_t, std::uint16_t>(
        ifm_type, out_type, false, attr);
    sd_silu.debug(debug);
    sd_silu.set_params(xclbin, pdi_name, a_shape);
    std::vector<float> raw_ifms(a_size, 0);
    initialize_random_float(raw_ifms, 2, -2);
    auto bf16_ifms = float2bf16_vec(raw_ifms);

    std::vector<size_t> in_shape = a_shape;
    std::vector<Tensor> const_Tensor;
    std::vector<uint32_t> dummy_wts(32, 0); // 128 bytes 0
    const_Tensor.push_back({dummy_wts.data(), {32}, "uint32"});
    sd_silu.initialize_const_params(const_Tensor);

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
    PROFILE_THIS(sd_silu.execute(input_Tensor, output_Tensor));
#else
    sd_silu.execute(input_Tensor, output_Tensor);
#endif
    std::vector<OuT> bf16_ofm(a_size);
    // std::vector<size_t> in_shape = a_shape;
    std::vector<int64_t> torch_shape(in_shape.begin(), in_shape.end());
    auto torch_input_tensor =
        torch::from_blob(raw_ifms.data(), torch_shape, torch::kFloat);
    auto ret = torch::silu(torch_input_tensor);

    float *c_golden = ret.data_ptr<float>();
    // convert res from torch into bf16
    uint32_t *c_golden_u = reinterpret_cast<uint32_t *>(c_golden);
    for (size_t i = 0; i < a_size; ++i) {
      bf16_ofm[i] = c_golden_u[i] >> 16;
    }
    err_count = sd_silu_check_result<OuT>(bf16_ofm, aie_out, error_tolerance,
                                          pixel_L2_norm_tolerance);
    std::cout << "out err_count " << err_count << std::endl;
  }
  return err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape1) {
  std::vector<size_t> shape = {2, 64, 64, 320};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape2) {
  std::vector<size_t> shape = {1, 64, 64, 512};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape3) {
  std::vector<size_t> shape = {1, 128, 128, 512};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape4) {
  std::vector<size_t> shape = {1, 256, 256, 256};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape5) {
  std::vector<size_t> shape = {1, 256, 256, 512};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape6) {
  std::vector<size_t> shape = {1, 512, 512, 128};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape7) {
  std::vector<size_t> shape = {1, 512, 512, 256};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape8) {
  std::vector<size_t> shape = {2, 8, 8, 1280};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape9) {
  std::vector<size_t> shape = {2, 8, 8, 2560};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape10) {
  std::vector<size_t> shape = {2, 16, 16, 640};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape11) {
  std::vector<size_t> shape = {2, 16, 16, 1280};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape12) {
  std::vector<size_t> shape = {2, 16, 16, 1920};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape13) {
  std::vector<size_t> shape = {2, 16, 16, 2560};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape14) {
  std::vector<size_t> shape = {2, 32, 32, 320};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape15) {
  std::vector<size_t> shape = {2, 32, 32, 640};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape16) {
  std::vector<size_t> shape = {2, 32, 32, 960};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape17) {
  std::vector<size_t> shape = {2, 32, 32, 1280};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape18) {
  std::vector<size_t> shape = {2, 32, 32, 1920};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape19) {
  std::vector<size_t> shape = {2, 64, 64, 640};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape20) {
  std::vector<size_t> shape = {2, 64, 64, 960};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_KernelShape21) {
  std::vector<size_t> shape = {2, 1280};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// sd3.0
TEST(SD_SILU_Test, Golden_SD3KernelShape1) {
  std::vector<size_t> shape = {1, 512, 512, 512};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_Dec", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_SD3KernelShape2) {
  std::vector<size_t> shape = {1, 1024, 1024, 128};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_Dec", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_SD3KernelShape3) {
  std::vector<size_t> shape = {1, 1024, 1024, 256};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_Dec", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Golden_SD3KernelShape4) {
  std::vector<size_t> shape = {2, 1536};
  int err_count = test_sd_silu<uint16_t, uint16_t>(
      shape, false, "bfloat16", "bfloat16", "SD_UNet", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_2) {
  std::vector<size_t> shape = {2, 64, 64, 320};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_VAE512_1) {
  std::vector<size_t> shape = {1, 64, 64, 512};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_VAE_1) {
  std::vector<size_t> shape = {1, 64, 64, 512};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_VAE");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_VAE1024_3) {
  std::vector<size_t> shape = {1, 128, 128, 512};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_VAE512_2) {
  std::vector<size_t> shape = {1, 128, 128, 512};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_VAE_2) {
  std::vector<size_t> shape = {1, 128, 128, 512};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_VAE");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_VAE512_4) {
  std::vector<size_t> shape = {1, 256, 256, 256};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_VAE_4) {
  std::vector<size_t> shape = {1, 256, 256, 256};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_VAE");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_VAE1024_4) {
  std::vector<size_t> shape = {1, 256, 256, 512};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_VAE512_3) {
  std::vector<size_t> shape = {1, 256, 256, 512};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_VAE_3) {
  std::vector<size_t> shape = {1, 256, 256, 512};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_VAE");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_VAE512_6) {
  std::vector<size_t> shape = {1, 512, 512, 128};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_VAE_6) {
  std::vector<size_t> shape = {1, 512, 512, 128};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_VAE");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_VAE1024_5) {
  std::vector<size_t> shape = {1, 512, 512, 256};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_VAE512_5) {
  std::vector<size_t> shape = {1, 512, 512, 256};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_VAE_5) {
  std::vector<size_t> shape = {1, 512, 512, 256};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_VAE");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_7) {
  std::vector<size_t> shape = {2, 8, 8, 1280};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_8) {
  std::vector<size_t> shape = {2, 8, 8, 2560};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_5) {
  std::vector<size_t> shape = {2, 16, 16, 640};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_6) {
  std::vector<size_t> shape = {2, 16, 16, 1280};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_10) {
  std::vector<size_t> shape = {2, 16, 16, 1920};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_9) {
  std::vector<size_t> shape = {2, 16, 16, 2560};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_3) {
  std::vector<size_t> shape = {2, 32, 32, 320};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_4) {
  std::vector<size_t> shape = {2, 32, 32, 640};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_13) {
  std::vector<size_t> shape = {2, 32, 32, 960};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_12) {
  std::vector<size_t> shape = {2, 32, 32, 1280};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_11) {
  std::vector<size_t> shape = {2, 32, 32, 1920};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_15) {
  std::vector<size_t> shape = {2, 64, 64, 640};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_14) {
  std::vector<size_t> shape = {2, 64, 64, 960};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_UNET_1) {
  std::vector<size_t> shape = {2, 1280};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// sd3.0
TEST(SD_SILU_Test, Random_SD3_VAE1024_6) {
  std::vector<size_t> shape = {1, 512, 512, 512};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_VAE1024_1) {
  std::vector<size_t> shape = {1, 1024, 1024, 128};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_VAE1024_2) {
  std::vector<size_t> shape = {1, 1024, 1024, 256};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_DIT1024_1) {
  std::vector<size_t> shape = {2, 1536};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_DIT1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_DIT512_1) {
  std::vector<size_t> shape = {2, 1536};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_DIT512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_VAEENC512_1) {
  std::vector<size_t> shape = {1, 128, 128, 256};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD3_VAEENC512_2) {
  std::vector<size_t> shape = {1, 256, 256, 128};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD3_VAE512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// https://confluence.amd.com/display/XAEC/SD15+controlnet+preprocess
TEST(SD_SILU_Test, Random_SD15_CONTROLNET_1) {
  std::vector<size_t> shape = {2, 64, 64, 256};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_CONTROLNET_2) {
  std::vector<size_t> shape = {2, 128, 128, 96};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_CONTROLNET_3) {
  std::vector<size_t> shape = {2, 256, 256, 32};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_SILU_Test, Random_SD15_CONTROLNET_4) {
  std::vector<size_t> shape = {2, 512, 512, 16};
  int err_count = test_sd_silu<uint16_t, uint16_t>(shape, false, "bfloat16",
                                                   "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
