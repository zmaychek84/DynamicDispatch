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
#include <ops/sd/gemm.hpp>

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
static int sd_gemm_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
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

inline double round_half_to_even(double value) {
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

static void float2bf16_vec(std::vector<float> &x) {
  std::vector<uint32_t> x_uint32(x.size());
  std::memcpy(x_uint32.data(), x.data(), x.size() * sizeof(float));
  aie_srs(x_uint32);
  for (size_t i = 0; i < x_uint32.size(); ++i) {
    x_uint32[i] = (static_cast<uint16_t>(x_uint32[i]) << 16);
  }
  std::memcpy(x.data(), x_uint32.data(), x.size() * sizeof(float));
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
inline int get_exponent_cpu(float v) {
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

// Offline quantization, weight quantization
static std::vector<float> bfp_cpu_kernel_wts(const std::vector<float> &input,
                                             int n, int index, int stride,
                                             int bit_width) {
  int shared_exp = 0;
  std::vector<float> output(input.size(), 0.0f);

  // Loop over block to find shared exponent
  for (int i = index; i < n; i += stride) {
    int exp = get_exponent_cpu(input[i]);
    if (exp == 0xff) {
      exp = 0;
    }
    // Shared exponent is max of exponents
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }

  // Minus 127 to get unbiased value
  int shared_exp_value = shared_exp - 127;
  // 1 sign bit, 8 exp bits
  int m_bits = bit_width - 9;
  float scale = std::pow(2.0f, shared_exp_value - (m_bits - 1));
  float max_v = std::pow(2.0f, shared_exp_value + 1) - scale;

  for (int i = index; i < n; i += stride) {
    // Output +-0/NaN/Inf as is
    int exp = get_exponent_cpu(input[i]);
    if (exp == 0xff) {
      output[i] = input[i];
    } else {
      float x = py3_round(input[i] / scale) * scale;
      output[i] = std::max(-max_v, std::min(x, max_v));
    }
  }

  return output;
}

// Offline quantization, ifm quantization
static std::vector<float> bfp_cpu_kernel_ifm(const std::vector<float> &input,
                                             int n, int index, int stride,
                                             int bit_width) {
  int shared_exp = 0;
  std::vector<float> output(input.size(), 0.0f);

  // Loop over block to find shared exponent
  for (int i = index; i < n; i += stride) {
    int exp = get_exponent_cpu(input[i]);
    if (exp == 0xff) {
      exp = 0;
    }
    // Shared exponent is max of exponents
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }

  // Minus 127 to get unbiased value
  int shared_exp_value = shared_exp - 127;
  // 1 sign bit, 8 exp bits
  int m_bits = bit_width - 9;
  float scale = std::pow(2.0f, shared_exp_value - (m_bits - 1));

  for (int i = index; i < n; i += stride) {
    // Output +-0/NaN/Inf as is
    int exp = get_exponent_cpu(input[i]);
    if (exp == shared_exp) {
      float x = py3_round(input[i] / scale);
      if (x >= 128 || x < -128) {
        shared_exp++;
        shared_exp_value++;
        scale *= 2.0f;
        break;
      }
    }
  }

  float max_v = std::pow(2.0f, shared_exp_value) * (std::pow(2.0f, m_bits) - 1);
  float min_v = -std::pow(2.0f, shared_exp_value) * std::pow(2.0f, m_bits);
  for (int i = index; i < n; i += stride) {
    // Output +-0/NaN/Inf as is
    int exp = get_exponent_cpu(input[i]);
    if (exp == 0xff) {
      output[i] = input[i];
    } else {
      float x = py3_round(input[i] / scale) * scale;
      output[i] = std::max(min_v, std::min(x, max_v));
    }
  }

  return output;
}

static std::vector<float> bfp_cpu_kernel_hw(const std::vector<float> &input,
                                            int n, int index, int stride,
                                            int bit_width) {
  int shared_exp = 0;
  std::vector<float> output(n, 0.0f);

  // First pass to determine the shared exponent
  for (int i = index; i < n; i += stride) {
    int exp = get_exponent_cpu(input[i]);
    if (exp == 0xff) {
      exp = 0;
    }
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }

  int shared_exp_value = shared_exp - 127;
  int m_bits = bit_width - 9;
  float scale = std::pow(2.0f, shared_exp_value - (m_bits - 1));

  // Adjust shared exponent if needed
  for (int i = index; i < n; i += stride) {
    int exp = get_exponent_cpu(input[i]);
    if (exp == shared_exp) {
      float x = py3_round(input[i] / scale);
      if (x >= 128 || x < -128) {
        shared_exp++;
        shared_exp_value++;
        scale *= 2.0f;
        break;
      }
    }
  }

  float max_v = std::pow(2.0f, shared_exp_value) * (std::pow(2.0f, m_bits) - 1);
  float min_v = -std::pow(2.0f, shared_exp_value) * std::pow(2.0f, m_bits);

  // Final pass to quantize values
  for (int i = index; i < n; i += stride) {
    int exp = get_exponent_cpu(input[i]);
    if (exp == 0xff) {
      output[i] = input[i];
    } else {
      float x = py3_round(input[i] / scale) * scale;
      output[i] = std::max(min_v, std::min(x, max_v));
    }
  }
  return output;
}

static void initialize_random_float(std::vector<float> &vec, int max, int min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_sd_gemm(std::vector<size_t> a_shape, std::vector<size_t> b_shape,
                 bool debug = false,
                 const std::string &ifm_type = "bfloat16", // a bo
                 const std::string &wgt_type = "bfloat16", // b bo
                 const std::string &out_type = "bfloat16", // c bo
                 const std::string &model_name = "SD_VAE_DEC",
                 float pixel_L2_norm_tolerance = 0.01,
                 bool test_with_golden = false) {
  int quantize_err_count = 0;
  int unquantize_err_count = 0;
  float error_tolerance = 0.01f;
  int K = b_shape[0];
  int N = b_shape[1];
  int B, M;
  if (a_shape.size() == 2) {
    B = 1;
    M = a_shape[0];
  } else if (a_shape.size() == 3) {
    B = a_shape[0];
    M = a_shape[1];
  } else {
    std::cerr << "Invalid input shape" << std::endl;
    return -1;
  }

  std::map<std::string, std::string> txnbin_a_header = {
      {"bfloat16", "a16bf"}, {"bfp16ebs8", "a16bfp"}};
  std::map<std::string, std::string> txnbin_b_header = {
      {"bfloat16", "w16bf"}, {"bfp16ebs8", "w16bfp"}};
  std::map<std::string, std::string> txnbin_acc_header = {
      {"bfloat16", "acc16bf"}};

  std::vector<size_t> c_shape = a_shape;

  c_shape.back() = b_shape.back();

  std::vector<size_t> aie_out_shape = c_shape;
  size_t c_size = std::accumulate(c_shape.begin(), c_shape.end(), 1ULL,
                                  std::multiplies<>());
  std::vector<OuT> aie_out(c_size);

  size_t wgt_size = std::accumulate(b_shape.begin(), b_shape.end(), 1,
                                    std::multiplies<size_t>());
  size_t ifm_size = std::accumulate(a_shape.begin(), a_shape.end(), 1,
                                    std::multiplies<size_t>());
  std::map<std::string, std::any> attr;

  attr["input_shape"] = std::vector<int>(a_shape.begin(), a_shape.end());
  attr["output_shape"] = std::vector<int>(c_shape.begin(), c_shape.end());
  attr["weight_shape"] = std::vector<int>(b_shape.begin(), b_shape.end());
  attr["bias_enable"] = true;
  std::string xclbin = sd_get_xclbin(model_name);
  std::string pdi_name = xclbin.empty() ? "DPU" : sd_get_pdi(xclbin, "SDGemm");
  std::cerr << "xclbin: " << xclbin << " pdi_name: " << pdi_name << std::endl;
  if (test_with_golden) {
    const std::string bias_type = "bfloat16";
    ryzenai::sd::gemm sd_gemm = ryzenai::sd::gemm<std::uint16_t, std::uint8_t,
                                                  std::uint16_t, std::uint16_t>(
        ifm_type, wgt_type, bias_type, out_type, false, attr);
    sd_gemm.debug(debug);

    sd_gemm.set_params(xclbin, pdi_name);
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/sd_gemm/";
    std::vector<size_t> txn_shape = a_shape;
    txn_shape.push_back(b_shape.back());

    std::string shape_key = txnbin_a_header.at(ifm_type) +
                            txnbin_b_header.at(wgt_type) +
                            txnbin_acc_header.at(out_type) + "_";

    for (size_t i = 0; i < txn_shape.size(); ++i) {
      shape_key += std::to_string(txn_shape[i]);
      if (i != txn_shape.size() - 1) {
        shape_key += "_";
      }
    }

    std::string ifm_path = test_golden_root_dir + shape_key + "_ifm32.txt";
    std::vector<uint32_t> a_aie = read_hex_file(ifm_path);

    std::string wts_path = test_golden_root_dir + shape_key + "_wts32.txt";
    std::vector<uint32_t> b_aie = read_hex_file(wts_path);

    std::vector<size_t> b_shape = {b_aie.size() * sizeof(uint32_t)};
    std::vector<Tensor> const_Tensor;
    // bias is actually not used because it is merged to b_aie.
    const_Tensor = {{b_aie.data(), b_shape, wgt_type}};
    sd_gemm.initialize_const_params(const_Tensor);
    std::vector<Tensor> input_Tensor;

    input_Tensor = {{a_aie.data(), a_shape, ifm_type}};

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, out_type}};

#ifdef UNIT_TEST_PERF
    PROFILE_THIS(sd_gemm.execute(input_Tensor, output_Tensor));
#else
    sd_gemm.execute(input_Tensor, output_Tensor);
#endif

    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref.txt";

    std::vector<uint32_t> output_golden = read_hex_file(output_golden_path);
    std::vector<OuT> bf16_output_golden(aie_out.size());
    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));
    quantize_err_count = sd_gemm_check_result<OuT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  } else {
    ryzenai::sd::gemm sd_gemm =
        ryzenai::sd::gemm<std::uint16_t, float, float, std::uint16_t>(
            ifm_type, "float32", "float32", out_type, false, attr);
    sd_gemm.debug(debug);
    sd_gemm.set_params(xclbin, pdi_name);
    std::vector<float> raw_bias(N, 0);
    initialize_random_float(raw_bias, 2, -2);
    auto bf16_bias = raw_bias;
    float2bf16_vec(bf16_bias);
    std::vector<float> raw_wts(wgt_size, 0);
    initialize_random_float(raw_wts, 2, -2);
    auto bf16_wts = raw_wts;
    float2bf16_vec(bf16_wts);
    std::vector<float> raw_ifms(ifm_size, 0);
    initialize_random_float(raw_ifms, 2, -2);
    auto bf16_ifms = raw_ifms;
    float2bf16_vec(bf16_ifms);
    // aie computation
    // const tensor
    std::vector<Tensor> const_tensors;
    std::vector<size_t> weight_shape = {(size_t)K, (size_t)N};
    const_tensors.push_back({raw_wts.data(), weight_shape, "float"});
    std::vector<size_t> bias_shape = {(size_t)N};
    const_tensors.push_back({raw_bias.data(), bias_shape, "float"});
    sd_gemm.initialize_const_params(const_tensors);
    // input tensor
    std::vector<Tensor> input_Tensor;
    std::vector<uint16_t> aie_ifm_bf16(bf16_ifms.size());
    uint32_t *bf16_ifms_uint32_ptr =
        reinterpret_cast<uint32_t *>(bf16_ifms.data());

    for (size_t i = 0; i < bf16_ifms.size(); ++i) {
      aie_ifm_bf16[i] = bf16_ifms_uint32_ptr[i] >> 16;
    }
    input_Tensor = {{aie_ifm_bf16.data(), a_shape, ifm_type}};

    // output tensor
    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, out_type}};
#ifdef UNIT_TEST_PERF
    PROFILE_THIS(sd_gemm.execute(input_Tensor, output_Tensor));
#else
    sd_gemm.execute(input_Tensor, output_Tensor);
#endif

    // cpu computation
    auto torch_input_tensor =
        torch::from_blob(raw_ifms.data(), {B, M, K}, torch::kFloat);
    auto torch_wts_tensor =
        torch::from_blob(raw_wts.data(), {K, N}, torch::kFloat);
    auto torch_bias_tensor =
        torch::from_blob(raw_bias.data(), {1, N}, torch::kFloat);
    auto res = torch::matmul(torch_input_tensor, torch_wts_tensor);
    res = res + torch_bias_tensor;
    float *c_golden = res.data_ptr<float>();
    uint32_t *c_golden_u = reinterpret_cast<uint32_t *>(c_golden);
    std::vector<OuT> bf16_output_golden(aie_out.size());
    // convert golden to bf16
    for (size_t i = 0; i < bf16_output_golden.size(); ++i) {
      bf16_output_golden[i] = c_golden_u[i] >> 16;
    }

    // calculate l2norm between golden and aie output
    quantize_err_count = sd_gemm_check_result<OuT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  }
  return quantize_err_count;
}

// Golden unittest start
TEST(SD_GEMM_Test, Golden_SD3_Kernel1) {
  std::vector<size_t> a_shape = {2, 1, 1536};
  std::vector<size_t> b_shape = {1536, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel2) {
  std::vector<size_t> a_shape = {2, 1, 2048};
  std::vector<size_t> b_shape = {2048, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel3) {
  std::vector<size_t> a_shape = {2, 1, 256};
  std::vector<size_t> b_shape = {256, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel4) {
  std::vector<size_t> a_shape = {2, 154, 1536};
  std::vector<size_t> b_shape = {1536, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel5) {
  std::vector<size_t> a_shape = {2, 154, 1536};
  std::vector<size_t> b_shape = {1536, 6144};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel6) {
  std::vector<size_t> a_shape = {2, 154, 4096};
  std::vector<size_t> b_shape = {4096, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel7) {
  std::vector<size_t> a_shape = {2, 154, 6144};
  std::vector<size_t> b_shape = {6144, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel8) {
  std::vector<size_t> a_shape = {2, 1024, 1536};
  std::vector<size_t> b_shape = {1536, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel9) {
  std::vector<size_t> a_shape = {2, 1024, 1536};
  std::vector<size_t> b_shape = {1536, 6144};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel10) {
  std::vector<size_t> a_shape = {2, 1024, 1536};
  std::vector<size_t> b_shape = {1536, 64};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel11) {
  std::vector<size_t> a_shape = {2, 1024, 6144};
  std::vector<size_t> b_shape = {6144, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel12) {
  std::vector<size_t> a_shape = {2, 4096, 1536};
  std::vector<size_t> b_shape = {1536, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel13) {
  std::vector<size_t> a_shape = {2, 4096, 1536};
  std::vector<size_t> b_shape = {1536, 6144};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel14) {
  std::vector<size_t> a_shape = {2, 4096, 1536};
  std::vector<size_t> b_shape = {1536, 64};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel15) {
  std::vector<size_t> a_shape = {2, 4096, 6144};
  std::vector<size_t> b_shape = {6144, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD3_Kernel16) {
  std::vector<size_t> a_shape = {1, 16384, 512};
  std::vector<size_t> b_shape = {512, 512};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel1) {
  std::vector<size_t> a_shape = {2, 1, 1280};
  std::vector<size_t> b_shape = {1280, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel2) {
  std::vector<size_t> a_shape = {2, 1, 1280};
  std::vector<size_t> b_shape = {1280, 320};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel3) {
  std::vector<size_t> a_shape = {2, 1, 1280};
  std::vector<size_t> b_shape = {1280, 640};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel4) {
  std::vector<size_t> a_shape = {2, 1, 320};
  std::vector<size_t> b_shape = {320, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel5) {
  std::vector<size_t> a_shape = {2, 64, 1280};
  std::vector<size_t> b_shape = {1280, 10240};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel6) {
  std::vector<size_t> a_shape = {2, 64, 1280};
  std::vector<size_t> b_shape = {1280, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel7) {
  std::vector<size_t> a_shape = {2, 64, 1280};
  std::vector<size_t> b_shape = {1280, 5120};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel8) {
  std::vector<size_t> a_shape = {2, 64, 5120};
  std::vector<size_t> b_shape = {5120, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel9) {
  std::vector<size_t> a_shape = {2, 77, 768};
  std::vector<size_t> b_shape = {768, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel10) {
  std::vector<size_t> a_shape = {2, 77, 768};
  std::vector<size_t> b_shape = {768, 320};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel11) {
  std::vector<size_t> a_shape = {2, 77, 768};
  std::vector<size_t> b_shape = {768, 640};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel12) {
  std::vector<size_t> a_shape = {2, 256, 1280};
  std::vector<size_t> b_shape = {1280, 10240};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel13) {
  std::vector<size_t> a_shape = {2, 256, 1280};
  std::vector<size_t> b_shape = {1280, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel14) {
  std::vector<size_t> a_shape = {2, 256, 1280};
  std::vector<size_t> b_shape = {1280, 5120};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel15) {
  std::vector<size_t> a_shape = {2, 256, 5120};
  std::vector<size_t> b_shape = {5120, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel16) {
  std::vector<size_t> a_shape = {2, 1024, 2560};
  std::vector<size_t> b_shape = {2560, 640};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel17) {
  std::vector<size_t> a_shape = {2, 1024, 640};
  std::vector<size_t> b_shape = {640, 2560};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel18) {
  std::vector<size_t> a_shape = {2, 1024, 640};
  std::vector<size_t> b_shape = {640, 5120};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel19) {
  std::vector<size_t> a_shape = {2, 1024, 640};
  std::vector<size_t> b_shape = {640, 640};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel20) {
  std::vector<size_t> a_shape = {2, 4096, 1280};
  std::vector<size_t> b_shape = {1280, 320};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel21) {
  std::vector<size_t> a_shape = {2, 4096, 320};
  std::vector<size_t> b_shape = {320, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel22) {
  std::vector<size_t> a_shape = {2, 4096, 320};
  std::vector<size_t> b_shape = {320, 2560};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel23) {
  std::vector<size_t> a_shape = {2, 4096, 320};
  std::vector<size_t> b_shape = {320, 320};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Golden_SD1p5_Kernel24) {
  std::vector<size_t> a_shape = {1, 4096, 512};
  std::vector<size_t> b_shape = {512, 512};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD_VAE_DEC", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Random unittest start
TEST(SD_GEMM_Test, Random_SD3_DIT1024_1) {
  std::vector<size_t> a_shape = {2, 1, 1536};
  std::vector<size_t> b_shape = {1536, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT1024", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_3) {
  std::vector<size_t> a_shape = {2, 1, 1536};
  std::vector<size_t> b_shape = {1536, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT1024_2) {
  std::vector<size_t> a_shape = {2, 1, 2048};
  std::vector<size_t> b_shape = {2048, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT1024", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_2) {
  std::vector<size_t> a_shape = {2, 1, 2048};
  std::vector<size_t> b_shape = {2048, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT1024_3) {
  std::vector<size_t> a_shape = {2, 1, 256};
  std::vector<size_t> b_shape = {256, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT1024", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_4) {
  std::vector<size_t> a_shape = {2, 1, 256};
  std::vector<size_t> b_shape = {256, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT1024_4) {
  std::vector<size_t> a_shape = {2, 154, 1536};
  std::vector<size_t> b_shape = {1536, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT1024", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_5) {
  std::vector<size_t> a_shape = {2, 154, 1536};
  std::vector<size_t> b_shape = {1536, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT1024_5) {
  std::vector<size_t> a_shape = {2, 154, 1536};
  std::vector<size_t> b_shape = {1536, 6144};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT1024", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_8) {
  std::vector<size_t> a_shape = {2, 154, 1536};
  std::vector<size_t> b_shape = {1536, 6144};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT1024_6) {
  std::vector<size_t> a_shape = {2, 154, 4096};
  std::vector<size_t> b_shape = {4096, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT1024", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_1) {
  std::vector<size_t> a_shape = {2, 154, 4096};
  std::vector<size_t> b_shape = {4096, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT1024_7) {
  std::vector<size_t> a_shape = {2, 154, 6144};
  std::vector<size_t> b_shape = {6144, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT1024", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_10) {
  std::vector<size_t> a_shape = {2, 154, 6144};
  std::vector<size_t> b_shape = {6144, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_6) {
  std::vector<size_t> a_shape = {2, 1024, 1536};
  std::vector<size_t> b_shape = {1536, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_7) {
  std::vector<size_t> a_shape = {2, 1024, 1536};
  std::vector<size_t> b_shape = {1536, 6144};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_11) {
  std::vector<size_t> a_shape = {2, 1024, 1536};
  std::vector<size_t> b_shape = {1536, 64};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_9) {
  std::vector<size_t> a_shape = {2, 1024, 6144};
  std::vector<size_t> b_shape = {6144, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT1024_8) {
  std::vector<size_t> a_shape = {2, 4096, 1536};
  std::vector<size_t> b_shape = {1536, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT1024", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT1024_9) {
  std::vector<size_t> a_shape = {2, 4096, 1536};
  std::vector<size_t> b_shape = {1536, 6144};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT1024", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT1024_10) {
  std::vector<size_t> a_shape = {2, 4096, 1536};
  std::vector<size_t> b_shape = {1536, 64};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT1024", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT1024_11) {
  std::vector<size_t> a_shape = {2, 4096, 6144};
  std::vector<size_t> b_shape = {6144, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT1024", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_VAE1024_1) {
  std::vector<size_t> a_shape = {1, 16384, 512};
  std::vector<size_t> b_shape = {512, 512};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_VAE1024", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_5) {
  std::vector<size_t> a_shape = {2, 1, 1280};
  std::vector<size_t> b_shape = {1280, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_6) {
  std::vector<size_t> a_shape = {2, 1, 1280};
  std::vector<size_t> b_shape = {1280, 320};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_7) {
  std::vector<size_t> a_shape = {2, 1, 1280};
  std::vector<size_t> b_shape = {1280, 640};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_4) {
  std::vector<size_t> a_shape = {2, 1, 320};
  std::vector<size_t> b_shape = {320, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_23) {
  std::vector<size_t> a_shape = {2, 64, 1280};
  std::vector<size_t> b_shape = {1280, 10240};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_17) {
  std::vector<size_t> a_shape = {2, 64, 1280};
  std::vector<size_t> b_shape = {1280, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_18) {
  std::vector<size_t> a_shape = {2, 64, 1280};
  std::vector<size_t> b_shape = {1280, 5120};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_19) {
  std::vector<size_t> a_shape = {2, 64, 5120};
  std::vector<size_t> b_shape = {5120, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_2) {
  std::vector<size_t> a_shape = {2, 77, 768};
  std::vector<size_t> b_shape = {768, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_3) {
  std::vector<size_t> a_shape = {2, 77, 768};
  std::vector<size_t> b_shape = {768, 320};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_1) {
  std::vector<size_t> a_shape = {2, 77, 768};
  std::vector<size_t> b_shape = {768, 640};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_21) {
  std::vector<size_t> a_shape = {2, 256, 1280};
  std::vector<size_t> b_shape = {1280, 10240};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_14) {
  std::vector<size_t> a_shape = {2, 256, 1280};
  std::vector<size_t> b_shape = {1280, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_15) {
  std::vector<size_t> a_shape = {2, 256, 1280};
  std::vector<size_t> b_shape = {1280, 5120};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_16) {
  std::vector<size_t> a_shape = {2, 256, 5120};
  std::vector<size_t> b_shape = {5120, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_13) {
  std::vector<size_t> a_shape = {2, 1024, 2560};
  std::vector<size_t> b_shape = {2560, 640};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_12) {
  std::vector<size_t> a_shape = {2, 1024, 640};
  std::vector<size_t> b_shape = {640, 2560};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_20) {
  std::vector<size_t> a_shape = {2, 1024, 640};
  std::vector<size_t> b_shape = {640, 5120};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_11) {
  std::vector<size_t> a_shape = {2, 1024, 640};
  std::vector<size_t> b_shape = {640, 640};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_10) {
  std::vector<size_t> a_shape = {2, 4096, 1280};
  std::vector<size_t> b_shape = {1280, 320};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_9) {
  std::vector<size_t> a_shape = {2, 4096, 320};
  std::vector<size_t> b_shape = {320, 1280};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_22) {
  std::vector<size_t> a_shape = {2, 4096, 320};
  std::vector<size_t> b_shape = {320, 2560};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_UNET_8) {
  std::vector<size_t> a_shape = {2, 4096, 320};
  std::vector<size_t> b_shape = {320, 320};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_UNET",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_VAE512_1) {
  std::vector<size_t> a_shape = {1, 4096, 512};
  std::vector<size_t> b_shape = {512, 512};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_VAE512", 0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD15_VAE_1) {
  std::vector<size_t> a_shape = {1, 4096, 512};
  std::vector<size_t> b_shape = {512, 512};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16", "SD15_VAE",
      0.02f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_160_1) {
  std::vector<size_t> a_shape = {2, 160, 1536};
  std::vector<size_t> b_shape = {1536, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_160_2) {
  std::vector<size_t> a_shape = {2, 160, 1536};
  std::vector<size_t> b_shape = {1536, 6144};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_160_3) {
  std::vector<size_t> a_shape = {2, 160, 4096};
  std::vector<size_t> b_shape = {4096, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GEMM_Test, Random_SD3_DIT512_160_4) {
  std::vector<size_t> a_shape = {2, 160, 6144};
  std::vector<size_t> b_shape = {6144, 1536};
  int err_count = test_sd_gemm<uint16_t, uint8_t, uint16_t>(
      a_shape, b_shape, false, "bfloat16", "bfp16ebs8", "bfloat16",
      "SD3_DIT512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
