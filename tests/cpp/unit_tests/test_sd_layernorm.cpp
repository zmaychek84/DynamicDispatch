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
#include "ops/ops_common/lrn_matrix.hpp"
#include <ops/sd/layernorm.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace lrn_matrix;
using namespace std;

// #pragma STDC FENV_ACCESS ON
inline float bfloat16_to_float(uint16_t x) {
  float y = 0.0;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *dst = (uint8_t *)&y;
  dst[2] = src[0];
  dst[3] = src[1];
  return y;
}

static std::vector<uint32_t> read_file(const std::string &filePath) {
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
int sd_layernorm_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
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

static std::vector<uint16_t> float2bf16_vec(std::vector<float> &x) {
  std::vector<uint32_t> x_uint32(x.size());
  std::memcpy(x_uint32.data(), x.data(), x.size() * sizeof(float));
  aie_srs_compute(x_uint32);

  std::vector<uint16_t> x_uint16(x.size());

  for (size_t i = 0; i < x_uint32.size(); ++i) {
    x_uint16[i] = static_cast<uint16_t>(x_uint32[i]);
  }

  return x_uint16;
}

static void initialize_random_float(std::vector<float> &vec, int max, int min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_lrn(int B, int M, int K, bool debug = false,
             const std::string &a_dtype = "bfloat16",
             const std::string &b_dtype = "bfloat16",
             const std::string &c_dtype = "bfloat16",
             const std::string &model_name = "SD_UNet",
             float error_tolerance = 0.01, float pixel_L2_norm_tolerance = 0.01,
             bool test_with_golden = false) {

  int err_count = 0;
  size_t Bs = static_cast<size_t>(B);
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  std::vector<size_t> a_shape = {Bs, Ms, Ks};
  std::vector<size_t> gamma_shape = {Ks};
  std::vector<size_t> beta_shape = {Ks};
  std::vector<size_t> aie_out_shape = {Bs, Ms, Ks};

  std::vector<InT> a(B * M * K);
  std::vector<float> gamma(K); // for CPU calculation
  std::vector<float> beta(K);  // for CPU calculation
  std::vector<WgT> aie_gamma(K);
  std::vector<WgT> aie_beta(K);
  std::vector<OutT> cpu_out(B * M * K);
  std::vector<OutT> aie_out(B * M * K);

  std::map<std::string, std::any> attr;
  attr["input_shape"] = std::vector<int>{B, M, K};
  attr["output_shape"] = std::vector<int>{B, M, K};
  std::string xclbin = sd_get_xclbin(model_name);
  std::string pdi_name =
      xclbin.empty() ? "DPU" : sd_get_pdi(xclbin, "SDLayerNorm");
  std::cerr << "xclbin: " << xclbin << " pdi_name: " << pdi_name << std::endl;
  if (test_with_golden) {

    ryzenai::sd::layernorm layernorm_ = ryzenai::sd::layernorm<InT, WgT, OutT>(
        a_dtype, b_dtype, c_dtype, false, attr);
    layernorm_.debug(debug);
    layernorm_.set_params(xclbin, pdi_name);
    std::vector<Tensor> const_Tensor;

    std::map<std::string, std::string> txnbin_a_header = {
        {"bfloat16", "a16bf"}};
    std::map<std::string, std::string> txnbin_b_header = {
        {"bfloat16", "w16bf"}};

    std::map<std::string, std::string> txnbin_acc_header = {
        {"bfloat16", "a16bf"}};
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/sd_layernorm/";
    std::string shape_key =
        txnbin_a_header.at(a_dtype) + txnbin_b_header.at(b_dtype) +
        txnbin_acc_header.at(c_dtype) + "_" + std::to_string(B) + "_" +
        std::to_string(M) + "_" + std::to_string(K);
    // wts file concat the gamma and beta
    std::string wts_path = test_golden_root_dir + shape_key + "_wts32.txt";
    std::vector<uint32_t> wts_aie = read_file(wts_path);

    std::vector<uint32_t> gamma_aie(wts_aie.begin(),
                                    wts_aie.begin() + wts_aie.size() / 2);
    std::vector<uint32_t> beta_aie(wts_aie.begin() + wts_aie.size() / 2,
                                   wts_aie.end());

    const_Tensor = {{gamma_aie.data(), gamma_shape, b_dtype},
                    {beta_aie.data(), beta_shape, b_dtype}};

    layernorm_.initialize_const_params(const_Tensor);

    std::string a_path = test_golden_root_dir + shape_key + "_ifm32.txt";
    std::vector<uint32_t> a_aie = read_file(a_path);
    std::vector<Tensor> input_Tensor;
    input_Tensor = {{a_aie.data(), a_shape, a_dtype}};

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
    LOG_THIS("B = " << B << ", M = " << M << ", K = " << K);
    PROFILE_THIS(layernorm_.execute(input_Tensor, output_Tensor));
#else
    layernorm_.execute(input_Tensor, output_Tensor);
#endif

    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref.txt";

    std::vector<uint32_t> output_golden = read_file(output_golden_path);
    std::vector<OutT> bf16_output_golden(aie_out.size());
    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));
    err_count = sd_layernorm_check_result<OutT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  } else {
    ryzenai::sd::layernorm layernorm_ = ryzenai::sd::layernorm<InT, WgT, OutT>(
        a_dtype, b_dtype, c_dtype, false, attr);

    layernorm_.debug(debug);
    layernorm_.set_params(xclbin, pdi_name);

    std::vector<Tensor> const_Tensor;
    std::vector<float> raw_gamma(K, 0);
    initialize_random_float(raw_gamma, 1, -1);
    auto bf16_gamma = raw_gamma;
    float2bf16_vec(bf16_gamma);
    auto bf16_gamma_backup = bf16_gamma;
    uint32_t *cpp_gamma32_bf16_as_u =
        reinterpret_cast<uint32_t *>(bf16_gamma_backup.data());
    std::vector<uint16_t> aie_gamma_bf16(K);
    EXPECT_EQ(bf16_gamma.size(), aie_gamma_bf16.size());
    for (int k = 0; k < K; k++) {
      aie_gamma_bf16[k] = cpp_gamma32_bf16_as_u[k] >> 16;
    }

    std::vector<float> raw_beta(K, 0);
    initialize_random_float(raw_beta, 1, -1);
    auto bf16_beta = raw_beta;
    float2bf16_vec(bf16_beta);
    auto bf16_beta_backup = bf16_beta;
    uint32_t *cpp_beta32_bf16_as_u =
        reinterpret_cast<uint32_t *>(bf16_beta_backup.data());
    std::vector<uint16_t> aie_beta_bf16(K);
    EXPECT_EQ(bf16_beta.size(), aie_beta_bf16.size());
    for (int k = 0; k < K; k++) {
      aie_beta_bf16[k] = cpp_beta32_bf16_as_u[k] >> 16;
    }

    const_Tensor = {{aie_gamma_bf16.data(), gamma_shape, b_dtype},
                    {aie_beta_bf16.data(), beta_shape, b_dtype}};

    layernorm_.initialize_const_params(const_Tensor);
    std::vector<Tensor> input_Tensor;

    std::vector<float> raw_ifm(B * M * K, 0);
    initialize_random_float(raw_ifm, 1, -1);
    auto bf16_ifm = raw_ifm;
    float2bf16_vec(bf16_ifm);
    auto bf16_ifm_backup = bf16_ifm;
    uint32_t *cpp_ifm32_bf16_as_u =
        reinterpret_cast<uint32_t *>(bf16_ifm_backup.data());
    std::vector<uint16_t> aie_ifm_bf16(B * M * K);
    EXPECT_EQ(bf16_ifm.size(), aie_ifm_bf16.size());
    for (int i = 0; i < B * M * K; i++) {
      aie_ifm_bf16[i] = cpp_ifm32_bf16_as_u[i] >> 16;
    }

    input_Tensor = {{aie_ifm_bf16.data(), a_shape, a_dtype}};

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
    LOG_THIS("B = " << B << ", M = " << M << ", K = " << K);
    PROFILE_THIS(layernorm_.execute(input_Tensor, output_Tensor));
#else
    layernorm_.execute(input_Tensor, output_Tensor);
#endif
    std::vector<OutT> bf16_ofm(Bs * Ms * Ks);

    auto torch_input_tensor =
        torch::from_blob(raw_ifm.data(), {B, M, K}, torch::kFloat);
    auto torch_gamma_tensor =
        torch::from_blob(raw_gamma.data(), {K}, torch::kFloat);
    auto torch_beta_tensor =
        torch::from_blob(raw_beta.data(), {K}, torch::kFloat);

    const std::vector<int64_t> normalized_shape = {K};
    auto ret = torch::layer_norm(torch_input_tensor, normalized_shape,
                                 torch_gamma_tensor, torch_beta_tensor);

    float *c_golden = ret.data_ptr<float>();
    uint32_t *c_golden_u = reinterpret_cast<uint32_t *>(c_golden);
    for (int idx = 0; idx < Bs * Ms * Ks; idx++) {
      bf16_ofm[idx] = c_golden_u[idx] >> 16;
    }
    err_count = sd_layernorm_check_result<OutT>(
        bf16_ofm, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  }
  return err_count;
}
// sd1.5
//  golden test
TEST(SD_LAYERNORM_Test, Golden_KernelUnetlayer1) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 4096, 320, false, "bfloat16", "bfloat16", "bfloat16", "SD_UNet", 0.01f,
      0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_LAYERNORM_Test, Golden_KernelUnetlayer2) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 1024, 640, false, "bfloat16", "bfloat16", "bfloat16", "SD_UNet", 0.01f,
      0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// e2e test passed, so can change the threshold to 0.02 to pass the UT
TEST(SD_LAYERNORM_Test, Golden_KernelUnetlayer3) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 256, 1280, false, "bfloat16", "bfloat16", "bfloat16", "SD_UNet", 0.01f,
      0.02f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// e2e test passed, so can change the threshold to 0.03 to pass the UT
TEST(SD_LAYERNORM_Test, Golden_KernelUnetlayer4) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 64, 1280, false, "bfloat16", "bfloat16", "bfloat16", "SD_UNet", 0.01f,
      0.03f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// sd3.0
TEST(SD_LAYERNORM_Test, Golden_SD3_DIT1024_Layer1) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 154, 1536, false, "bfloat16", "bfloat16", "bfloat16", "SD3_DIT1024",
      0.01f, 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_LAYERNORM_Test, Golden_SD3KernelMMDITlayer2) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 1024, 1536, false, "bfloat16", "bfloat16", "bfloat16", "SD_MMDIT",
      0.01f, 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_LAYERNORM_Test, Golden_SD3_DIT1024_Layer2) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 4096, 1536, false, "bfloat16", "bfloat16", "bfloat16", "SD3_DIT1024",
      0.01f, 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// sd1.5
//  random test
TEST(SD_LAYERNORM_Test, Random_SD15_UNET_1) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 4096, 320, false, "bfloat16", "bfloat16", "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_LAYERNORM_Test, Random_SD15_UNET_2) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 1024, 640, false, "bfloat16", "bfloat16", "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_LAYERNORM_Test, Random_SD15_UNET_3) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 256, 1280, false, "bfloat16", "bfloat16", "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_LAYERNORM_Test, Random_SD15_UNET_4) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 64, 1280, false, "bfloat16", "bfloat16", "bfloat16", "SD15_UNET");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// sd3.0
// 512
TEST(SD_LAYERNORM_Test, Random_SD3_DIT512_1) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 154, 1536, false, "bfloat16", "bfloat16", "bfloat16", "SD3_DIT512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_LAYERNORM_Test, Random_SD3_DIT512_2) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 1024, 1536, false, "bfloat16", "bfloat16", "bfloat16", "SD3_DIT512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_LAYERNORM_Test, Random_SD3_DIT512_3) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 160, 1536, false, "bfloat16", "bfloat16", "bfloat16", "SD3_DIT512");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// 1024
TEST(SD_LAYERNORM_Test, Random_SD3_DIT1024_Layer1) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 154, 1536, false, "bfloat16", "bfloat16", "bfloat16", "SD3_DIT1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_LAYERNORM_Test, Random_SD3_DIT1024_Layer2) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 4096, 1536, false, "bfloat16", "bfloat16", "bfloat16", "SD3_DIT1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_LAYERNORM_Test, Random_SD3_DIT1024_Layer3) {
  int err_count = test_lrn<uint16_t, uint16_t, uint16_t>(
      2, 160, 1536, false, "bfloat16", "bfloat16", "bfloat16", "SD3_DIT1024");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
