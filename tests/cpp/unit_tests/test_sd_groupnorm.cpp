/*
 * Copyright � 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

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
#include <ops/sd/groupnorm.hpp>

#include "test_common.hpp"

#define RANDOM_DATA

static void dumpTensorToFile(const std::vector<float> &tensorData,
                             const std::string &fileName, size_t Bs, size_t Ms,
                             size_t Ns) {
  std::ofstream outFile(fileName); // Create an output file stream
  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open the file for writing: " << fileName
              << std::endl;
    return;
  }

  for (size_t i = 0; i < tensorData.size(); ++i) {
    outFile << tensorData[i]; // Write the tensor value
    if ((i + 1) % Ns == 0) {  // Newline after every Ns elements (for 3D shape)
      outFile << "\n";
    } else {
      outFile << " "; // Space between elements
    }
  }

  outFile.close(); // Close the file
  std::cout << "Tensor data written to: " << fileName << std::endl;
}

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
int sd_groupnorm_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
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
int test_groupnorm(int B, int H, int W, int C, bool debug = false,
                   const std::string &a_dtype = "bfloat16",
                   const std::string &b_dtype = "bfloat16",
                   const std::string &c_dtype = "bfloat16",
                   float pixel_L2_norm_tolerance = 0.01,
                   bool test_with_golden = false,
                   const std::string &model_name = "SD1.5") {
  std::map<std::string, std::string> txnbin_a_header = {{"bfloat16", "a16bf"}};
  std::map<std::string, std::string> txnbin_b_header = {{"bfloat16", "w16bf"}};
  std::map<std::string, std::string> txnbin_acc_header = {
      {"bfloat16", "acc16bf"}};
  int err_count = 0;
  float error_tolerance = 0.01;
  if (a_dtype != "bfloat16") {
    throw std::invalid_argument("a_dtype is not supported");
  }
  if (b_dtype != "bfloat16") {
    throw std::invalid_argument("b_dtype is not supported");
  }
  if (c_dtype != "bfloat16") {
    throw std::invalid_argument("c_dtype is not supported");
  }
  size_t Bs = static_cast<size_t>(B);
  size_t Hs = static_cast<size_t>(H);
  size_t Ws = static_cast<size_t>(W);
  size_t Cs = static_cast<size_t>(C);
  std::vector<size_t> a_shape = {Bs, Hs, Ws, Cs};
  std::vector<size_t> wts_shape = {Cs * 2};
  std::vector<size_t> g_shape = {Cs};
  std::vector<size_t> aie_out_shape = {Bs, Hs, Ws, Cs};
  std::vector<OuT> aie_out(Bs * Hs * Ws * Cs);

  std::map<std::string, std::any> attr;
  attr["input_shape"] = std::vector<int>{B, H, W, C};
  attr["output_shape"] = std::vector<int>{B, H, W, C};
  attr["wts_shape"] = std::vector<int>{C * 2};

  if (test_with_golden) {
    ryzenai::sd::groupnorm sd_groupnorm =
        ryzenai::sd::groupnorm<std::uint16_t, std::uint16_t, std::uint16_t>(
            a_dtype, b_dtype, c_dtype, false, attr);
    sd_groupnorm.debug(debug);
    std::vector<size_t> shapes = {Bs, Hs, Ws, Cs};
    sd_groupnorm.set_params();
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/sd_groupnorm/";
    std::string shape_key = txnbin_a_header.at(a_dtype) +
                            txnbin_b_header.at(b_dtype) +
                            txnbin_acc_header.at(c_dtype) + "_" +
                            std::to_string(Bs) + "_" + std::to_string(Hs) +
                            "_" + std::to_string(Ws) + "_" + std::to_string(Cs);
    // may need to change according to provided data
    std::string ifm_path = test_golden_root_dir + shape_key + "_ifm32.txt";
    std::vector<uint32_t> a_aie = read_file(ifm_path);
    // may need to change according to provided data
    std::string wts_path = test_golden_root_dir + shape_key + "_wts32.txt";
    std::vector<uint32_t> wts_aie = read_file(wts_path);

    std::vector<Tensor> const_Tensor;
    const_Tensor = {{wts_aie.data(), wts_shape, "bfloat16"}};
    sd_groupnorm.initialize_const_params(const_Tensor);
    std::vector<Tensor> input_Tensor;

    input_Tensor = {{a_aie.data(), a_shape, a_dtype}};

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
    LOG_THIS("B = " << Bs << ", H = " << Hs << ", W = " << Ws
                    << ", C = " << Cs);
    PROFILE_THIS(sd_groupnorm.execute(input_Tensor, output_Tensor));
#else
    sd_groupnorm.execute(input_Tensor, output_Tensor);
#endif
    // may need to change according to provided data
    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref.txt";

    std::vector<uint32_t> output_golden = read_file(output_golden_path);
    std::vector<OuT> bf16_output_golden(aie_out.size());
    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));

    err_count = sd_groupnorm_check_result<OuT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  } else {
    ryzenai::sd::groupnorm sd_groupnorm =
        ryzenai::sd::groupnorm<std::uint16_t, std::uint16_t, std::uint16_t>(
            a_dtype, b_dtype, c_dtype, false, attr);
    sd_groupnorm.debug(debug);
    std::vector<size_t> shapes = {Bs, Hs, Ws, Cs};
    sd_groupnorm.set_params();
    // gen rand
    std::vector<float> raw_gamma(Cs, 0);
    initialize_random_float(raw_gamma, 1, -1);

    std::vector<float> raw_beta(Cs, 0);
    initialize_random_float(raw_beta, 1, -1);
    // Concatenate raw_gamma and raw_beta into raw_wts
    std::vector<float> raw_wts;
    raw_wts.reserve(raw_gamma.size() + raw_beta.size()); // Pre-allocate memory
    raw_wts.insert(raw_wts.end(), raw_gamma.begin(), raw_gamma.end());
    raw_wts.insert(raw_wts.end(), raw_beta.begin(), raw_beta.end());
    auto bf16_wts = float_2_bf16_vec(raw_wts);

    std::vector<float> raw_ifms(Bs * Hs * Ws * Cs, 0);
    initialize_random_float(raw_ifms, 1, -1);
    auto bf16_ifms = float_2_bf16_vec(raw_ifms);

    std::vector<Tensor> const_tensors;

    const_tensors.push_back({bf16_wts.data(), wts_shape, "bfloat16"});
    sd_groupnorm.initialize_const_params(const_tensors);

    std::vector<Tensor> input_Tensor;
    input_Tensor = {{bf16_ifms.data(), a_shape, a_dtype}};
    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};
#ifdef UNIT_TEST_PERF
    LOG_THIS("B = " << Bs << ", H = " << Hs << ", W = " << Ws
                    << ", C = " << Cs);
    PROFILE_THIS(sd_groupnorm.execute(input_Tensor, output_Tensor));
#else
    sd_groupnorm.execute(input_Tensor, output_Tensor);
#endif

    std::vector<OuT> bf16_ofm(Bs * Hs * Ws * Cs);

    std::vector<float> raw_ifm_nchw(Bs * Cs * Hs * Ws, 0);

    for (int b = 0; b < Bs; ++b) {
      for (int h = 0; h < Hs; ++h) {
        for (int w = 0; w < Ws; ++w) {
          for (int c = 0; c < Cs; ++c) {
            // Compute original and transposed indices
            int original_index =
                b * (Hs * Ws * Cs) + h * (Ws * Cs) + w * Cs + c;
            int transposed_index =
                b * (Cs * Hs * Ws) + c * (Hs * Ws) + h * Ws + w;

            // Assign the value to the transposed position
            raw_ifm_nchw[transposed_index] = raw_ifms[original_index];
          }
        }
      }
    }

    auto torch_input_tensor =
        torch::from_blob(raw_ifm_nchw.data(), {B, C, H, W}, torch::kFloat);

    auto torch_gamma_tensor =
        torch::from_blob(raw_gamma.data(), {C}, torch::kFloat);
    auto torch_beta_tensor =
        torch::from_blob(raw_beta.data(), {C}, torch::kFloat);
    // all groupnorm in unet or vae decoder is 32
    int group_num = 32;
    double eps = 0.000009999999747378752;
    auto ret = torch::group_norm(torch_input_tensor, group_num,
                                 torch_gamma_tensor, torch_beta_tensor, eps);
    float *c_golden = ret.data_ptr<float>();
    uint32_t *c_golden_u = reinterpret_cast<uint32_t *>(c_golden);
    for (int b = 0; b < Bs; b++) {
      for (int c = 0; c < Cs; c++) {
        for (int h = 0; h < Hs; h++) {
          for (int w = 0; w < Ws; w++) {
            int original_index =
                b * (Cs * Hs * Ws) + c * (Hs * Ws) + h * Ws + w;
            int transposed_index =
                b * (Hs * Ws * Cs) + h * (Ws * Cs) + w * Cs + c;

            bf16_ofm[transposed_index] = c_golden_u[original_index] >> 16;
          }
        }
      }
    }
    err_count = sd_groupnorm_check_result<OuT>(
        bf16_ofm, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  }
  return err_count;
}

// sd1.5
// golden test
TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer1) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 128, 128, 512, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer2) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 256, 256, 256, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer3) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 256, 256, 512, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer4) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 512, 512, 128, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer5) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 512, 512, 256, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer6) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 64, 64, 512, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer7) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 16, 16, 1280, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer8) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 16, 16, 1920, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer9) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 16, 16, 2560, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer10) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 16, 16, 640, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer11) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 32, 32, 1280, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer12) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 32, 32, 1920, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer13) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 32, 32, 320, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer14) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 32, 32, 640, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer15) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 32, 32, 960, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer16) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 64, 64, 320, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer17) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 64, 64, 640, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer18) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 64, 64, 960, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer19) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 8, 8, 1280, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelUnetlayer20) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 8, 8, 2560, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// sd3.0
TEST(SD_GROUPNORM_Test, Golden_KernelSD3Unetlayer1) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 512, 512, 512, false, "bfloat16", "bfloat16", "bfloat16", 0.01, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelSD3Unetlayer2) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 1024, 1024, 128, false, "bfloat16", "bfloat16", "bfloat16", 0.01,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Golden_KernelSD3Unetlayer3) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 1024, 1024, 256, false, "bfloat16", "bfloat16", "bfloat16", 0.01,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// sd1.5
// Random test
TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer1) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 128, 128, 512, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer2) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 256, 256, 256, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer3) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 256, 256, 512, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer4) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 512, 512, 128, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer5) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 512, 512, 256, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer6) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 64, 64, 512, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer7) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 16, 16, 1280, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer8) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 16, 16, 1920, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer9) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 16, 16, 2560, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer10) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 16, 16, 640, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer11) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 32, 32, 1280, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer12) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 32, 32, 1920, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer13) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 32, 32, 320, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer14) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 32, 32, 640, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer15) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 32, 32, 960, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer16) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 64, 64, 320, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer17) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 64, 64, 640, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer18) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 64, 64, 960, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer19) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 8, 8, 1280, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelUnetlayer20) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      2, 8, 8, 2560, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// sd3.0
TEST(SD_GROUPNORM_Test, Random_KernelSD3Unetlayer1) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 512, 512, 512, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelSD3Unetlayer2) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 1024, 1024, 128, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GROUPNORM_Test, Random_KernelSD3Unetlayer3) {
  int err_count = test_groupnorm<uint16_t, uint16_t, uint16_t>(
      1, 1024, 1024, 256, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
