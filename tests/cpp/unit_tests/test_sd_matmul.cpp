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
#include <ops/sd/matmul.hpp>

#include "test_common.hpp"

#define RANDOM_DATA

void dumpTensorToFile(const std::vector<float> &tensorData,
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

std::vector<uint32_t> read_file(const std::string &filePath) {
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
int sd_matmul_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
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

double round_half_2_even(double value) {
  // Set rounding mode to "round to nearest, ties to even"
  std::fesetround(FE_TONEAREST);

  // Use nearbyint, which rounds according to the current rounding mode
  return std::nearbyint(value);
}

void aie_srs_compute(std::vector<uint32_t> &input_output) {
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

std::vector<uint16_t> float_2_bf16_vec(std::vector<float> &x) {
  std::vector<uint32_t> x_uint32(x.size());
  std::memcpy(x_uint32.data(), x.data(), x.size() * sizeof(float));
  aie_srs_compute(x_uint32);

  std::vector<uint16_t> x_uint16(x.size());

  for (size_t i = 0; i < x_uint32.size(); ++i) {
    x_uint16[i] = static_cast<uint16_t>(x_uint32[i]);
  }

  return x_uint16;
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

template <typename InT = uint16_t, typename WgT = uint8_t,
          typename OuT = uint16_t>
int test_matmul(int B, int M, int K, int N, bool debug = false,
                const std::string &a_dtype = "bfloat16",
                const std::string &b_dtype = "bfloat16",
                const std::string &c_dtype = "bfloat16",
                float pixel_L2_norm_tolerance = 0.01f,
                bool test_with_golden = false,
                const std::string &model_name = "SD1.5") {
  std::map<std::string, std::string> txnbin_a_header = {{"bfloat16", "a16bf"}};
  std::map<std::string, std::string> txnbin_b_header = {{"float", "w16bf"}};
  std::map<std::string, std::string> txnbin_acc_header = {
      {"bfloat16", "acc16bf"}};
  int err_count = 0;
  float error_tolerance = 0.01f;
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
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Bs, Ms, Ks};
  std::vector<size_t> b_shape = {Ks * Ns};
  std::vector<size_t> aie_out_shape = {Bs, Ms, Ns};
  std::vector<OuT> aie_out(Bs * Ms * Ns);

  std::map<std::string, std::any> attr;
  attr["input_shape"] = std::vector<int>{B, M, K};
  attr["output_shape"] = std::vector<int>{B, M, N};
  attr["weight_shape"] = std::vector<int>{K, N};
  if (test_with_golden) {
    ryzenai::sd::matmul sd_matmul =
        ryzenai::sd::matmul<std::uint16_t, std::uint16_t, std::uint16_t>(
            a_dtype, b_dtype, c_dtype, false, attr);
    sd_matmul.debug(debug);
    std::vector<size_t> shapes = {Bs, Ms, Ks, Ns};
    sd_matmul.set_params(model_name, shapes);
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/sd_vae_dec_matmul/";
    std::string shape_key = "a16bfw16bfacc16bf_" + std::to_string(Bs) + "_" +
                            std::to_string(Ms) + "_" + std::to_string(Ks) +
                            "_" + std::to_string(Ns);
    // may need to change according to provided data
    std::string ifm_path = test_golden_root_dir + shape_key + "_ifm32.txt";
    std::vector<uint32_t> a_aie = read_file(ifm_path);
    // may need to change according to provided data
    std::string wts_path = test_golden_root_dir + shape_key + "_wts32.txt";
    std::vector<uint32_t> b_aie = read_file(wts_path);
    std::vector<Tensor> const_Tensor;
    const_Tensor = {{b_aie.data(), b_shape, b_dtype}};
    sd_matmul.initialize_const_params(const_Tensor);
    std::vector<Tensor> input_Tensor;

    input_Tensor = {{a_aie.data(), a_shape, a_dtype}};

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
    LOG_THIS("B = " << Bs << ", M = " << Ms << ", K = " << Ks
                    << ", N = " << Ns);
    PROFILE_THIS(sd_matmul.execute(input_Tensor, output_Tensor));
#else
    sd_matmul.execute(input_Tensor, output_Tensor);
#endif
    // may need to change according to provided data
    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref.txt";

    std::vector<uint32_t> output_golden = read_file(output_golden_path);
    std::vector<OuT> bf16_output_golden(aie_out.size());
    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));

    err_count = sd_matmul_check_result<OuT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  } else {
    ryzenai::sd::matmul sd_matmul =
        ryzenai::sd::matmul<std::uint16_t, std::uint16_t, std::uint16_t>(
            a_dtype, b_dtype, c_dtype, false, attr);
    sd_matmul.debug(debug);
    std::vector<size_t> shapes = {Bs, Ms, Ks, Ns};
    sd_matmul.set_params(model_name, shapes);
    // gen rand

    std::vector<float> raw_wts(Ks * Ns, 0);
    initialize_random_float(raw_wts, 1, -1);
    // auto bf16_wts = float_2_bf16_vec(raw_wts);
    auto bf16_wts = sd_matmul.shuffle_wts_bf16(raw_wts);

    std::vector<float> raw_ifms(Bs * Ms * Ks, 0);
    initialize_random_float(raw_ifms, 1, -1);
    auto bf16_ifms = float_2_bf16_vec(raw_ifms);

    std::vector<Tensor> const_tensors;

    const_tensors.push_back({bf16_wts.data(), b_shape, "bfloat16"});

    sd_matmul.initialize_const_params(const_tensors);

    std::vector<Tensor> input_Tensor;
    input_Tensor = {{bf16_ifms.data(), a_shape, a_dtype}};
    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
    LOG_THIS("B = " << Bs << ", M = " << Ms << ", K = " << Ks
                    << ", N = " << Ns);
    PROFILE_THIS(sd_matmul.execute(input_Tensor, output_Tensor));
#else
    sd_matmul.execute(input_Tensor, output_Tensor);
#endif

    std::vector<OuT> bf16_ofm(Bs * Ms * Ns);

    auto torch_input_tensor =
        torch::from_blob(raw_ifms.data(), {B, M, K}, torch::kFloat);
    auto torch_wts_tensor =
        torch::from_blob(raw_wts.data(), {K, N}, torch::kFloat);
    auto ret = torch::matmul(torch_input_tensor, torch_wts_tensor);
    float *c_golden = ret.data_ptr<float>();
    uint32_t *c_golden_u = reinterpret_cast<uint32_t *>(c_golden);
    for (int b = 0; b < Bs; b++) {
      for (int m = 0; m < Ms; m++) {
        for (int n = 0; n < Ns; n++) {
          size_t idx = n + m * Ns + b * Ms * Ns;
          bf16_ofm[idx] = c_golden_u[idx] >> 16;
        }
      }
    }
    err_count = sd_matmul_check_result<OuT>(bf16_ofm, aie_out, error_tolerance,
                                            pixel_L2_norm_tolerance);
  }
  return err_count;
}

// Matmul
// Random test
// Unet
TEST(SD_MATMUL_Test, Random_KernelUnetlayer1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 2560, 640, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 640, 5120, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 640, 640, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 256, 1280, 10240, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer5) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 256, 1280, 1280, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer6) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 256, 5120, 1280, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer7) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 1280, 320, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer8) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 320, 2560, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer9) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 320, 320, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer10) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 64, 1280, 10240, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer11) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 64, 1280, 1280, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer12) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 64, 5120, 1280, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer13) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 77, 768, 1280, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer14) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 77, 768, 320, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetlayer15) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 77, 768, 640, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Vae
TEST(SD_MATMUL_Test, Random_KernelVaelayer1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      1, 4096, 512, 512, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from unet gemm1
TEST(SD_MATMUL_Test, Random_KernelUnetGemm1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 320, 1280, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from unet gemm2
TEST(SD_MATMUL_Test, Random_KernelUnetGemm2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 1280, 320, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from unet gemm3
TEST(SD_MATMUL_Test, Random_KernelUnetGemm3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 1280, 640, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from unet gemm4
TEST(SD_MATMUL_Test, Random_KernelUnetGemm4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 1280, 1280, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// 4 shapes from matmul_add_to_matmul pass
TEST(SD_MATMUL_Test, Random_KernelUnetMmadd1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 64, 1280, 5120, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetMmadd2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 256, 1280, 5120, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetMmadd3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 640, 2560, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_KernelUnetMmadd4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 320, 1280, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from sd3 mmdit512 - Random
TEST(SD_MATMUL_Test, Random_SD3KernelMMDIT512layer1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 1536, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_SD3KernelMMDIT512layer2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 1536, 6144, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_SD3KernelMMDIT512layer3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 1536, 64, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_SD3KernelMMDIT512layer4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 6144, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_SD3KernelMMDIT512layer10) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 2048, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_SD3KernelMMDIT512layer11) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 256, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from sd3 mmdit 512 and 1024
TEST(SD_MATMUL_Test, Random_SD3KernelMMDITCommonlayer1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 1536, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_SD3KernelMMDITCommonlayer2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 154, 1536, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_SD3KernelMMDITCommonlayer3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 154, 1536, 6144, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_SD3KernelMMDITCommonlayer4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 154, 4096, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Random_SD3KernelMMDITCommonlayer5) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 154, 6144, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from sd3 mmdit1024- Random
TEST(SD_MATMUL_Test, Random_SD3KernelMMDIT1024layer1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 1536, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from sd3 mmdit1024- Random
TEST(SD_MATMUL_Test, Random_SD3KernelMMDIT1024layer2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 1536, 6144, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from sd3 mmdit 1024 - Random
TEST(SD_MATMUL_Test, Random_SD3KernelMMDIT1024layer3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 1536, 64, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from sd3 mmdit 1024 - Random
TEST(SD_MATMUL_Test, Random_SD3KernelMMDIT1024layer4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 6144, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from  sd3 vae decoder 1024
TEST(SD_MATMUL_Test, Random_SD3KernelDecoder1024layer1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      1, 16384, 512, 512, false, "bfloat16", "bfloat16", "bfloat16", 0.01f);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Golden test
// Unet
TEST(SD_MATMUL_Test, Golden_KernelUnetlayer1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 2560, 640, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 640, 5120, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 640, 640, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 256, 1280, 10240, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer5) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 256, 1280, 1280, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer6) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 256, 5120, 1280, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer7) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 1280, 320, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer8) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 320, 2560, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer9) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 320, 320, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer10) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 64, 1280, 10240, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer11) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 64, 1280, 1280, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer12) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 64, 5120, 1280, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer13) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 77, 768, 1280, false, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer14) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 77, 768, 320, false, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetlayer15) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 77, 768, 640, false, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Vae
TEST(SD_MATMUL_Test, Golden_KernelVaelayer1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      1, 4096, 512, 512, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from unet gemm1
TEST(SD_MATMUL_Test, Golden_KernelUnetGemm1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 320, 1280, false, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from unet gemm2
TEST(SD_MATMUL_Test, Golden_KernelUnetGemm2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 1280, 320, false, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from unet gemm3
TEST(SD_MATMUL_Test, Golden_KernelUnetGemm3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 1280, 640, false, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from unet gemm4
TEST(SD_MATMUL_Test, Golden_KernelUnetGemm4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 1280, 1280, false, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from sd3 mmdit 512
TEST(SD_MATMUL_Test, Golden_SD3KernelMMDIT512layer1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 1536, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_SD3KernelMMDIT512layer2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 1536, 6144, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_SD3KernelMMDIT512layer3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 1536, 64, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_SD3KernelMMDIT512layer4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 6144, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_SD3KernelMMDIT512layer10) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 2048, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_SD3KernelMMDIT512layer11) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 256, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from sd3 mmdit 512 and 1024
TEST(SD_MATMUL_Test, Golden_SD3KernelMMDITCommonlayer1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1, 1536, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_SD3KernelMMDITCommonlayer2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 154, 1536, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_SD3KernelMMDITCommonlayer3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 154, 1536, 6144, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_SD3KernelMMDITCommonlayer4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 154, 4096, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_SD3KernelMMDITCommonlayer5) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 154, 6144, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from  sd3 mmdit 1024
TEST(SD_MATMUL_Test, Golden_SD3KernelMMDIT1024layer1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 1536, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from  sd3 mmdit 1024
TEST(SD_MATMUL_Test, Golden_SD3KernelMMDIT1024layer2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 1536, 6144, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from  sd3 mmdit 1024
TEST(SD_MATMUL_Test, Golden_SD3KernelMMDIT1024layer3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 1536, 64, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from  sd3 mmdit 1024
TEST(SD_MATMUL_Test, Golden_SD3KernelMMDIT1024layer4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 6144, 1536, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// from  sd3 vae decoder 1024
TEST(SD_MATMUL_Test, Golden_SD3KernelDecoder1024layer1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      1, 16384, 512, 512, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// 4 shapes from matmul_add_to_matmul pass
TEST(SD_MATMUL_Test, Golden_KernelUnetMmadd1) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 64, 1280, 5120, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetMmadd2) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 256, 1280, 5120, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetMmadd3) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 1024, 640, 2560, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MATMUL_Test, Golden_KernelUnetMmadd4) {
  int err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
      2, 4096, 320, 1280, false, "bfloat16", "bfloat16", "bfloat16", 0.01f,
      true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
