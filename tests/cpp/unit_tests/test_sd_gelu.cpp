/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <cfenv>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <torch/torch.h>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/sd/gelu.hpp>

#include "test_common.hpp"
using namespace matmul_matrix;
using namespace std;

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

inline double round_half_to_even(double value) {
  // Set rounding mode to "round to nearest, ties to even"
  std::fesetround(FE_TONEAREST);

  // Use nearbyint, which rounds according to the current rounding mode
  return std::nearbyint(value);
}

inline void aie_srs(std::vector<uint32_t> &input_output) {
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
inline uint32_t float_as_uint(float v) {
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
inline float py3_round(float x) {
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

inline void initialize_random_float(std::vector<float> &vec, int max, int min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

template <typename T>
int sd_gelu_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
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

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OutT = uint16_t>
int test_gelu(int B, int M, int N, bool debug = false,
              const std::string &a_dtype = "bfloat16",
              const std::string &b_dtype = "bfloat16",
              const std::string &c_dtype = "bfloat16",
              const std::string &model_name = "SD_VAE_DEC",
              float pixel_L2_norm_tolerance = 0.01,
              bool test_with_golden = false) {
  float error_tolerance = 0.01;
  int err_count = 0;
  size_t Bs = static_cast<size_t>(B);
  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Bs, Ms, Ns};
  std::vector<size_t> aie_out_shape = {Bs, Ms, Ns};
  std::vector<OutT> aie_out(B * M * N);
  std::map<std::string, std::any> attr;
  attr["input_shape"] = std::vector<int>{B, M, N};
  attr["output_shape"] = std::vector<int>{B, M, N};
  std::map<std::string, std::string> txnbin_a_header = {{"bfloat16", "a16bf"},
                                                        {"uint16", "a16"}};
  std::map<std::string, std::string> txnbin_acc_header = {
      {"bfloat16", "acc16bf"}, {"uint16", "acc16"}};
  std::vector<WgT> dummy_wts(64, 0); // 128 bytes 0
  std::vector<Tensor> const_Tensor;
  const_Tensor.push_back({dummy_wts.data(), {64}, "bfloat16"});
  if (test_with_golden) {
    ryzenai::sd::gelu sd_gelu = ryzenai::sd::gelu<InT, WgT, OutT>(
        a_dtype, b_dtype, c_dtype, false, attr);
    sd_gelu.debug(debug);
    sd_gelu.set_params();
    sd_gelu.initialize_const_params(const_Tensor);

    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/sd_gelu/";
    std::string shape_key =
        txnbin_a_header.at(a_dtype) + txnbin_acc_header.at(c_dtype) + "_" +
        std::to_string(B) + "_" + std::to_string(M) + "_" + std::to_string(N);
    std::string ifm_path = test_golden_root_dir + shape_key + "_ifm32.txt";
    std::vector<uint32_t> a_aie = read_hex_file(ifm_path);

    std::vector<Tensor> input_Tensor;
    input_Tensor = {{a_aie.data(), a_shape, a_dtype}};

    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
    LOG_THIS("B = " << B << ", M = " << M << ", N = " << N);
    PROFILE_THIS(sd_gelu.execute(input_Tensor, output_Tensor));
#else
    sd_gelu.execute(input_Tensor, output_Tensor);
#endif

    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref.txt";

    std::vector<uint32_t> output_golden = read_hex_file(output_golden_path);
    std::vector<OutT> bf16_output_golden(aie_out.size());
    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));
    err_count = sd_gelu_check_result<OutT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);

  } else {
    ryzenai::sd::gelu sd_gelu = ryzenai::sd::gelu<InT, WgT, OutT>(
        a_dtype, b_dtype, c_dtype, false, attr);
    sd_gelu.debug(debug);
    sd_gelu.set_params();
    sd_gelu.initialize_const_params(const_Tensor);
    std::vector<float> raw_ifms(B * M * N, 0);
    initialize_random_float(raw_ifms, 2, -2);
    auto bf16_ifms = raw_ifms;
    float2bf16_vec(bf16_ifms);
    auto bf16_ifms_backup = bf16_ifms;
    uint32_t *cpp_ifm32_bf16_as_u =
        reinterpret_cast<uint32_t *>(bf16_ifms_backup.data());

    std::vector<uint16_t> aie_ifm_bf16(B * M * N);
    for (int b = 0; b < B; b++) {
      for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
          int index = b * M * N + m * N + n;
          aie_ifm_bf16[index] = cpp_ifm32_bf16_as_u[index] >> 16;
        }
      }
    }
    std::vector<Tensor> input_Tensor;
    input_Tensor = {{aie_ifm_bf16.data(), a_shape, a_dtype}};
    std::vector<Tensor> output_Tensor;
    output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};
#ifdef UNIT_TEST_PERF
    LOG_THIS("B = " << B << ", M = " << M << ", N = " << N);
    PROFILE_THIS(sd_gelu.execute(input_Tensor, output_Tensor));
#else
    sd_gelu.execute(input_Tensor, output_Tensor);
#endif
    std::vector<OutT> bf16_ofm(Bs * Ms * Ns);
    auto torch_input_tensor =
        torch::from_blob(raw_ifms.data(), {B, M, N}, torch::kFloat);
    auto ret = torch::gelu(torch_input_tensor);
    float *c_golden = ret.data_ptr<float>();
    uint32_t *c_golden_u = reinterpret_cast<uint32_t *>(c_golden);
    for (int b = 0; b < Bs; b++) {
      for (int m = 0; m < Ms; m++) {
        for (int n = 0; n < Ns; n++) {
          int idx = n + m * Ns + b * Ms * Ns;
          bf16_ofm[idx] = c_golden_u[idx] >> 16;
        }
      }
    }
    err_count = sd_gelu_check_result<OutT>(bf16_ofm, aie_out, error_tolerance,
                                           pixel_L2_norm_tolerance);
  }
  return err_count;
}

// Random test
// Unet
// sd1.5
TEST(SD_GELU_Test, Random_KernelUnetlayer1) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 64, 5120, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GELU_Test, Random_KernelUnetlayer2) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 1024, 2560, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GELU_Test, Random_KernelUnetlayer3) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 256, 5120, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GELU_Test, Random_KernelUnetlayer4) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 4096, 1280, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
// sd3.0
TEST(SD_GELU_Test, Random_SD3KernelUnetlayer1) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 154, 6144, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GELU_Test, Random_SD3KernelUnetlayer2) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 1024, 6144, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GELU_Test, Random_SD3KernelUnetlayer3) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 4096, 6144, false, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Golden test
// Unet
TEST(SD_GELU_Test, Golden_KernelUnetlayer1) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 64, 5120, false, "bfloat16", "bfloat16", "bfloat16", "SD_VAE_DEC",
      0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GELU_Test, Golden_KernelUnetlayer2) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 1024, 2560, false, "bfloat16", "bfloat16", "bfloat16", "SD_VAE_DEC",
      0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GELU_Test, Golden_KernelUnetlayer3) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 256, 5120, false, "bfloat16", "bfloat16", "bfloat16", "SD_VAE_DEC",
      0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GELU_Test, Golden_KernelUnetlayer4) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 4096, 1280, false, "bfloat16", "bfloat16", "bfloat16", "SD_VAE_DEC",
      0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// sd3.0
TEST(SD_GELU_Test, Golden_SD3KernelUnetlayer1) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 154, 6144, false, "bfloat16", "bfloat16", "bfloat16", "SD_MMDIT",
      0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GELU_Test, Golden_SD3KernelUnetlayer2) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 1024, 6144, false, "bfloat16", "bfloat16", "bfloat16", "SD_MMDIT",
      0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_GELU_Test, Golden_SD3KernelUnetlayer3) {
  int err_count = test_gelu<uint16_t, uint16_t, uint16_t>(
      2, 4096, 6144, false, "bfloat16", "bfloat16", "bfloat16", "SD_MMDIT",
      0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
