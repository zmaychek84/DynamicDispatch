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
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/flat/mlp.hpp>

#include "test_common.hpp"

std::vector<float> read_npy(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file.");
  }

  char header[6];
  file.read(header, 6);
  if (std::memcmp(header, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("Not a valid .npy file.");
  }

  file.seekg(128, std::ios::beg);

  std::vector<float> data;
  float value;
  while (file.read(reinterpret_cast<char *>(&value), sizeof(float))) {
    data.push_back(value);
  }

  file.close();
  return data;
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

  std::vector<uint32_t> buffer;
  uint32_t temp;

  while (fileStream >> std::hex >> temp) {
    buffer.push_back(temp);
  }

  fileStream.close();
  return buffer;
}

template <typename T>
std::vector<T> readBinaryFile(const std::string &filePath) {
  std::ifstream file(filePath, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Unable to open : " << filePath << std::endl;
    return {};
  }

  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::cout << " open filePath  " << filePath << " fileSize  " << fileSize
            << std::endl;

  std::vector<T> buffer(fileSize / sizeof(T));
  if (file.read(reinterpret_cast<char *>(buffer.data()), fileSize)) {
    std::cout << "read  " << buffer.size() << " elements" << std::endl;
  } else {
    std::cerr << "fail to read file " << filePath << std::endl;
    buffer.clear(); // 清空缓冲区
  }
  return buffer;
}

template <typename T>
int flat_mlp_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
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
      if (err_count < 100) {
        std::cout << "ERROR: Y[" << i << "]: "
                  << "Expected: " << bfloat16_to_float(cpu_Y.at(i)) << ","
                  << "Received: " << bfloat16_to_float(aie_Y.at(i)) << "\n";
      }
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

static void initialize_random_float(std::vector<float> &vec, float max,
                                    float min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

static void initialize_random_uint(std::vector<uint8_t> &vec, uint8_t max_val,
                                   uint8_t min_val) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 255);
  for (auto &v : vec) {
    v = static_cast<uint8_t>(dis(gen));
  }
}

std::vector<uint8_t> convertToUint8(const std::vector<uint32_t> &input) {
  std::vector<uint8_t> output;
  output.resize(input.size() * sizeof(uint32_t));
  std::memcpy(output.data(), input.data(), input.size() * sizeof(uint32_t));
  return output;
}

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_flat_mlp(uint64_t M, uint64_t K, uint64_t N,
                  const std::string &a_dtype = "bfloat16",
                  const std::string &b_dtype = "uint4",
                  const std::string &c_dtype = "bfloat16",
                  float pixel_L2_norm_tolerance = 0.01,
                  bool test_with_golden = false,
                  const std::string &model_name = "flat") {
  int err_count = 0;
  float error_tolerance = 0.01f;
  std::vector<OuT> aie_out(M * N, 0);
  std::map<std::string, std::any> attr;
  std::vector<int> input_shape = {int(M), int(K), int(N)};
  std::vector<uint64_t> uint64t_input_shape = {M, K, N};
  attr["input_shape"] = input_shape;
  auto flat_mlp =
      ryzenai::flat::mlp<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype, false, attr);
  flat_mlp.debug(true);
  flat_mlp.set_params(uint64t_input_shape, false);
  std::vector<Tensor> const_tensors;
  std::vector<Tensor> input_Tensor;
  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), {}, c_dtype}};
  if (test_with_golden) {
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/flat_mlp/";

    // std::vector<uint32_t> gate_wts =
    //     read_file(test_golden_root_dir + "gate_wts.txt");
    // std::vector<uint32_t> gate_zp =
    //     read_file(test_golden_root_dir + "gate_zp.txt");
    // std::vector<float> gate_scale =
    //     read_npy(test_golden_root_dir + "gate_scale.npy");
    // std::vector<float> gate_bias =
    //     read_npy(test_golden_root_dir + "gate_bias.npy");
    // std::vector<uint32_t> up_wts =
    //     read_file(test_golden_root_dir + "up_wts.txt");
    // std::vector<uint32_t> up_zp = read_file(test_golden_root_dir +
    // "up_zp.txt"); std::vector<float> up_scale =
    //     read_npy(test_golden_root_dir + "up_scale.npy");
    // std::vector<float> up_bias = read_npy(test_golden_root_dir +
    // "up_bias.npy");
    std::vector<uint8_t> gate_wts = readBinaryFile<uint8_t>("gw.bin");
    std::vector<uint8_t> gate_zp(N * 24 / 2, 0);
    std::vector<float> gate_scale = readBinaryFile<float>("gs.bin");
    std::vector<float> gate_bias(N, 0);
    std::vector<uint8_t> up_wts = readBinaryFile<uint8_t>("uw.bin");
    std::vector<uint8_t> up_zp(N * 24 / 2, 0);
    std::vector<float> up_scale = readBinaryFile<float>("us.bin");
    std::vector<float> up_bias(N, 0);

    const_tensors.push_back({gate_wts.data(), {}, "uint8"});
    const_tensors.push_back({gate_scale.data(), {}, "float"});
    const_tensors.push_back({gate_zp.data(), {}, "uint8"});
    const_tensors.push_back({gate_bias.data(), {}, "float"});
    const_tensors.push_back({up_wts.data(), {}, "uint8"});
    const_tensors.push_back({up_scale.data(), {}, "float"});
    const_tensors.push_back({up_zp.data(), {}, "uint8"});
    const_tensors.push_back({up_bias.data(), {}, "float"});
    flat_mlp.initialize_const_params(const_tensors);

    std::vector<uint32_t> ifm_aie = readBinaryFile<uint32_t>("ifm.bin");
    // read_file("ifm32.txt");
    std::cout << "ifm_aie size is " << ifm_aie.size() * 4 << std::endl;
    input_Tensor.push_back({ifm_aie.data(), {}, "bfloat16"});
#ifdef UNIT_TEST_PERF
    LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
    PROFILE_THIS(flat_mlp.execute(input_Tensor, output_Tensor));
#else
    flat_mlp.execute(input_Tensor, output_Tensor);
#endif
    // std::string output_golden_path = test_golden_root_dir + "ofm32_ref.txt";
    // std::vector<uint32_t> output_golden = read_file("ofm32_ref.txt");

    std::vector<uint32_t> output_golden = readBinaryFile<uint32_t>("ofm.bin");

    std::vector<OuT> bf16_output_golden(aie_out.size());
    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));
    err_count = flat_mlp_check_result<OuT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  } else {
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/flat_mlp/";

    std::vector<float> raw_ifm(M * K, 0);
    initialize_random_float(raw_ifm, 1, -1);
    auto bf16_ifm = float_2_bf16_vec(raw_ifm);

    std::vector<uint8_t> gate_wts(K * N / 2, 0);
    initialize_random_uint(gate_wts, 255, 0);
    std::vector<uint8_t> gate_zp(K / 128 * N / 2, 0);
    initialize_random_uint(gate_zp, 255, 0);
    std::vector<float> raw_gate_scale(N * K / 128, 0);
    initialize_random_float(raw_gate_scale, 0.001, 0);
    auto bf16_gate_scale = float_2_bf16_vec(raw_gate_scale);
    std::vector<float> raw_gate_bias(N, 0);
    initialize_random_float(raw_gate_bias, 0.1, -0.1);
    auto bf16_gate_bias = float_2_bf16_vec(raw_gate_bias);

    std::vector<uint8_t> up_wts(K * N / 2, 0);
    initialize_random_uint(up_wts, 255, 0);
    std::vector<uint8_t> up_zp(K / 128 * N / 2, 0);
    initialize_random_uint(up_zp, 255, 0);
    std::vector<float> raw_up_scale(N * K / 128, 0);
    initialize_random_float(raw_up_scale, 0.001, 0);
    auto bf16_up_scale = float_2_bf16_vec(raw_up_scale);
    std::vector<float> raw_up_bias(N, 0);
    initialize_random_float(raw_up_bias, 0.1, -0.1);
    auto bf16_up_bias = float_2_bf16_vec(raw_up_bias);

    const_tensors.push_back({gate_wts.data(), {}, "uint8"});
    const_tensors.push_back({raw_gate_scale.data(), {}, "float"});
    const_tensors.push_back({gate_zp.data(), {}, "uint8"});
    const_tensors.push_back({raw_gate_bias.data(), {}, "float"});
    const_tensors.push_back({up_wts.data(), {}, "uint8"});
    const_tensors.push_back({raw_up_scale.data(), {}, "float"});
    const_tensors.push_back({up_zp.data(), {}, "uint8"});
    const_tensors.push_back({raw_up_bias.data(), {}, "float"});
    flat_mlp.initialize_const_params(const_tensors);
    input_Tensor.push_back({bf16_ifm.data(), {}, "bfloat16"});

#ifdef UNIT_TEST_PERF
    LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
    PROFILE_THIS(flat_mlp.execute(input_Tensor, output_Tensor));
#else
    flat_mlp.execute(input_Tensor, output_Tensor);
#endif
    auto torch_ifm =
        torch::from_blob(raw_ifm.data(), {int(M), int(K)}, torch::kFloat32);
    std::vector<float> quant_gate_wts(K * N);
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        int g_idx = (k / 128);
        uint8_t zp_value = gate_zp[K / 128 * n / 2 + g_idx / 2];
        zp_value = (g_idx & 1) ? (zp_value & 0xF0) >> 4 : (zp_value & 0xF);
        int zero = static_cast<int>(zp_value);

        uint8_t wts_value = gate_wts[(n * K + k) / 2];
        wts_value = (k & 1) ? (wts_value & 0xF0) >> 4 : (wts_value & 0xF);

        int wts = static_cast<int>(wts_value);
        quant_gate_wts[N * k + n] =
            (wts - zero) * raw_gate_scale[K / 128 * n + g_idx];
      }
    }
    auto torch_gate_wts = torch::from_blob(quant_gate_wts.data(),
                                           {int(K), int(N)}, torch::kFloat32);

    std::vector<float> quant_up_wts(K * N);
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        int g_idx = (k / 128);
        uint8_t zp_value = up_zp[K / 128 * n / 2 + g_idx / 2];
        zp_value = (g_idx & 1) ? (zp_value & 0xF0) >> 4 : (zp_value & 0xF);
        int zero = static_cast<int>(zp_value);
        uint8_t wts_value = up_wts[(n * K + k) / 2];
        wts_value = (k & 1) ? (wts_value & 0xF0) >> 4 : (wts_value & 0xF);
        int wts = static_cast<int>(wts_value);

        quant_up_wts[N * k + n] =
            (wts - zero) * raw_up_scale[K / 128 * n + g_idx];
      }
    }
    auto torch_up_wts = torch::from_blob(quant_up_wts.data(), {int(K), int(N)},
                                         torch::kFloat32);

    auto torch_gate_bias =
        torch::from_blob(raw_gate_bias.data(), {int(N)}, torch::kFloat32);
    auto torch_up_bias =
        torch::from_blob(raw_up_bias.data(), {int(N)}, torch::kFloat32);
    auto gate = torch::matmul(torch_ifm, torch_gate_wts);
    gate = gate + torch_gate_bias;
    auto up = torch::matmul(torch_ifm, torch_up_wts);
    up = up + torch_up_bias;
    auto silu_result = gate * torch::sigmoid(gate);
    auto elw_mul_result = silu_result * up;

    std::vector<float> float_vec(elw_mul_result.data_ptr<float>(),
                                 elw_mul_result.data_ptr<float>() +
                                     elw_mul_result.numel());

    std::vector<OuT> bf16_torch_out = float_2_bf16_vec(float_vec);
    err_count = flat_mlp_check_result<OuT>(
        bf16_torch_out, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  }
  return err_count;
}

// MLP
// Random test
// Unet
TEST(FLAT_MLP_Test, Random_KernelLayer1) {
  int err_count = test_flat_mlp<uint16_t, uint8_t, uint16_t>(
      1, 3072, 8192, "bfloat16", "uint4", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Golden test
// Unet
TEST(FLAT_MLP_Test, Golden_KernelLayer1) {
  int err_count = test_flat_mlp<uint16_t, uint8_t, uint16_t>(
      1, 3072, 8192, "bfloat16", "int4", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

void dumpVectorToBinFile(const std::string &filename,
                         const std::vector<uint8_t> &data) {
  std::ofstream outFile(filename, std::ios::binary | std::ios::out);
  if (!outFile) {
    std::cerr << "cannot open file\n";
    return;
  }

  outFile.write(reinterpret_cast<const char *>(data.data()), data.size());

  if (outFile.good()) {
    std::cout << "Dump data to " << filename << "\n";
  } else {
    std::cerr << "wite file error\n";
  }

  outFile.close();
}

TEST(FLAT_MLP_Test, Golden_TestShuffle) {
  uint64_t M = 1;
  uint64_t K = 3072;
  uint64_t N = 8192;
  std::map<std::string, std::any> attr;
  std::vector<int> input_shape = {int(M), int(K), int(N)};
  std::vector<uint64_t> uint64t_input_shape = {M, K, N};
  attr["input_shape"] = input_shape;
  auto flat_mlp =
      ryzenai::flat::mlp<std::uint16_t, std::uint8_t, std::uint16_t>(
          "bfloat16", "int4", "bfloat16", false, attr);
  flat_mlp.debug(true);
  flat_mlp.set_params(uint64t_input_shape, false);
  std::vector<Tensor> const_tensors;
  std::string wts_shuffle_test_dir =
      "tests/cpp/unit_tests/testDataMladf/flat_mlp/wts_shuffle/";
  // std::vector<uint32_t> gate_wts =
  //     read_file(wts_shuffle_test_dir + "gate_wts.txt");
  // std::vector<uint32_t> gate_zp =
  //     read_file(wts_shuffle_test_dir + "gate_zp.txt");
  // std::vector<float> gate_scale =
  //     read_npy(wts_shuffle_test_dir + "gate_scale.npy");
  // std::vector<float> gate_bias =
  //     read_npy(wts_shuffle_test_dir + "gate_bias.npy");
  // std::vector<uint32_t> up_wts = read_file(wts_shuffle_test_dir +
  // "up_wts.txt"); std::vector<uint32_t> up_zp = read_file(wts_shuffle_test_dir
  // + "up_zp.txt"); std::vector<float> up_scale = read_npy(wts_shuffle_test_dir
  // + "up_scale.npy"); std::vector<float> up_bias =
  // read_npy(wts_shuffle_test_dir + "up_bias.npy");
  std::vector<uint8_t> gate_wts = readBinaryFile<uint8_t>("gw.bin");
  std::vector<uint8_t> gate_zp(N * 24 / 2, 0);
  std::vector<float> gate_scale = readBinaryFile<float>("gs.bin");
  std::vector<float> gate_bias(N, 0);
  std::vector<uint8_t> up_wts = readBinaryFile<uint8_t>("uw.bin");
  std::vector<uint8_t> up_zp(N * 24 / 2, 0);
  std::vector<float> up_scale = readBinaryFile<float>("us.bin");
  std::vector<float> up_bias(N, 0);

  std::vector<uint8_t> bo_map;
  flat_mlp.wts_shuffle(bo_map, reinterpret_cast<uint8_t *>(gate_wts.data()),
                       reinterpret_cast<uint8_t *>(gate_zp.data()),
                       reinterpret_cast<float *>(gate_scale.data()),
                       reinterpret_cast<float *>(gate_bias.data()),
                       reinterpret_cast<uint8_t *>(up_wts.data()),
                       reinterpret_cast<uint8_t *>(up_zp.data()),
                       reinterpret_cast<float *>(up_scale.data()),
                       reinterpret_cast<float *>(up_bias.data()));
  // std::vector<uint32_t> wts_aie = read_file(wts_shuffle_test_dir +
  // "wts32.txt");
  std::vector<uint32_t> wts_aie = read_file("wts32.txt");
  uint8_t *wts_aie_uint8_ptr = reinterpret_cast<uint8_t *>(wts_aie.data());
  uint64_t diff_count = 0;
  std::ofstream outFile("pos_diff.txt", std::ios::app);

  for (int i = 0; i < bo_map.size(); i++) {
    auto exp = static_cast<int>(*(wts_aie_uint8_ptr + i));
    auto get = static_cast<int>(bo_map[i]);
    if (i % 4288 == 0) {
      outFile << "##################### " << i / 4288
              << " start #####################" << std::endl;
    }
    if (exp != get) {
      diff_count++;
      outFile << i % 4288 << std::endl;
    }
    if ((i + 1) % 4288 == 0) {
      outFile << "##################### " << i / 4288
              << " end #####################" << std::endl;
    }
  }
  outFile.close();
  EXPECT_TRUE(diff_count == 0)
      << "Wts shuffle diff_count Count = " << diff_count;
}
