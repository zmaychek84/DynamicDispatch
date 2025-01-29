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
#include <ops/sd/mha.hpp>

#include "test_common.hpp"

torch::Tensor scaled_dot_product_attention(
    const torch::Tensor &query, // Shape: (batch_size, seq_len_q, embed_dim)
    const torch::Tensor &key,   // Shape: (batch_size, seq_len_kv, embed_dim)
    const torch::Tensor &value, // Shape: (batch_size, seq_len_kv, embed_dim)
    int num_heads) {

  int batch_size = (int)query.size(0);
  int seq_len_q = (int)query.size(1);
  int seq_len_kv = (int)key.size(1);
  int embed_dim = (int)query.size(2);
  int head_dim = embed_dim / num_heads;
  double scale_factor = 1.0 / std::sqrt(static_cast<double>(head_dim));

  auto query_reshaped =
      query.view({batch_size, seq_len_q, num_heads, head_dim}).transpose(1, 2);
  auto key_reshaped =
      key.view({batch_size, seq_len_kv, num_heads, head_dim}).transpose(1, 2);
  auto value_reshaped =
      value.view({batch_size, seq_len_kv, num_heads, head_dim}).transpose(1, 2);

  auto scores = torch::matmul(query_reshaped, key_reshaped.transpose(-2, -1)) *
                scale_factor;
  auto attn_weights = torch::softmax(scores, -1);
  auto attn_output = torch::matmul(attn_weights, value_reshaped);
  attn_output = attn_output.transpose(1, 2).contiguous();
  attn_output = attn_output.view({batch_size, seq_len_q, embed_dim});
  return attn_output;
}

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
int sd_mha_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
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

static void initialize_random_float(std::vector<float> &vec, float max,
                                    float min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

uint16_t float_to_bfloat16(float x) {
  uint32_t i;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *tmp = (uint8_t *)&i;
  std::memcpy(tmp, src, sizeof(float));
  uint32_t lsb = (i >> 16) & 0x1;
  uint32_t bias = 0x7fff + lsb;
  i += bias;
  uint16_t y = uint16_t(i >> 16);
  return y;
}

template <typename InT = uint16_t, typename WgT = uint8_t,
          typename OuT = uint16_t>
int test_sd_mha(int B, int M, int K, int N, int heads,
                const std::string &a_dtype = "bfloat16",
                const std::string &b_dtype = "bfloat16",
                const std::string &c_dtype = "bfloat16",
                float pixel_L2_norm_tolerance = 0.01,
                bool test_with_golden = false,
                const std::string &model_name = "SD1.5") {
  int err_count = 0;
  float error_tolerance = 0.01f;
  std::vector<OuT> aie_out(B * M * K, 0);
  std::map<std::string, std::any> attr;
  attr["input_shape"] = std::vector<int>{B, M, K, N};
  attr["num_heads"] = std::vector<int>{heads};
  auto sd_mha = ryzenai::sd::mha<std::uint16_t, std::uint16_t, std::uint16_t>(
      a_dtype, b_dtype, c_dtype, false, attr);
  sd_mha.debug(true);
  sd_mha.set_params();
  // generate mask
  int N_padded = N < 128 ? 128 : N;
  uint64_t N_256_aligned = Utils::align_to_next(N, 256);
  auto out_bo_bf16_cnt = B * K * N_256_aligned;
  // vae use 2 dims for mask
  std::vector<uint16_t> mask(M * N_padded, 0);
  // ff80 is -inf, ff7f is closest bfloat16 to -inf
  const uint16_t neg_inf_ui16 = 0xff80;
  if (B == 2) {
    // unet only need 1 dim for mask
    if (K == 1536) {
      // sd3 mmdit mha
      aie_out.resize(out_bo_bf16_cnt, 0);
      mask.resize(N_256_aligned, 0);
      for (int n = N; n < N_256_aligned; n++) {
        mask[n] = neg_inf_ui16;
      }
    } else {
      // sd1.5 unet mha
      mask.resize(N_padded, 0);
      for (int n = N; n < N_padded; n++) {
        mask[n] = neg_inf_ui16;
      }
    }
  }
  std::vector<Tensor> const_tensors;
  const_tensors.push_back({mask.data(), {}, "bfloat16"});
  sd_mha.initialize_const_params(const_tensors);
  std::vector<Tensor> input_Tensor;
  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), {}, c_dtype}};
  if (test_with_golden) {
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/sd_mha/";
    std::string shape_key = std::to_string(B) + "_" + std::to_string(M) + "_" +
                            std::to_string(K) + "_" + std::to_string(N);
    std::vector<uint32_t> qkv_aie =
        read_file(test_golden_root_dir + shape_key + "_ifm32.txt");
    input_Tensor = {{qkv_aie.data(), {}, "bfloat16"}};
#ifdef UNIT_TEST_PERF
    LOG_THIS("B = " << B << ", M = " << M << ", K = " << K << ", N = " << N);
    PROFILE_THIS(sd_mha.execute(input_Tensor, output_Tensor));
#else
    sd_mha.execute(input_Tensor, output_Tensor);
#endif
    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref.txt";
    std::vector<uint32_t> output_golden = read_file(output_golden_path);
    std::vector<OuT> bf16_output_golden(aie_out.size());
    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));

    err_count = sd_mha_check_result<OuT>(
        bf16_output_golden, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  } else {
    // gen rand
    std::vector<float> raw_q(B * M * K, 0);
    initialize_random_float(raw_q, 1, -1);
    auto bf16_q = float_2_bf16_vec(raw_q);

    std::vector<float> raw_k(B * N * K, 0);
    initialize_random_float(raw_k, 1, -1);
    auto bf16_k = float_2_bf16_vec(raw_k);

    std::vector<float> raw_v(B * N * K, 0);
    initialize_random_float(raw_v, 1, -1);
    auto bf16_v = float_2_bf16_vec(raw_v);

    auto v_size = K == 1536 ? out_bo_bf16_cnt : bf16_v.size();

    // combine qkv
    std::vector<uint16_t> bf16_qkv(bf16_q.size() + bf16_k.size() + v_size);
    std::copy(bf16_q.begin(), bf16_q.end(), bf16_qkv.begin());
    std::copy(bf16_k.begin(), bf16_k.end(), bf16_qkv.begin() + bf16_q.size());
    std::copy(bf16_v.begin(), bf16_v.end(),
              bf16_qkv.begin() + bf16_q.size() + bf16_k.size());

    input_Tensor = {{bf16_qkv.data(), {}, "bfloat16"}};

#ifdef UNIT_TEST_PERF
    LOG_THIS("B = " << B << ", M = " << M << ", K = " << K << ", N = " << N);
    PROFILE_THIS(sd_mha.execute(input_Tensor, output_Tensor));
#else
    sd_mha.execute(input_Tensor, output_Tensor);
#endif

    std::vector<OuT> bf16_torch_out(B * M * K);
    auto torch_q = torch::from_blob(raw_q.data(), {B, M, K}, torch::kFloat);
    auto torch_k = torch::from_blob(raw_k.data(), {B, N, K}, torch::kFloat);
    auto torch_v = torch::from_blob(raw_v.data(), {B, N, K}, torch::kFloat);
    auto attn_output = scaled_dot_product_attention(torch_q, torch_k, torch_v,
                                                    static_cast<int>(heads));

    float *c_golden = attn_output.data_ptr<float>();
    uint32_t *c_golden_u = reinterpret_cast<uint32_t *>(c_golden);
    for (int b = 0; b < B; b++) {
      for (int m = 0; m < M; m++) {
        for (int n = 0; n < K; n++) {
          int idx = n + m * K + b * M * K;
          bf16_torch_out[idx] = c_golden_u[idx] >> 16;
        }
      }
    }
    err_count = sd_mha_check_result<OuT>(
        bf16_torch_out, aie_out, error_tolerance, pixel_L2_norm_tolerance);
  }
  return err_count;
}

// MHA
// Random test
// Unet
TEST(SD_MHA_Test, Random_KernelUnetlayer1) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 1024, 640, 1024, 8, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Random_KernelUnetlayer2) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 1024, 640, 77, 8, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Random_KernelUnetlayer3) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 256, 1280, 256, 8, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Random_KernelUnetlayer4) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 256, 1280, 77, 8, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Random_KernelUnetlayer5) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 4096, 320, 4096, 8, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Random_KernelUnetlayer6) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 4096, 320, 77, 8, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Random_KernelUnetlayer7) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 64, 1280, 64, 8, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Random_KernelUnetlayer8) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 64, 1280, 77, 8, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SD3 mmdit aka SD 3.0 MHA_mmdit layer 1
TEST(SD_MHA_Test, Random_KernelMMDITLayer1) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 1178, 1536, 1178, 24, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SD 3.0 MHA_mmdit layer 2
TEST(SD_MHA_Test, Random_KernelMMDITLayer2) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 4250, 1536, 4250, 24, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Vae
TEST(SD_MHA_Test, Random_KernelVaelayer1) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      1, 4096, 512, 4096, 1, "bfloat16", "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Golden test
// Unet
TEST(SD_MHA_Test, Golden_KernelUnetlayer1) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 1024, 640, 1024, 8, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Golden_KernelUnetlayer2) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 1024, 640, 77, 8, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Golden_KernelUnetlayer3) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 256, 1280, 256, 8, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Golden_KernelUnetlayer4) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 256, 1280, 77, 8, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Golden_KernelUnetlayer5) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 4096, 320, 4096, 8, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Golden_KernelUnetlayer6) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 4096, 320, 77, 8, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Golden_KernelUnetlayer7) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 64, 1280, 64, 8, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(SD_MHA_Test, Golden_KernelUnetlayer8) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 64, 1280, 77, 8, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SD3 mmdit aka SD 3.0 MHA_mmdit layer 1
TEST(SD_MHA_Test, Golden_KernelMMDITLayer1) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 1178, 1536, 1178, 24, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// SD 3.0 MHA_mmdit layer 2
TEST(SD_MHA_Test, Golden_KernelMMDITLayer2) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      2, 4250, 1536, 4250, 24, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Vae
TEST(SD_MHA_Test, Golden_KernelVaelayer1) {
  int err_count = test_sd_mha<uint16_t, uint16_t, uint16_t>(
      1, 4096, 512, 4096, 1, "bfloat16", "bfloat16", "bfloat16", 0.01f, true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
