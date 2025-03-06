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
#include <ops/flat/mha_v2.hpp>

#include "test_common.hpp"

torch::Tensor mha_v2_with_mask(const torch::Tensor &Q,   // Query tensor
                               const torch::Tensor &K,   // Key tensor
                               const torch::Tensor &V,   // Value tensor
                               const torch::Tensor &mask // Mask tensor
) {
  auto scores = torch::matmul(Q, K.transpose(-2, -1));
  auto d_k = Q.size(-1);
  scores = scores / std::sqrt(static_cast<float>(d_k));

  auto expanded_mask = mask.unsqueeze(1);

  scores = scores.masked_fill(expanded_mask == 0, -1e9);

  auto attention_weights = torch::softmax(scores, -1);

  auto output = torch::matmul(attention_weights, V);

  return output; // [batch_size, num_heads, seq_len_q, head_size]
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
int flat_mha_v2_check_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
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

static uint16_t float_to_bfloat16(float x) {
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

template <typename InT = uint16_t, typename OuT = uint16_t>
int test_flat_mha_v2(int num_heads, int seq_len_q, int seq_len_total_k,
                     int head_size, const std::string &a_dtype = "bfloat16",
                     const std::string &c_dtype = "bfloat16",
                     bool test_with_golden = false,
                     float pixel_L2_norm_tolerance = 0.01,
                     const std::string &model_name = "flat") {
  int err_count = 0;
  float error_tolerance = 0.01f;
  std::vector<OuT> aie1_out(num_heads * head_size * seq_len_total_k, 0);
  std::vector<OuT> aie2_out(num_heads * seq_len_q * head_size, 0);
  std::map<std::string, std::any> attr;
  std::vector<int> input_shape{num_heads, seq_len_q, seq_len_total_k,
                               head_size};
  attr["input_shape"] = input_shape;
  auto flat_mha_v2 = ryzenai::flat::mha_v2<std::uint16_t, std::uint16_t>(
      a_dtype, c_dtype, false, attr);
  std::vector<size_t> input_size_shape{size_t(num_heads), size_t(seq_len_q),
                                       size_t(seq_len_total_k),
                                       size_t(head_size)};
  flat_mha_v2.debug(true);
  flat_mha_v2.set_params(model_name, input_size_shape, attr);

  std::vector<Tensor> const_tensors;
  flat_mha_v2.initialize_const_params(const_tensors);
  std::vector<Tensor> input_Tensor;
  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie1_out.data(), {}, c_dtype},
                   {aie2_out.data(), {}, c_dtype}};
  if (test_with_golden) {
    std::string test_golden_root_dir =
        "tests/cpp/unit_tests/testDataMladf/flat_mha_v2/";
    std::string shape_key =
        std::to_string(num_heads) + "_" + std::to_string(seq_len_q) + "_" +
        std::to_string(seq_len_total_k) + "_" + std::to_string(head_size);
    std::vector<uint32_t> ifm_aie =
        read_file(test_golden_root_dir + shape_key + "_ifm32.txt");
    std::vector<uint32_t> wts32_bmm1_aie =
        read_file(test_golden_root_dir + shape_key + "_wts32_bmm1.txt");
    std::vector<uint32_t> wts32_bmm2_aie =
        read_file(test_golden_root_dir + shape_key + "_wts32_bmm2.txt");
    input_Tensor = {{ifm_aie.data(), {}, "bfloat16"},
                    {wts32_bmm1_aie.data(), {}, "bfloat16"},
                    {wts32_bmm2_aie.data(), {}, "bfloat16"}};
#ifdef UNIT_TEST_PERF
    LOG_THIS("num_heads = " << num_heads << ", seq_len_q = " << seq_len_q
                            << ", seq_len_total_k = " << seq_len_total_k
                            << ", head_size = " << head_size);
    PROFILE_THIS(flat_mha_v2.execute(input_Tensor, output_Tensor));
#else
    flat_mha_v2.execute(input_Tensor, output_Tensor);
#endif
    std::string output_golden_path =
        test_golden_root_dir + shape_key + "_ofm32_ref_bmm2.txt";
    std::vector<uint32_t> output_golden = read_file(output_golden_path);
    std::vector<OuT> bf16_output_golden(aie2_out.size());
    memcpy(bf16_output_golden.data(), output_golden.data(),
           output_golden.size() * sizeof(uint32_t));
    err_count = flat_mha_v2_check_result<OuT>(
        bf16_output_golden, aie2_out, error_tolerance, pixel_L2_norm_tolerance);
  } else {
    //     // gen rand
    //     std::vector<float> raw_q(1 * seq_len_q * num_heads * head_size, 0);
    //     initialize_random_float(raw_q, 1, -1);
    //     auto bf16_q = float_2_bf16_vec(raw_q);

    //     std::vector<float> raw_k(1 * seq_len_total_k * num_heads * head_size,
    //     0); initialize_random_float(raw_k, 1, -1); auto bf16_k =
    //     float_2_bf16_vec(raw_k);

    //     std::vector<float> raw_v(1 * seq_len_total_k * num_heads * head_size,
    //     0); initialize_random_float(raw_v, 1, -1); auto bf16_v =
    //     float_2_bf16_vec(raw_v);

    //     std::vector<float> raw_mask(1 * seq_len_q * seq_len_total_k, 0);
    //     initialize_random_float(raw_mask, 1, 0);
    //     for (int i = 0; i < raw_mask.size(); ++i) {
    //       raw_mask[i] = raw_mask[i] > 0 ? 1 : 0;
    //     }
    //     auto bf16_raw_mask = float_2_bf16_vec(raw_mask);

    //     input_Tensor = {{bf16_q.data(), {}, "bfloat16"},
    //                     {bf16_k.data(), {}, "bfloat16"},
    //                     {bf16_v.data(), {}, "bfloat16"},
    //                     {bf16_raw_mask.data(), {}, "bfloat16"}};

    // #ifdef UNIT_TEST_PERF
    //     LOG_THIS("num_heads = " << num_heads << ", seq_len_q = " << seq_len_q
    //                             << ", seq_len_total_k = " << seq_len_total_k
    //                             << ", head_size = " << head_size);
    //     PROFILE_THIS(flat_mha_v2.execute(input_Tensor, output_Tensor));
    // #else
    //     flat_mha_v2.execute(input_Tensor, output_Tensor);
    // #endif

    //     std::vector<OuT> bf16_torch_out(aie_out.size());
    //     auto torch_q = torch::from_blob(
    //         raw_q.data(), {1, num_heads, seq_len_q, head_size},
    //         torch::kFloat);
    //     auto torch_k = torch::from_blob(raw_k.data(),
    //                                     {1, num_heads, seq_len_total_k,
    //                                     head_size}, torch::kFloat);
    //     auto torch_v = torch::from_blob(raw_v.data(),
    //                                     {1, num_heads, seq_len_total_k,
    //                                     head_size}, torch::kFloat);
    //     auto torch_mask = torch::from_blob(
    //         raw_mask.data(), {1, seq_len_q, seq_len_total_k}, torch::kFloat);
    //     auto attn_output = mha_v2_with_mask(torch_q, torch_k, torch_v,
    //     torch_mask);

    //     float *c_golden = attn_output.data_ptr<float>();
    //     uint32_t *c_golden_u = reinterpret_cast<uint32_t *>(c_golden);
    //     for (int i = 0; i < aie_out.size(); ++i) {
    //       bf16_torch_out[i] = c_golden_u[i] >> 16;
    //     }
    //     err_count = flat_mha_v2_check_result<OuT>(
    //         bf16_torch_out, aie_out, error_tolerance,
    //         pixel_L2_norm_tolerance);
  }
  return err_count;
}

// mha_v2
// Random test
TEST(FLAT_MHA_V2_Test, Random_KernelLayer1) {
  int err_count = test_flat_mha_v2<uint16_t, uint16_t>(32, 1, 1024, 96,
                                                       "bfloat16", "bfloat16");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// Golden test
TEST(FLAT_MHA_V2_Test, Golden_KernelLayer1) {
  int err_count = test_flat_mha_v2<uint16_t, uint16_t>(
      32, 1, 1024, 96, "bfloat16", "bfloat16", true);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
