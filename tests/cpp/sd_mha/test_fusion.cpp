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
#include <algorithm>
#include <cfenv>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <random>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "test_common.hpp"
double round_half_to_even(double value) {
  // Set rounding mode to "round to nearest, ties to even"
  std::fesetround(FE_TONEAREST);

  // Use nearbyint, which rounds according to the current rounding mode
  return std::nearbyint(value);
}

void aie_srs(std::vector<uint32_t> &input_output) {
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
static void initialize_random_float(std::vector<float> &vec, float max,
                                    float min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

void float2bf16_vec(std::vector<float> &x) {
  std::vector<uint32_t> x_uint32(x.size());
  std::memcpy(x_uint32.data(), x.data(), x.size() * sizeof(float));
  aie_srs(x_uint32);
  for (size_t i = 0; i < x_uint32.size(); ++i) {
    x_uint32[i] = (static_cast<uint16_t>(x_uint32[i]) << 16);
  }
  std::memcpy(x.data(), x_uint32.data(), x.size() * sizeof(float));
}

std::vector<uint16_t> gen_rand_bf16(int size) {
  std::vector<float> aie_float(size, 0);
  initialize_random_float(aie_float, 1, -1);
  float2bf16_vec(aie_float);
  std::vector<uint16_t> aie_bf16(aie_float.size(), 0);
  union {
    uint32_t u;
    float f;
  } uf;
  for (int i = 0; i < aie_float.size(); ++i) {
    uf.f = aie_float[i];
    aie_bf16[i] = uf.u >> 16;
  }
  return aie_bf16;
}

int test_mha(const std::string &meta_json, int B = 2, int M = 1024, int K = 640,
             int N = 1024) {
  auto meta = OpsFusion::load_meta_json(meta_json);
  OpsFusion::FusionRuntime rt_cmp;
  OpsFusion::DDConfig cfg;
  std::string xclbin_fname =
      Utils::get_env_var("DD_ROOT") + "\\xclbin\\stx\\SD15_unet_2x4x4.xclbin";

  auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_fname);
  cfg.xclbin_content = &xclbin_content;
  cfg.model_name = "SD15";
  rt_cmp.compile(meta, "", cfg);
  rt_cmp.save_state("dd_metastate");
  std::cerr << "Compiled" << std::endl;

  OpsFusion::FusionRuntime rt(xclbin_fname, xclbin_content);
  rt.load_state("dd_metastate");
  rt.init(meta, "", cfg);

  auto q_bf16 = gen_rand_bf16(B * M * K);
  auto k_bf16 = gen_rand_bf16(B * N * K);
  auto v_bf16 = gen_rand_bf16(B * N * K);

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{q_bf16.data(), {}, "bfloat16"},
                   {k_bf16.data(), {}, "bfloat16"},
                   {v_bf16.data(), {}, "bfloat16"}};

  std::vector<uint16_t> aie_out(B * M * K, 0);
  std::vector<Tensor> output_Tensors;
  struct Tensor c_T = {
      aie_out.data(), {(size_t)B, (size_t)M, (size_t)K}, "bfloat16"};
  output_Tensors.push_back(c_T);
  rt.execute(input_Tensors, output_Tensors);

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : ops_fusion.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_mha(meta_json);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
