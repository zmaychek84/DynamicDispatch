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

#include "ops/ops_common/matmul_matrix.hpp"
#include "test_common.hpp"
#include <algorithm>
#include <cfenv>
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <random>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

using namespace matmul_matrix;

static float bfloat16_2_float(uint16_t x) {
  float y = 0.0;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *dst = (uint8_t *)&y;
  dst[2] = src[0];
  dst[3] = src[1];
  return y;
}

static void dumpVectorToTxt(const std::vector<float> &data,
                            const std::vector<size_t> &c_shape,
                            const std::string &filename) {
  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }
  int N = c_shape[0];
  int H = c_shape[1];
  int W = c_shape[2];
  int C = c_shape[3];
  int index = 0;

  for (int n = 0; n < N; ++n) {
    outFile << "Batch " << n << ":\n";
    for (int h = 0; h < H; ++h) {
      outFile << "  Height " << h << ":\n";
      for (int w = 0; w < W; ++w) {
        outFile << "    Width " << w << ": ";
        for (int c = 0; c < C; ++c) {
          outFile << std::setw(8) << data[index++] << " ";
        }
        outFile << "\n";
      }
      outFile << "\n";
    }
    outFile << "\n";
  }

  outFile.close();
  std::cout << "Data dumped to " << filename << std::endl;
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

static void initialize_random_float(std::vector<float> &vec, int max, int min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = dis(gen);
  }
}

static int get_shape_ele_num(const std::vector<size_t> &shape) {
  int total_num = 1;
  for (int dim : shape) {
    total_num *= dim;
  }
  return total_num;
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

int test_sd_resize(const std::string &meta_json) {
  auto meta = OpsFusion::load_meta_json(meta_json);
  OpsFusion::FusionRuntime rt_cmp;
  OpsFusion::DDConfig cfg;
  std::string xclbin_fname =
      Utils::get_env_var("DD_ROOT") + "\\xclbin\\stx\\SD3_VAE_2x4x4.xclbin";

  auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_fname);
  cfg.xclbin_content = &xclbin_content;
  cfg.model_name = "SD30";
  rt_cmp.compile(meta, "", cfg);
  rt_cmp.save_state("dd_metastate");
  std::cerr << "Compiled" << std::endl;

  OpsFusion::FusionRuntime rt(xclbin_fname, xclbin_content);
  rt.load_state("dd_metastate");
  rt.init(meta, "", cfg);
  // const std::vector<size_t> a_shape = {2, 8, 8, 1280};
  // const std::vector<size_t> c_shape = {2, 16, 16, 1280};
  const std::vector<size_t> a_shape = {1, 256, 256, 256};
  const std::vector<size_t> c_shape = {1, 512, 512, 256};
  std::vector<float> a_aie_float(get_shape_ele_num(a_shape));
  initialize_random_float(a_aie_float, 2, -2);
  // dumpVectorToTxt(a_aie_float, a_shape, "resize_fusion_input.txt");
  std::vector<uint16_t> a_aie_bf16 = float_2_bf16_vec(a_aie_float);

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{a_aie_bf16.data(), a_shape, "bfloat16"}};

  std::vector<uint16_t> aie_out(get_shape_ele_num(c_shape));
  std::vector<float> float_aie_out(get_shape_ele_num(c_shape));
  std::vector<Tensor> output_Tensors;
  output_Tensors.push_back({aie_out.data(), c_shape, "bfloat16"});
  rt.execute(input_Tensors, output_Tensors);

  for (int i = 0; i < float_aie_out.size(); ++i) {
    float_aie_out[i] = bfloat16_2_float(aie_out[i]);
  }
  // dumpVectorToTxt(float_aie_out, c_shape, "resize_fusion.txt");

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_sd_resize.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_sd_resize(meta_json);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
