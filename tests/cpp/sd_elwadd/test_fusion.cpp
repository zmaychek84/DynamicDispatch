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

static void initialize_random_float(std::vector<float> &vec, int max, int min) {
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

template <typename T>
void dump_data_as_uint32_to_file(const std::vector<T> &data,
                                 const std::string &output_file) {
  static_assert(sizeof(T) <= sizeof(uint32_t),
                "Data type is larger than uint32_t!");

  size_t num_uint32_elements = (data.size() * sizeof(T)) / sizeof(uint32_t);
  const uint32_t *data_as_uint32 =
      reinterpret_cast<const uint32_t *>(data.data());

  // Open the output file for writing
  std::ofstream ofm_ofs(output_file, std::ofstream::out | std::ofstream::trunc);
  if (!ofm_ofs.is_open()) {
    std::cerr << "Failed to open file " << output_file << " for writing!"
              << std::endl;
    return;
  }

  std::cout << "Opened file " << output_file << " for writing OFM!"
            << std::endl;

  for (size_t i = 0; i < num_uint32_elements; i++) {
    ofm_ofs << std::setw(8) << std::hex << std::setfill('0')
            << data_as_uint32[i] << std::endl;
  }
  ofm_ofs.close();
  std::cout << "Data has been successfully written to " << output_file
            << std::endl;
}

int get_shape_ele_num(const std::vector<size_t> &shape) {
  int total_num = 1;
  for (int dim : shape) {
    total_num *= dim;
  }
  return total_num;
}

int test_sd_elwadd(const std::string &meta_json, bool use_wts) {
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
  std::vector<size_t> a_shape = {2, 8, 8, 1280};
  std::vector<size_t> b_shape = {2, 8, 8, 1280};
  if (use_wts) {
    a_shape = {2, 4096, 320};
    b_shape = {320};
  }
  // only test ifm1 size == ifm2 size
  std::vector<float> a_aie_float(get_shape_ele_num(a_shape));
  initialize_random_float(a_aie_float, 2, -2);
  float2bf16_vec(a_aie_float);
  std::vector<uint16_t> a_aie_bf16(a_aie_float.size(), 0);
  union {
    uint32_t u;
    float f;
  } uf;
  for (int i = 0; i < a_aie_float.size(); ++i) {
    uf.f = a_aie_float[i];
    a_aie_bf16[i] = uf.u >> 16;
  }

  std::vector<float> b_aie_float(get_shape_ele_num(b_shape));
  initialize_random_float(b_aie_float, 2, -2);
  float2bf16_vec(b_aie_float);
  std::vector<uint16_t> b_aie_bf16(b_aie_float.size(), 0);
  for (int i = 0; i < b_aie_float.size(); ++i) {
    uf.f = b_aie_bf16[i];
    b_aie_bf16[i] = uf.u >> 16;
  }

  std::vector<Tensor> input_Tensors;
  input_Tensors.push_back({a_aie_bf16.data(), a_shape, "bfloat16"});
  if (!use_wts) {
    input_Tensors.push_back({b_aie_bf16.data(), b_shape, "bfloat16"});
  }
  std::vector<size_t> aie_out_shape = a_shape;
  std::vector<uint16_t> aie_out(get_shape_ele_num(aie_out_shape));
  std::vector<Tensor> output_Tensors;
  output_Tensors.push_back({aie_out.data(), aie_out_shape, "bfloat16"});
  rt.execute(input_Tensors, output_Tensors);

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage : test_sd_elwadd.exe <meta.json> use_wts" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    std::string meta_json = std::string(argv[1]);
    bool use_wts = std::stoi(argv[2]);
    int err_count = 0;
    err_count = test_sd_elwadd(meta_json, use_wts);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
