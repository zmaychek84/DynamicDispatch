// Copyright (c) 2024 Advanced Micro Devices, Inc
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
#include <iostream>
#include <op_fuser/fusion_rt.hpp>

#include "ops/ops_common/lrn_matrix.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/elwmul/elwmul.hpp>

#include "mladfelwmul_helpers.hpp"

// z = x * y
template <typename LhsT = int16_t, typename RhsT = int16_t,
          typename OuT = int16_t>
static int test_elwmul(const std::string &meta_json, size_t M, size_t N,
                       bool debug = false,
                       const std::string &x_dtype = "bfloat16",
                       const std::string &y_dtype = "bfloat16",
                       const std::string &z_dtype = "bfloat16") {

  // Without the prefix 1 in the batch dimension this does not work
  std::vector<size_t> x_shape = {1, M, N};
  std::vector<size_t> y_shape = {1, M, N};
  std::vector<size_t> z_shape = {1, M, N};

  srand(42);

  mladfelwmul_helpers::ElwMul elwmul(M, N);
  elwmul.InitializeRandom();
  elwmul.ForwardCPU();

  auto &x = elwmul.x;
  auto &y = elwmul.y;
  auto &z_golden = elwmul.z_golden;
  std::vector<OuT> z(M * N, garbage_value);

  const auto &xclbin_fname =
      Utils::get_env_var("DD_ROOT") +
      ryzenai::
          LLAMA2_MLADF_2x4x4_V1_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_PATH;
  auto meta = OpsFusion::load_meta_json(meta_json);

  OpsFusion::FusionRuntime rt_cmp;
  OpsFusion::DDConfig cfg;
  auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_fname);
  cfg.xclbin_content = &xclbin_content;
  rt_cmp.compile(meta, "", cfg);
  rt_cmp.save_state("dd_metastate");

  OpsFusion::FusionRuntime rt(xclbin_fname, xclbin_content);
  rt.load_state("dd_metastate");
  rt.init(meta);

  std::vector<Tensor> input_Tensors;
  input_Tensors = {{x.data(), x_shape, x_dtype}, {y.data(), y_shape, y_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{z.data(), z_shape, z_dtype}};

  rt.execute(input_Tensors, output_Tensors);

  float const EPSILON = 0.0;
  return dd::count_errors_floatvsbfloat16(z_golden, z, {1, y.size()}, EPSILON);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_elwmul.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }
  std::runtime_error("HERE");
  std::cout << std::fixed;
  size_t M = 1;
  size_t N = 11008;
  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count =
        test_elwmul<uint16_t, uint16_t, uint16_t>(meta_json, M, N, false);
    if (err_count > 0) {
      std::cout << "Silu test failed with err_count = " << err_count
                << std::endl;
      return EXIT_FAILURE;
    }
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
