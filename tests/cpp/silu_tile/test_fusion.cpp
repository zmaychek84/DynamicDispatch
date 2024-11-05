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
#include <ops/silu/silu.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "mladfsilu_helpers.hpp"

// y = x * sigmoid(x)
template <typename InT = uint16_t, typename OuT = uint16_t>
static int test_silu(const std::string &meta_json, size_t M, size_t N,
                     bool debug = false,
                     const std::string &x_dtype = "bfloat16",
                     const std::string &y_dtype = "bfloat16") {

  std::vector<size_t> x_shape = {1, M, N};
  std::vector<size_t> y_shape = {1, M, N};

  srand(42);

  mladfsilu_helpers::SiLU silu(M, N);
  silu.InitializeRandom();
  silu.ForwardCPU();

  auto &x = silu.x;
  auto &y_golden = silu.y_golden;
  std::vector<OuT> y(M * N, 0xCD);

  const std::string xclbin_fname =
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
  input_Tensors = {{x.data(), x_shape, x_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{y.data(), y_shape, y_dtype}};

  rt.execute(input_Tensors, output_Tensors);

  return dd::count_errors_floatvsbfloat16(
      y_golden, y, y_shape, ryzenai::silu<uint16_t, uint16_t>::EPSILON);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_silu.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::fixed;
  size_t M = 4096;
  size_t N = 11008;
  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_silu(meta_json, M, N, false);
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
