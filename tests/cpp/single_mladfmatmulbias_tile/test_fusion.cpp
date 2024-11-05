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
#include <iostream>

#include <algorithm>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "ops/ops_common/mladf_matmul_matrix.hpp"
#include <ops/mladfmatmulbias/mladfmatmulbias.hpp>
#include <ops/ops_common.hpp>

#include "mladfmatmulbias_helpers.hpp"

template <typename InT = int16_t, typename WgT = int8_t, typename OuT = int16_t>
static int test_matmul(const std::string &meta_json, size_t M, size_t K,
                       size_t N, bool debug = false,
                       const std::string &a_dtype = "bfloat16",
                       const std::string &b_dtype = "uint4",
                       const std::string &c_dtype = "bfloat16") {

  std::vector<size_t> a_shape = {1, M, K};
  std::vector<size_t> b_shape = {1, K, N};
  std::vector<size_t> c_shape = {1, M, N};

  srand(42);

  mladfmatmulbias_helpers::MladfMatMulBias matmul(M, K, N, 128);

  matmul.InitializeRandom();

  std::string matmul_dir = "test_mladfmatmul", weights_name = "0.const",
              bias_name = "1.const", scales_name = "2.const",
              zeros_name = "3.const";

  matmul.WriteParams(matmul_dir, weights_name, bias_name, scales_name,
                     zeros_name);

  auto &a = matmul.a;
  std::vector<OuT> c(M * N, -1);

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
  input_Tensors = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{c.data(), c_shape, c_dtype}};

  rt.execute(input_Tensors, output_Tensors);

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_mladfmatmul.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::fixed;
  size_t M = 4096;
  size_t K = 4096;
  size_t N = 11008;
  try {
    std::string meta_json = std::string(argv[1]);

    int err_count = 0;
    err_count = test_matmul(meta_json, M, K, N, false);
    if (err_count > 0) {
      std::cout << "single_mladfmatmul test failed with err_count = "
                << err_count << std::endl;
      return EXIT_FAILURE;
    }
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
