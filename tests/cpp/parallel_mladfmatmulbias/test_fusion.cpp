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
#include <iostream>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "ops/ops_common/mladf_matmul_matrix.hpp"
#include <ops/mladfmatmulbias/mladfmatmulbias.hpp>

#include "mladfmatmulbias_helpers.hpp"

using namespace std;

template <typename InT = int16_t, typename WgT = int8_t, typename OuT = int16_t>
static int test_parallel_matmul(const string &meta_json, size_t M, size_t K,
                                size_t N, bool debug = false,
                                const string &a_dtype = "bfloat16",
                                const string &b_dtype = "uint4",
                                const string &c_dtype = "bfloat16") {

  vector<size_t> a_shape = {1, M, K};
  vector<size_t> b_shape = {1, K, N};
  vector<size_t> c_shape = {1, M, N};

  srand(42);

  mladfmatmulbias_helpers::MladfMatMulBias matmul0(M, K, N, 128);
  mladfmatmulbias_helpers::MladfMatMulBias matmul1(M, K, N, 128);

  matmul0.InitializeRandom();
  matmul1.InitializeRandom();

  matmul1.a = matmul0.a; // Use the same input

  {
    std::string matmul_dir = "test_parallel_mladfmatmul",
                weights_name = "0.const", bias_name = "1.const",
                scales_name = "2.const", zeros_name = "3.const";

    matmul0.WriteParams(matmul_dir, weights_name, bias_name, scales_name,
                        zeros_name);
  }
  {
    std::string matmul_dir = "test_parallel_mladfmatmul",
                weights_name = "4.const", bias_name = "5.const",
                scales_name = "6.const", zeros_name = "7.const";

    matmul1.WriteParams(matmul_dir, weights_name, bias_name, scales_name,
                        zeros_name);
  }
  matmul0.ForwardCPU();
  matmul1.ForwardCPU();

  auto &a = matmul0.a;
  vector<vector<float>> c_golden = {matmul0.c_golden, matmul1.c_golden};
  vector<vector<OuT>> c(2, vector<OuT>(M * N, -1));

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
  vector<Tensor> input_Tensors;
  input_Tensors = {{a.data(), a_shape, a_dtype}};

  vector<Tensor> output_Tensors;
  output_Tensors = {{c[0].data(), c_shape, c_dtype},
                    {c[1].data(), c_shape, c_dtype}};

  rt.execute(input_Tensors, output_Tensors);

  float const EPSILON = 512.0;
  int err_count = 0;

  err_count += dd::count_errors_floatvsbfloat16(c_golden[0], c[0],
                                                {1, c[0].size()}, EPSILON);
  err_count += dd::count_errors_floatvsbfloat16(c_golden[1], c[1],
                                                {1, c[1].size()}, EPSILON);
  return err_count;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage : test_parallel_mladfmatmul.exe <meta.json>" << endl;
    return EXIT_FAILURE;
  }

  size_t M = 1;
  size_t K = 4096;
  size_t N = 11008;
  try {
    string meta_json = string(argv[1]);

    int err_count = 0;
    err_count = test_parallel_matmul<uint16_t, uint16_t, uint16_t>(meta_json, M,
                                                                   K, N, false);
    if (err_count > 0) {
      cout << "parallel mladfmatmul test failed with err_count = " << err_count
           << endl;
      return EXIT_FAILURE;
    }
  } catch (exception &e) {
    cout << e.what() << endl;
    return EXIT_FAILURE;
  }

  cout << "Finished Successfully" << endl;
  return EXIT_SUCCESS;
}
