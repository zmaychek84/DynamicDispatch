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
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/matmul_cpu/matmul_cpu.hpp>

#include "test_common.hpp"

using namespace matmul_matrix;
using namespace std;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
static int test_matmul_cpu(const std::string &meta_json, size_t M, size_t K,
                           size_t N, bool debug = false,
                           const std::string &a_dtype = "int16",
                           const std::string &b_dtype = "int8",
                           const std::string &c_dtype = "int32") {
  int err_count = 0;
  int Msubv_act = 0;
  if (a_dtype == "int16" || a_dtype == "uint16") {
    Msubv_act = 32;
  } else if (a_dtype == "int8" || a_dtype == "uint8") {
    Msubv_act = 64;
  } else {
    throw std::invalid_argument("a_dtype is not supported");
  }
  int N_w = N;
  if (N_w < Nsubv * 2) {
    N_w = Nsubv * 2; // This is the miminum N
  }
  std::vector<size_t> a_shape = {1, M, K};
  std::vector<size_t> b_shape = {K, N};
  std::vector<size_t> qdq_shape = {N};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};
  std::vector<size_t> aie_out_shape = {1, M, N};

  std::vector<InT> a(M * K);
  std::vector<WgT> b(K * N);
  std::vector<OuT> aie_out(M * N, garbage_value);

  initialize_random<InT>(a, M * N, 32, 0);

  std::string xclbin_fname;
  if (a_dtype == "uint16") {
    xclbin_fname =
        Utils::get_env_var("DD_ROOT") + mxpzi_A16W8_QDQ_XCLBIN_REL_PATH;
  } else {
    xclbin_fname =
        Utils::get_env_var("DD_ROOT") + mdsqr_A8W8_QDQ_XCLBIN_REL_PATH;
  }

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

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

  rt.execute(input_Tensor, output_Tensor);
  RowMajorMatrix<OuT> ref(M, N, a.data());
  RowMajorMatrix<OuT> matmul_cpu_out(M, N, aie_out.data());
  err_count = check_result(matmul_cpu_out, ref);

  return err_count;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : ops_fusion.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::fixed;
  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;

    err_count = test_matmul_cpu<uint16_t, uint8_t, uint16_t>(
        meta_json, 128, 768, 768, false, "uint16", "uint8", "uint16");

    std::cout << "err count" << err_count << std::endl;
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
