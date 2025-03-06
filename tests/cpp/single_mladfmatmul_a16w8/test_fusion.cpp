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

#include "ops/ops_common/matmul_a16w8_mladf_matrix.hpp"
#include "test_common.hpp"
#include <ops/matmul_a16w8_mladf/matmul_a16w8_mladf.hpp>

using namespace matmul_a16w8_mladf_matrix;
using namespace std;

template <typename InT = uint16_t, typename WgT = uint8_t,
          typename OuT = uint16_t>
static int test_matmul(const std::string &meta_json, size_t M, size_t K,
                       size_t N, bool debug = false,
                       const std::string &a_dtype = "uint16",
                       const std::string &b_dtype = "uint8",
                       const std::string &c_dtype = "uint16") {
  int err_count = 0;

  std::vector<size_t> a_shape = {M, K};
  std::vector<size_t> b_shape = {K, N};
  std::vector<size_t> aie_out_shape = {M, N};

  std::vector<InT> a(M * K);
  std::vector<WgT> b(K * N);
  std::vector<int64_t> qdq_c0(1 * N);
  std::vector<int32_t> qdq_c1c2(QDQparam_size);

  std::vector<OuT> cpu_out(M * N);
  std::vector<OuT> aie_out(M * N, garbage_value);

  RowMajorMatrix<InT> X(M, K, a.data());
  RowMajorMatrix<WgT> W(K, N, b.data());
  RowMajorMatrix<OuT> cpu_Y(M, N, cpu_out.data());
  RowMajorMatrix<OuT> aie_Y(M, N, aie_out.data());

  srand(0xABCD);
  init_random(X, 0, 4);
  init_random(W, 0, 4);
  initialize_random<int64_t>(qdq_c0, 1 * N, 4, 0);
  initialize_random<int32_t>(qdq_c1c2, QDQparam_size, 4, 0);

  int32_t shift_gemm_out = 1;
  int32_t shift_qdq_out = 1;
  std::string matmul_dir = "test_mladfmatmul";
  matmul_dir += "_a16w8";

  std::ofstream wts_f(matmul_dir + "/0.const",
                      std::ios::out | std::ios::binary);
  std::ofstream qdq_c0_f(matmul_dir + "/1.const",
                         std::ios::out | std::ios::binary);
  std::ofstream qdq_c1c2_f(matmul_dir + "/2.const",
                           std::ios::out | std::ios::binary);
  confirmOpen(wts_f);
  confirmOpen(qdq_c0_f);
  confirmOpen(qdq_c1c2_f);
  wts_f.write((char *)b.data(), b.size() * sizeof(WgT));
  qdq_c0_f.write((char *)qdq_c0.data(), qdq_c0.size() * sizeof(int64_t));
  qdq_c1c2_f.write((char *)qdq_c1c2.data(), QDQparam_size * sizeof(int32_t));
  wts_f.close();
  qdq_c0_f.close();
  qdq_c1c2_f.close();
  cpu_qdq_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>, RowMajorMatrix<OuT>>(
      X, W, cpu_Y, qdq_c0, qdq_c1c2[0], qdq_c1c2[1], shift_gemm_out,
      shift_qdq_out, "uint16");

  std::string xclbin_fname =
      Utils::get_env_var("DD_ROOT") + MLADF_2x4x2_GEMM_A16A16_XCLBIN_PATH;

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

  err_count = check_result(cpu_Y, aie_Y);

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
    if (meta_json.find("a16w8") != string::npos) {
      err_count = test_matmul<uint16_t, uint8_t, uint16_t>(
          meta_json, 4096, 512, 512, false, "uint16", "uint8", "uint16");
    }
    if (err_count > 0) {
      std::cout << "Matmul test failed with err_count = " << err_count
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
