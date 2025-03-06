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
// #include "enable_perf.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <op_fuser/fusion_rt.hpp>
#include <ops/bmm/bmm.hpp>
#include <utils/meta_utils.hpp>
#include <utils/utils.hpp>

#include "bmm_helpers.hpp"
#include "test_common.hpp"

using namespace ryzenai;
using namespace matmul_matrix;

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_bmm(const std::string &meta_json, int M, int K, int N, int B = 32,
             bool debug = false, const std::string &a_dtype = "bfloat16",
             const std::string &b_dtype = "bfloat16",
             const std::string &c_dtype = "bfloat16", bool trans = false) {
  int BM = M * B;
  int err_count = 0;

  size_t BMs = static_cast<size_t>(BM);
  size_t Bs = static_cast<size_t>(B);
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Bs, Ms, Ks};
  std::vector<size_t> b_shape = {Bs, Ks, Ns};
  std::vector<size_t> aie_out_shape = {Bs, Ms, Ns};
  std::vector<InT> a(BM * K);
  std::vector<WgT> b(B * K * N);
  std::vector<uint16_t> cpu_out(BM * N);
  std::vector<OuT> aie_out(BM * N, garbage_value);
  RowMajorMatrix<InT> X(BM, K, a.data());
  RowMajorMatrix<WgT> *W;
  W = new RowMajorMatrix<WgT>(B * K, N, b.data());

  RowMajorMatrix<uint16_t> cpu_Y(BM, N, cpu_out.data());
  RowMajorMatrix<OuT> aie_Y(BM, N, aie_out.data());

  srand(0xABCD);
  dd::initialize_random_bfloat16(a, 1.5);
  dd::initialize_random_bfloat16(b, 1.5);

  const auto &xclbin_fname =
      Utils::get_env_var("DD_ROOT") +
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
  input_Tensors = {{a.data(), a_shape, a_dtype}, {b.data(), b_shape, b_dtype}};

  std::vector<Tensor> output_Tensors;
  output_Tensors = {{aie_out.data(), aie_out_shape, c_dtype}};

  rt.execute(input_Tensors, output_Tensors);

  RowMajorMatrix<InT> XX(B * M, K, a.data());
  RowMajorMatrix<WgT> WW(B * K, N, b.data());
  RowMajorMatrix<uint16_t> cpu_YY(M * B, N, cpu_out.data());
  bmm_helpers::cpu_bmm2<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
                        RowMajorMatrix<OuT>>(XX, WW, cpu_YY, B);
  std::cout << std::endl;
  err_count =
      check_add_result_bfloat16<OuT>(cpu_out, aie_out, aie_out_shape, 0.75);
  return err_count;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage : test_bmm2.exe <meta.json>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << std::fixed;
  size_t M = 2048;
  size_t K = 2048; // switch N K
  size_t N = 128;
  size_t B = 32;
  bool trans = false;

  try {
    std::string meta_json = std::string(argv[1]);
    int err_count = 0;
    err_count = test_bmm<uint16_t, uint16_t, uint16_t>(
        meta_json, M, K, N, B, false, "bfloat16", "bfloat16", "bfloat16",
        trans);
    if (err_count > 0) {
      std::cout << "BMM2 test failed with err_count = " << err_count
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
