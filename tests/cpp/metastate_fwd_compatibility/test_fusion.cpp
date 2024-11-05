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
#include <ops/matmul/matmul.hpp>

#include "test_common.hpp"

using namespace matmul_matrix;
using namespace std;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
static int test_matmul(const std::string &meta_json, size_t M, size_t K,
                       size_t N, bool debug = false,
                       const std::string &a_dtype = "int16",
                       const std::string &b_dtype = "int8",
                       const std::string &c_dtype = "int32",
                       const std::string &meta_path = {}) {
  int err_count = 0;
  int Msubv_act = 64;
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
  std::vector<int64_t> qdq(1 * N); // c0
  std::vector<int32_t> qdq_params(QDQparam_size);
  std::vector<int32_t> cpu_out(M * N_w);
  std::vector<OuT> cpu_out_qdq(M * N_w);
  std::vector<OuT> aie_out(M * N, garbage_value);

  RowMajorMatrix<InT> X(M, K, a.data());
  RowMajorMatrix<int32_t> cpu_Y(M, N_w, cpu_out.data());
  RowMajorMatrix<OuT> cpu_Y_qdq(M, N_w, cpu_out_qdq.data());
  RowMajorMatrix<OuT> aie_Y(M, N, aie_out.data());

  srand(0xABCD);
  init_random(X, 0, 32);
  initialize_random<WgT>(b, K * N, 32, 0);
  initialize_random<int64_t>(qdq, 1 * N, 32, 0);

  uint32_t C1 = 0;
  uint32_t C2 = 10;
  uint8_t SQb = 0;
  uint8_t Sout = 13;
  uint8_t Stdm = 0;
  int64_t *C0_vec = (int64_t *)qdq.data();
  int64_t c0 = 0;
  *(int64_t *)(&qdq_params[qdq_c0_idx]) = c0; // qdq_params[0] = c0;
  qdq_params[qdq_c1_idx] = C1;
  qdq_params[qdq_c2_idx] = C2;
  qdq_params[qdq_c3_idx] = 0;
  // qdq_params[qdq_Mv_idx] = Msubv;
  // qdq_params[qdq_Nv_idx] = Nsubv;
  qdq_params[qdq_SQb_idx] = SQb;
  qdq_params[qdq_Sout_idx] = Sout;
  qdq_params[qdq_Stdm_idx] = Stdm;

  {
    std::string matmul_dir = meta_path;
    std::ofstream wts_f(matmul_dir + "/0.const",
                        std::ios::out | std::ios::binary);
    std::ofstream qdq_f(matmul_dir + "/1.const",
                        std::ios::out | std::ios::binary);
    std::ofstream qdq_p_f(matmul_dir + "/2.const",
                          std::ios::out | std::ios::binary);
    confirmOpen(wts_f);
    confirmOpen(qdq_f);
    confirmOpen(qdq_p_f);
    wts_f.write((char *)b.data(), b.size() * sizeof(WgT));
    qdq_f.write((char *)qdq.data(), qdq.size() * sizeof(int64_t));
    qdq_p_f.write((char *)qdq_params.data(),
                  qdq_params.size() * sizeof(int32_t));
  }

  RowMajorMatrix<WgT> W(K, N_w, b.data());
  cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>, RowMajorMatrix<int32_t>>(
      X, W, cpu_Y, "int32");
  qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>, RowMajorMatrix<OuT>>(
      X, cpu_Y, C2, C1, C0_vec, SQb, Sout, cpu_Y_qdq, "uint8");

  std::string xclbin_fname =
      Utils::get_env_var("DD_ROOT") + "\\" + mdsqr_A8W8_QDQ_XCLBIN_REL_PATH;

  auto meta = OpsFusion::load_meta_json(meta_json);
  auto xclbin_content = OpsFusion::read_bin_file<char>(xclbin_fname);
  OpsFusion::FusionRuntime rt(xclbin_fname, xclbin_content);
  rt.load_state(meta_path + "dd_metastate");
  rt.init(meta);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

  rt.execute(input_Tensor, output_Tensor);

  err_count = check_result(cpu_Y_qdq, aie_Y);

  return err_count;
}

int main(int argc, char *argv[]) {

  std::string meta_path =
      Utils::get_env_var("DD_ROOT") + "/tests/cpp/metastate_fwd_compatibility/";

  std::cout << std::fixed;
  try {
    std::string meta_json = meta_path + "model_matmul1_meta_a8w8.json";
    int err_count = 0;
    err_count = test_matmul<uint8_t, uint8_t, uint8_t>(
        meta_json, 512, 768, 1152, false, "uint8", "uint8", "uint8", meta_path);
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
