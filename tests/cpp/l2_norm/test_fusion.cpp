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

#include "ops/l2_norm/l2_norm.hpp"
#include "ops/ops_common/help_file.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <op_fuser/fusion_rt.hpp>
#include <utils/meta_utils.hpp>

#include "ops/ops_common/matmul_matrix.hpp"

#include "test_common.hpp"
using namespace matmul_matrix;

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_l2_norm(const std::string &meta_json, int M, int N, bool debug = false,
                 const std::string &a_dtype = "int16",
                 const std::string &b_dtype = "int16",
                 const std::string &c_dtype = "int16",
                 const std::string &model_name = "mdsqr") {

  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ns};
  std::vector<size_t> aie_out_shape = {Ms, Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> a(M * N);
  std::vector<OutT> cpu_out(M * N);
  std::vector<OutT> cpu_q_out(M * N); // not used during model data
  std::vector<OutT> aie_out(M * N);
  std::vector<int32_t> qdq_params(QDQparam_size);

  RowMajorMatrix<OutT> cpu_Y(M, N, cpu_out.data());
  RowMajorMatrix<OutT> cpu_q_Y(M, N,
                               cpu_q_out.data()); // not used during model data
  RowMajorMatrix<OutT> aie_Y(M, N, aie_out.data());
  RowMajorMatrix<InT> inputMat(M, N, a.data());

#ifdef RANDOM_DATA
  int32_t is_output_uint16 = 0;

  if (c_dtype == "uint16") {
    is_output_uint16 = 1;
  }

  float sc_float = 0.01;
  int16_t sc_out = 1.0 / sc_float; // bfloat16
  OutT zp_out = 129;

  srand(0xABCD);
  initialize_random_bfloat16(a, M * N, -20, 20);

  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float in_gold = bfloat16_to_float(inputMat.at(r, c));
      cpu_Y.at(r, c) = float_to_bfloat16(silu_golden(in_gold, r, c));
    }
  }
  // quant_bfloat_to_uint16(cpu_Y, sc_out, zp_out, cpu_q_Y);
  quant_bfloat16_to_int16(cpu_Y, cpu_q_Y, sc_out, zp_out);

  qdq_params[0] = zp_out; // for silu
  qdq_params[1] = float_to_bfloat16(sc_out);
  qdq_params[2] = 1; // out_quant_enable
  qdq_params[3] = 0;
  qdq_params[4] = 0;
  qdq_params[5] = 0; // if 1, enalbe de-quant at input

#else
  std::string fld_name;
  fld_name = "//bins//l2_norm//";

  read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "ifm.bin",
                reinterpret_cast<char *>(a.data()));

  read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "wgt.bin",
                reinterpret_cast<char *>(qdq_params.data()));

  read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "ofm.bin",
                reinterpret_cast<char *>(cpu_out.data()));
#endif
  {
    // this mul_dir should be consistent with 'dir_name' in model.py
    std::string mul_dir = "test_l2_norm";
    std::ofstream wts_f(mul_dir + "/0.const", std::ios::out | std::ios::binary);
    confirmOpen(wts_f);
    wts_f.write((char *)qdq_params.data(), qdq_params.size() * sizeof(int32_t));
    wts_f.close();
  }

  std::string xclbin_fname;
  if (a_dtype == "uint16") {
    xclbin_fname =
        Utils::get_env_var("DD_ROOT") + ryzenai::PSU_4x4_A16W8_QDQ_XCLBIN_PATH;
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
  std::vector<Tensor> input_tensor;
  Tensor a_T = {a.data(), a_shape, a_dtype};

  Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_tensor.push_back(a_T);

  std::vector<Tensor> output_tensor;
  output_tensor.push_back(c_T);
  rt.execute(input_tensor, output_tensor);

  err_count = check_result_bfloat(cpu_Y, aie_Y, 0.01);

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

    err_count = test_l2_norm<uint16_t, uint16_t, uint16_t>(
        meta_json, 64, 3072, false, "uint16", "uint16", "uint16", "PSU_LP");
    if (err_count > 1) {
      std::cout << "EltwiseMul Test failed with err count : " << err_count
                << std::endl;
      return EXIT_FAILURE;
    } else {
      std::cout << "EltwiseMul Test Passed with err count : " << err_count
                << std::endl;
    }

  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Finished Successfully" << std::endl;
  return EXIT_SUCCESS;
}
