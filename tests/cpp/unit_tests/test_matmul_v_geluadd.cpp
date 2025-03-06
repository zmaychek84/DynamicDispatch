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

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <string>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/dmacompiler/matmul_v_geluadd/matmul_v_geluadd.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_matmul_v_gelu(int M, int K, int N, bool debug = false,
                       const std::string &a_dtype = "int16",
                       const std::string &b_dtype = "int8",
                       const std::string &c_dtype = "int32",
                       const std::string &model_name = "mdsqr") {
  int err_count = 0;
  int Msubv_act = 0;
  if (a_dtype == "uint16") {
    Msubv_act = 32;
  } else if (a_dtype == "uint8") {
    Msubv_act = 64;
  } else {
    throw std::invalid_argument("a_dtype is not supported");
  }
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ks, Ns};
  std::vector<size_t> qdq_shape = {Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};
  std::vector<size_t> aie_out_shape = {Ms, Ns};

  std::vector<InT> a(M * K);
  std::vector<WgT> b(K * N);
  std::vector<int64_t> qdq(1 * N); // c0
  std::vector<int32_t> qdq_params(QDQparam_size);
  std::vector<int32_t> gelu_qdq_params(QDQparam_size);
  // std::vector<InT> bias(1 * N);
  std::vector<int32_t> cpu_out(M * N, garbage_value);
  std::vector<OuT> cpu_out_qdq(M * N, garbage_value);
  std::vector<uint16_t> gelu_out_golden(M * N, garbage_value);
  std::vector<OuT> gelu_out_golden_quant(M * N, garbage_value);
  std::vector<OuT> aie_out(1 * N, garbage_value);
  std::vector<OuT> ref_out(1 * N, garbage_value);

  RowMajorMatrix<InT> X(M, K, a.data());
  RowMajorMatrix<WgT> W(K, N, b.data());
  // BiasVector<InT, 1, Nsubv> B(N, bias.data());
  RowMajorMatrix<int32_t> cpu_Y(M, N, cpu_out.data());
  RowMajorMatrix<OuT> cpu_Y_qdq(M, N, cpu_out_qdq.data());
  // golden Q(X*W)
  RowMajorMatrix<uint16_t> gelu_out_gold_mat(1, N, gelu_out_golden.data());
  // golden Q(X*W)
  RowMajorMatrix<OuT> gelu_out_gold_quant_mat(1, N, ref_out.data());

  RowMajorMatrix<OuT> aie_Y(1, N, aie_out.data());

  srand(0xABCD);
  init_random(X, 0, 16);
  initialize_random<WgT>(b, K * N, 16, 0);
  initialize_random<int64_t>(qdq, 1 * N, 255, 0);
  // init_random(B, -16, 16);

  int32_t C1 = 0;
  int32_t C2 = 20;
  int64_t *C0_vec = (int64_t *)qdq.data();
  uint8_t SQb = 0;
  uint8_t Sout = 13;
  int64_t c0 = 0;
  uint8_t Stdm = 0;

  int isint16 = 1;

  if (a_dtype == "uint16") {
    srand(0xABCD);
    init_random(X, 0, 32);
    initialize_random<WgT>(b, K * N, 32, 0);
    initialize_random<int64_t>(qdq, 1 * N, 10, -10);
    if (model_name == "m3uec") {
      c0 = 0;
      C1 = -11;
      SQb = 0;
      Sout = 17; // 11;
      Stdm = 2;  // round(log2(K)) - 8;
      C2 = 3;    // 2 << Stdm;
    } else {
      c0 = 0;
      C1 = -11;
      SQb = 0;
      Sout = 11;
      Stdm = round(log2(K)) - 8;
      C2 = 2 << Stdm;
    }
  }

#ifdef RANDOM_DATA
  *(int64_t *)(&qdq_params[qdq_c0_idx]) = c0; // qdq_params[0] = c0;
  qdq_params[qdq_c1_idx] = C1;                // C1
  qdq_params[qdq_c2_idx] = C2;                // C2
  qdq_params[qdq_c3_idx] = 0;                 // C3
  // qdq_params[5] = Msubv;          // M
  // qdq_params[6] = Nsubv;          // N
  qdq_params[qdq_SQb_idx] = SQb;   // Shift_Qb
  qdq_params[qdq_Sout_idx] = Sout; // Shift_ou
  qdq_params[qdq_Stdm_idx] = Stdm;
  // for mxgan, user needs to set it based on Q datatype
  qdq_params[qdq_isint16_idx] = isint16;

  // NOTE: Set the Q/DQ params here
  InT gelu_in_dq_zero_point = 138;
  float gelu_in_dq_scale = 0.02688;
  InT gelu_out_q_zero_point = 13;
  float gelu_out_q_scale = 0.01295;

  if (a_dtype == "uint16") {
    gelu_in_dq_zero_point = 23799;
    gelu_in_dq_scale = 0.00101977;
    gelu_out_q_zero_point = 261;
    gelu_out_q_scale = 0.00065204;
  }
  // GELU IN dequant params
  gelu_qdq_params[gelu_dq_zp_idx] = gelu_in_dq_zero_point;
  gelu_qdq_params[gelu_dq_scale_idx] = float_to_bfloat16(gelu_in_dq_scale);
  // GELU OUT quant params
  gelu_qdq_params[gelu_q_zp_idx] = gelu_out_q_zero_point;
  gelu_qdq_params[gelu_q_scale_idx] = float_to_bfloat16(1.0 / gelu_out_q_scale);
  // for mxgan, user needs to set it based on Q datatype
  gelu_qdq_params[gelu_isint16_idx] = isint16;
  std::string Ytype;
  if (a_dtype == "uint16") {
    cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
               RowMajorMatrix<int32_t>>(X, W, cpu_Y, Stdm, Msubv_act, Ksubv,
                                        Nsubv);
    Ytype = "uint16";
  } else {
    cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
               RowMajorMatrix<int32_t>>(X, W, cpu_Y, "int32");

    Ytype = "uint8";
  }
  qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>, RowMajorMatrix<OuT>>(
      X, cpu_Y, C2, C1, C0_vec, SQb, Sout, cpu_Y_qdq, Ytype);
  // Compute GELU golden
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float in_gold =
          (cpu_Y_qdq.at(r, c) - gelu_in_dq_zero_point) * gelu_in_dq_scale;
      // if (in_gold > 8) {
      //    std::cout << "r, c: " << r << ", " << c << " in_gold: " << in_gold
      //    << std::endl;
      // }
      gelu_out_gold_mat.at(r, c) = float_to_bfloat16(gelu_golden(in_gold));
    }
  }
  // Quantisze gelu output
  quant(gelu_out_gold_mat, gelu_out_gold_quant_mat, gelu_out_q_scale,
        gelu_out_q_zero_point, Ytype);
#else
  std::string input_bin_name =
      OpInterface::get_dd_base_dir() +
      "//bin_files//malmulgeluaddV_test_data//input.bin";
  read_bin_file(input_bin_name, (char *)a.data());

  std::string weight_bin_name =
      OpInterface::get_dd_base_dir() +
      "//bin_files//malmulgeluaddV_test_data//weight.bin";
  read_bin_file(weight_bin_name, (char *)(b.data()));
  // Transpose the weights
  int Col = 768;
  int Row = 768;
  for (int i = 0; i < Col; i++) {
    for (int j = 0; j < i; j++) {
      uint8_t tmp = ((uint8_t *)b.data())[i * Col + j];
      ((uint8_t *)b.data())[i * Col + j] = ((uint8_t *)b.data())[i + j * Row];
      ((uint8_t *)b.data())[i + j * Row] = tmp;
    }
  }
  std::string c0_bin_name = OpInterface::get_dd_base_dir() +
                            "//bin_files//malmulgeluaddV_test_data//C0.bin";
  read_bin_file(c0_bin_name, (char *)(qdq.data()));

  *(int64_t *)(&qdq_params[qdq_c0_idx]) = 0; // qdq_params[0] = c0;
  qdq_params[qdq_c1_idx] = -466808478;       // C1
  qdq_params[qdq_c2_idx] = 30610392;         // C2
  qdq_params[qdq_c3_idx] = 0;                // C3
  // qdq_params[5] = Msubv;          // M
  // qdq_params[6] = Nsubv;          // N
  qdq_params[qdq_SQb_idx] = 0;   // Shift_Qb
  qdq_params[qdq_Sout_idx] = 31; // Shift_ou
  qdq_params[qdq_Stdm_idx] = 3;
  // for mxgan, user needs to set it based on Q datatype
  qdq_params[qdq_isint16_idx] = 1;

  // GELU IN dequant params
  gelu_qdq_params[gelu_dq_zp_idx] = 25681;
  gelu_qdq_params[gelu_dq_scale_idx] =
      float_to_bfloat16(0.00011030500900233164);
  // GELU OUT quant params
  gelu_qdq_params[gelu_q_zp_idx] = 2440;
  gelu_qdq_params[gelu_q_scale_idx] =
      float_to_bfloat16(1.0 / 0.00006967326044104993);
  // for mxgan, user needs to set it based on Q datatype
  gelu_qdq_params[gelu_isint16_idx] = 1;

  std::string golden_out_name =
      OpInterface::get_dd_base_dir() +
      "//bin_files//malmulgeluaddV_test_data//ofm_ref.bin";
  read_bin_file(golden_out_name, (char *)(ref_out.data()));

#endif
  std::map<std::string, std::any> attr;
  if (model_name.find("4x4") != std::string::npos) {
    attr["design_param"] = std::vector<string>{"4x4"};
  }

  ryzenai::matmul_v_geluadd matmul_v_gelu_ =
      ryzenai::matmul_v_geluadd<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype, false,
                                               attr);

  matmul_v_gelu_.debug(debug);
  matmul_v_gelu_.set_params(model_name, a_shape);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{b.data(), b_shape, b_dtype},
                  {qdq.data(), qdq_shape, "int64"},
                  {qdq_params.data(), qdq_params_shape, "int32"},
                  {gelu_qdq_params.data(), qdq_params_shape, "int32"}};

  matmul_v_gelu_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
  PROFILE_THIS(matmul_v_gelu_.execute(input_Tensor, output_Tensor));
#else
  matmul_v_gelu_.execute(input_Tensor, output_Tensor);
#endif
  err_count = check_result(gelu_out_gold_quant_mat, aie_Y);

  return err_count;
}

TEST(PSW_MatmulVAddGelu_Testa16w8, Kernel_64x768_768x768) {
  int err_count = test_matmul_v_gelu<uint16_t, uint8_t, uint16_t>(
      64, 768, 768, false, "uint16", "uint8", "uint16", "4x4PSW1.0");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
