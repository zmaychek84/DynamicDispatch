/*
 * Copyright � 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/conv2matmul_silu/conv2matmul_silu.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_conv2matmul_silu(int H, int W, int C, int N, int vec_coeffs,
                          bool debug = false,
                          const std::string &a_dtype = "int16",
                          const std::string &b_dtype = "int8",
                          const std::string &c_dtype = "int32",
                          const std::string &model_name = "mdsqr") {
  int err_count = 0;

  size_t Hs = static_cast<size_t>(H); // M = H*W
  size_t Ws = static_cast<size_t>(W);
  size_t Cs = static_cast<size_t>(C); // K
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {1, Hs, Ws, Cs};
  std::vector<size_t> b_shape = {Ns, Cs};
  std::vector<size_t> qdq_shape = {Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};
  std::vector<size_t> silu_qdq_params_shape = {QDQparam_size};
  std::vector<size_t> aie_out_shape = {1, Hs, Ws, Ns};

  int M, K;
  M = H * W;
  K = C;
  std::vector<InT> a(M * K);
  std::vector<WgT> b(K * N);
  if (b_dtype == "int4") {
    b.resize(K * N / 2);
  }
  std::vector<WgT> b_trans(K * N);
  std::vector<int64_t> qdq(1 * N);       // c0
  std::vector<int32_t> c1_vec(1 * N, 0); // c1_vec
  std::vector<int32_t> c2_vec(1 * N);    // c2_vec
  std::vector<int32_t> qdq_params(QDQparam_size);
  std::vector<int32_t> silu_qdq_params(QDQparam_size);
  std::vector<int32_t> cpu_out(M * N);
  std::vector<OuT> cpu_out_qdq(M * N);
  std::vector<OuT> cpu_gemm_out(M * N);
  std::vector<uint16_t> silu_out(M * N); // bfloat
  std::vector<OuT> aie_out(M * N, garbage_value);

  RowMajorMatrix<InT> X(M, K, a.data());
  RowMajorMatrix<int32_t> cpu_Y(M, N, cpu_out.data());
  RowMajorMatrix<OuT> cpu_Y_qdq(M, N, cpu_out_qdq.data());
  RowMajorMatrix<OuT> cpu_gemm_out_mat(M, N, cpu_gemm_out.data());
  RowMajorMatrix<uint16_t> silu_out_mat(M, N, silu_out.data());
  RowMajorMatrix<OuT> aie_Y(M, N, aie_out.data());

  srand(0xABCD);
  init_random(X, 0, 2048);
  initialize_random<WgT>(b, b.size(), 128, 0);
  initialize_random<int64_t>(qdq, 1 * N, 10, 0);
  // initialize_random<int32_t>(c1_vec, 1 * N, 10, 0); // for PSU, zp_wgt = 0;
  initialize_random<int32_t>(c2_vec, 1 * N, 10, 0);
  uint32_t C1 = -11;
  if (model_name == "psu") {
    C1 = 0; // for PSU, zp_wgt = 0;
  }
  uint32_t C2 = 3;
  uint32_t SQb = 0;
  uint32_t Sout = 16;
  uint32_t Stdm = 2;
  int64_t *C0_vec = (int64_t *)qdq.data();
  int64_t c0 = 0;
  uint16_t silu_in_dq_zero_point = 0;
  float silu_in_dq_scale = 1.0;
  uint16_t silu_out_q_zero_point = 0;
  float silu_out_q_scale = 1.0;

#ifdef RANDOM_DATA
  *(int64_t *)(&qdq_params[qdq_c0_idx]) = c0; // qdq_params[0] = c0;
  qdq_params[qdq_c1_idx] = C1;
  qdq_params[qdq_c2_idx] = C2;
  qdq_params[qdq_c3_idx] = 0;
  // qdq_params[qdq_Mv_idx] = Msubv_act;
  // qdq_params[qdq_Nv_idx] = Nsubv;
  qdq_params[qdq_SQb_idx] = SQb;
  qdq_params[qdq_Sout_idx] = Sout;
  qdq_params[qdq_Stdm_idx] = Stdm;
  qdq_params[qdq_veccoeffs_idx] = vec_coeffs;

  // SILU IN dequant params
  silu_qdq_params[0] = silu_in_dq_zero_point;
  silu_qdq_params[1] = float_to_bfloat16(silu_in_dq_scale);
  // SILU OUT quant params
  silu_qdq_params[2] = silu_out_q_zero_point;
  silu_qdq_params[3] = float_to_bfloat16(1.0 / silu_out_q_scale);

  RowMajorMatrix<WgT> Wmat(K, N, b_trans.data());
  if (b_dtype == "int4") {
    // b is NxK int4 matrix 2x4bit in a single byte
    // b_trans is KxN int8 matrix
    int8_t temp;
    for (int r = 0; r < K; r += 2) {
      for (int c = 0; c < N; c++) {
        temp = b[(c * K / 2) + r / 2] & 0x0F;
        if (temp > 7) {
          temp = temp - 16;
        }
        Wmat.at(r, c) = (WgT)(temp);
        temp = b[(c * K / 2) + r / 2] >> 4;
        Wmat.at(r + 1, c) = (WgT)(temp);
      }
    }
  } else {
    for (int r = 0; r < K; r++) {
      for (int c = 0; c < N; c++) {
        Wmat.at(r, c) = (WgT)(b[c * K + r]);
      }
    }
  }
  cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>, RowMajorMatrix<int32_t>>(
      X, Wmat, cpu_Y, Stdm);
  if (vec_coeffs > 1) {
    qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>,
               RowMajorMatrix<OuT>>(X, cpu_Y, c2_vec.data(), c1_vec.data(),
                                    C0_vec, SQb, Sout, cpu_gemm_out_mat,
                                    c_dtype);
  } else {
    qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>,
               RowMajorMatrix<OuT>>(X, cpu_Y, C2, C1, C0_vec, SQb, Sout,
                                    cpu_gemm_out_mat, c_dtype);
  }

  // Compute SILU golden
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float in_gold = (cpu_gemm_out_mat.at(r, c) - silu_in_dq_zero_point) *
                      silu_in_dq_scale;
      silu_out_mat.at(r, c) = float_to_bfloat16(silu_golden(in_gold));
    }
  }
  // Quantize SILU output
  quant_bfloat16_to_int16(silu_out_mat, cpu_Y_qdq, silu_out_q_scale,
                          silu_out_q_zero_point);

#endif
  std::map<std::string, std::any> attr;
  attr["input_format"] = std::vector<string>{"NCHW"};

  if (model_name == "4x4PSU") {
    attr["design_param"] = std::vector<string>{"4x4PSU"};
    attr["input_shape"] = std::vector<int>{1, C, H, W};
  }
  ryzenai::conv2matmul_silu conv2matmul_silu_ =
      ryzenai::conv2matmul_silu<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype, false,
                                               attr);

  conv2matmul_silu_.debug(debug);
  std::vector<size_t> param_shape = {static_cast<size_t>(M), Cs, Ns};
  conv2matmul_silu_.set_params(model_name, param_shape);

  std::vector<Tensor> const_Tensor;
  if (vec_coeffs == 1) {
    const_Tensor = {{b.data(), b_shape, b_dtype},
                    {qdq.data(), qdq_shape, "int64"},
                    {qdq_params.data(), qdq_params_shape, "int32"},
                    {silu_qdq_params.data(), silu_qdq_params_shape, "int32"}};
  } else {
    const_Tensor = {{b.data(), b_shape, b_dtype},
                    {qdq.data(), qdq_shape, "int64"},
                    {qdq_params.data(), qdq_params_shape, "int32"},
                    {silu_qdq_params.data(), silu_qdq_params_shape, "int32"},
                    {c1_vec.data(), qdq_shape, "int32"},
                    {c2_vec.data(), qdq_shape, "int32"}};
  }

  conv2matmul_silu_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
  PROFILE_THIS(conv2matmul_silu_.execute(input_Tensor, output_Tensor));
#else
  conv2matmul_silu_.execute(input_Tensor, output_Tensor);
#endif
  // read_bin_file(golden_out_name,
  // reinterpret_cast<char*>(cpu_out_qdq.data()));
  err_count = check_add_result(cpu_Y_qdq, aie_Y, 0.1);

  return err_count;
}

// PSU0/1 v1.2 channelwise qdq
// TEST(PSU_CONV2GEMM_SILU_Testa16w4, Kernel_1_64_3072_16384) {
//  int err_count = test_conv2matmul_silu<uint16_t, int8_t, uint16_t>(
//      1, 64, 3072, 16384, 16384, false, "uint16", "int4", "uint16", "4x4PSU");
//  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
//}

// TEST(PSU_CONV2GEMM_SILU_Testa16w4, Kernel_1_1_3072_16384) {
//   int err_count = test_conv2matmul_silu<uint16_t, int8_t, uint16_t>(
//       1, 1, 3072, 16384, 16384, false, "uint16", "int4", "uint16", "4x4PSU");
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }

TEST(PSU_CONV2GEMM_SILU_Testa16w4, Kernel_1_1_3072_8192) {
  int err_count = test_conv2matmul_silu<uint16_t, int8_t, uint16_t>(
      1, 1, 3072, 8192, 8192, false, "uint16", "int4", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSU_CONV2GEMM_SILU_Testa16w4, Kernel_1_64_3072_8192) {
  int err_count = test_conv2matmul_silu<uint16_t, int8_t, uint16_t>(
      1, 64, 3072, 8192, 8192, false, "uint16", "int4", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
