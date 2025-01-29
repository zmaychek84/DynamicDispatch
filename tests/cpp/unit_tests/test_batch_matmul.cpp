/*
 * Copyright � 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/dmacompiler/batch_matmul/batch_matmul.hpp>

#include "test_common.hpp"
// #define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_batch_matmul(int B, int M, int K, int N, int shape_format = 0,
                      bool debug = false, const std::string &a_dtype = "int16",
                      const std::string &b_dtype = "int8",
                      const std::string &c_dtype = "int32",
                      const std::string &model_name = "mdsqr") {
  int err_count = 0;
  int Msubv_act = 0;
  int Nsubv_act = 0;
  int Ksubv_act = 0;
  if (model_name == "4x4PSW1.0") {
    Msubv_act = Msubv_PSW_BMM;
    Ksubv_act = Ksubv_PSW_BMM;
    Nsubv_act = Nsubv_PSW_BMM;
  }
  size_t Bs = static_cast<size_t>(B);
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Bs, Ms, Ks};
  std::vector<size_t> b_shape = {Bs, Ks, Ns};
  std::vector<size_t> qdq_shape = {Bs, Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};
  std::vector<size_t> aie_out_shape = {Bs, Ms, Ns};

  std::vector<InT> a(B * M * K);
  std::vector<WgT> b(B * K * N);
  std::vector<int64_t> qdq(B * N); // c0
  std::vector<int32_t> qdq_params(QDQparam_size);
  std::vector<int32_t> cpu_out(B * M * N);
  std::vector<OuT> cpu_out_qdq(B * M * N);
  std::vector<OuT> aie_out(B * K * N, garbage_value);

  RowMajorMatrix<OuT> cpu_Y_qdq(K * B, N, cpu_out_qdq.data());
  RowMajorMatrix<OuT> aie_Y(K * B, N, aie_out.data());

#ifdef RANDOM_DATA
  RowMajorMatrix<InT> X(M, K * B, a.data());
  RowMajorMatrix<int32_t> cpu_Y(K * B, N, cpu_out.data());

  srand(0xABCD);
  init_random(X, 0, 32);
  initialize_random<WgT>(b, K * B * N, 32, 0);
  initialize_random<int64_t>(qdq, B * N, 32, 0); // add batch variable

  uint32_t C1 = 0;
  uint32_t C2 = 10;
  uint32_t SQb = 0;
  uint32_t Sout = 13;
  uint32_t Stdm = 0;
  int64_t *C0_vec = (int64_t *)qdq.data();
  int64_t c0 = 0;
  int isint16 = 1;
  if (a_dtype == "uint16") {
    srand(0xABCD);
    init_random(X, 0, 2048);
    initialize_random<WgT>(b, K * N, 128, 0);
    initialize_random<int64_t>(qdq, 12 * N, 10, 0);
    c0 = 0;
    C1 = -11;
    SQb = 0;
    Sout = 16;
    Stdm = 2; // round(log2(K)) - 8;
    C2 = 3;   // 2 << Stdm;
  }

  *(int64_t *)(&qdq_params[qdq_c0_idx]) = c0; // qdq_params[0] = c0;
  qdq_params[qdq_c1_idx] = C1;
  qdq_params[qdq_c2_idx] = C2;
  qdq_params[qdq_c3_idx] = 0;
  // qdq_params[qdq_Mv_idx] = Msubv_act;
  // qdq_params[qdq_Nv_idx] = Nsubv;
  qdq_params[qdq_SQb_idx] = SQb;
  qdq_params[qdq_Sout_idx] = Sout;
  qdq_params[qdq_Stdm_idx] = Stdm;
  qdq_params[qdq_isint16_idx] =
      isint16; // for mxgan, user needs to set it based on Q datatype

  RowMajorMatrix<WgT> W(K, N, b.data());
  if (a_dtype == "uint16") {
    cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
               RowMajorMatrix<int32_t>>(X, W, cpu_Y, Stdm, Msubv_act, Ksubv_act,
                                        Nsubv_act);
    qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>,
               RowMajorMatrix<OuT>>(X, cpu_Y, C2, C1, C0_vec, SQb, Sout,
                                    cpu_Y_qdq, "uint16");
  } else {
    cpu_matmul<RowMajorMatrix<InT>, RowMajorMatrix<WgT>,
               RowMajorMatrix<int32_t>>(X, W, cpu_Y, "int32");
    qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>,
               RowMajorMatrix<OuT>>(X, cpu_Y, C2, C1, C0_vec, SQb, Sout,
                                    cpu_Y_qdq, "uint8");
  }
#else

std:
  string fld_name = OpInterface::get_dd_base_dir() + "//..//bins//PSW//bmm//";

#if 0
  std::vector<uint32_t> aint(M * K);
  read_data_file<uint32_t>(OpInterface::get_dd_base_dir() + fld_name +
                               "//ifm.txt",
                           (uint32_t *)aint.data());
  for (int r = 0; r < M * K; r++) {
    a[r] = (InT)(aint[r]);
  }

  std::vector<uint32_t> bint(N * K);
  read_data_file<uint32_t>(OpInterface::get_dd_base_dir() + fld_name +
                               "//wgt.txt",
                           (uint32_t *)bint.data());
  for (int r = 0; r < N * K; r++) {
    b[r] = (InT)(bint[r]);
  }

  read_data_file<uint64_t>(OpInterface::get_dd_base_dir() + fld_name +
                               "//c0.txt",
                           (uint64_t *)qdq.data());

  read_data_file<uint32_t>(
      OpInterface::get_dd_base_dir() + fld_name + "//c1.txt", (uint32_t *)&C1);

  read_data_file<uint32_t>(
      OpInterface::get_dd_base_dir() + fld_name + "//c2.txt", (uint32_t *)&C2);

  read_data_file<uint32_t>(OpInterface::get_dd_base_dir() + fld_name +
                               "//shift_final.txt",
                           (uint32_t *)&Sout);

  read_data_file<uint32_t>(OpInterface::get_dd_base_dir() + fld_name +
                               "//shift_batch_matmul.txt",
                           (uint32_t *)&Stdm);

  *(int64_t *)(&qdq_params[qdq_c0_idx]) = c0; // qdq_params[0] = c0;
  qdq_params[qdq_c1_idx] = C1;
  qdq_params[qdq_c2_idx] = C2;
  qdq_params[qdq_c3_idx] = 0;
  // qdq_params[qdq_Mv_idx] = Msubv_act;
  // qdq_params[qdq_Nv_idx] = Nsubv;
  qdq_params[qdq_SQb_idx] = SQb;
  qdq_params[qdq_Sout_idx] = Sout;
  qdq_params[qdq_Stdm_idx] = Stdm;
  qdq_params[qdq_isint16_idx] =
      isint16; // for mxgan, user needs to set it based on Q datatype

  std::vector<uint32_t> outint(M * N);
  read_data_file<uint32_t>(OpInterface::get_dd_base_dir() + fld_name +
                               "//ofm.txt",
                           (uint32_t *)outint.data());
  for (int r = 0; r < N * M; r++) {
    cpu_out_qdq[r] = (OuT)(outint[r]);
  }
#else
  read_bin_file(fld_name + "Ifminput.bin", (char *)a.data());
  read_bin_file(fld_name + "weight.bin", (char *)b.data());
  read_bin_file(fld_name + "C0.bin", (char *)qdq.data());
  read_bin_file(fld_name + "output.bin", (char *)cpu_out_qdq.data());

  *(int64_t *)(&qdq_params[qdq_c0_idx]) = 0; // qdq_params[0] = c0;
  qdq_params[qdq_c1_idx] = -459354693;
  qdq_params[qdq_c2_idx] = 3734591;
  qdq_params[qdq_c3_idx] = 0;
  // qdq_params[qdq_Mv_idx] = Msubv_act;
  // qdq_params[qdq_Nv_idx] = Nsubv;
  qdq_params[qdq_SQb_idx] = 0;
  qdq_params[qdq_Sout_idx] = 30;
  qdq_params[qdq_Stdm_idx] = 0;
  qdq_params[qdq_isint16_idx] =
      1; // for mxgan, user needs to set it based on Q datatype

#endif

#endif
  std::map<std::string, std::any> attr;

  if (model_name == "4x4PSW1.0") {
    attr["design_param"] = std::vector<string>{"4x4"};
    attr["input_shape"] = std::vector<int>{B, M, K};
  }

  ryzenai::batch_matmul batch_matmul_ = ryzenai::batch_matmul<InT, WgT, OuT>(
      a_dtype, b_dtype, c_dtype, false, attr);

  batch_matmul_.debug(debug);
  std::vector<size_t> param_shape = {Bs, Ms, Ks, Ns};
  batch_matmul_.set_params(model_name, param_shape);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{b.data(), b_shape, b_dtype},
                  {qdq.data(), qdq_shape, "int64"},
                  {qdq_params.data(), qdq_params_shape, "int32"}};

  batch_matmul_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
  PROFILE_THIS(batch_matmul_.execute(input_Tensor, output_Tensor));
#else
  batch_matmul_.execute(input_Tensor, output_Tensor);
#endif
  // read_bin_file(golden_out_name,
  // reinterpret_cast<char*>(cpu_out_qdq.data()));
  err_count = check_result(cpu_Y_qdq, aie_Y, false);

#ifndef RANDOM_DATA
  return err_count;
#else
  return 0; // RANDOM_DATA not working as cpp reference implementation not
            // complete. Try adding reference implementation
#endif
}

// PSW 1.0 Batch Matmul 12x64x64 * 12x64x512 = 12x64x512 for remaining part of
// disentanglement attention
TEST(PSW_BatchMatMul_Testa16w8, Kernel_a16_w8_acc16_64x768_768x512_768x512) {
  int err_count = test_batch_matmul<uint16_t, uint8_t, uint16_t>(
      12, 64, 64, 512, 0, false, "uint16", "uint8", "uint16", "4x4PSW1.0");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
