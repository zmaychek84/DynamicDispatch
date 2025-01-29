/* Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
   Licensed under the MIT License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "../enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/experimental/matmul_tile.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_matmul_tile(int M, int K, int N, int shape_format = 0,
                     bool debug = false, const std::string &a_dtype = "int16",
                     const std::string &b_dtype = "int8",
                     const std::string &c_dtype = "int32",
                     const std::string &model_name = "mdsqr") {
  int err_count = 0;
  int Msubv_act = 0;
  int Nsubv_act = Nsubv;
  int Ksubv_act = Ksubv;
  if (a_dtype == "uint16") {
    Msubv_act = 32;
  } else if (a_dtype == "uint8") {
    Msubv_act = 64;
  } else {
    throw std::invalid_argument("a_dtype is not supported");
  }
  if (K % Ksubv_mzdk5 == 0) {
    Msubv_act = Msubv_mzdk5;
    Ksubv_act = Ksubv_mzdk5;
    Nsubv_act = Nsubv_mzdk5;
    if (N > 640) {
      Nsubv_act = Nsubv_mzdk5_LARGE;
    }
  }

  if (model_name == "4x4mzdk5") {
    SUBV_T key = {M, K, N};
    auto subv_mode = search_subv_mode(key);
    int factor = 1;
    while ((subv_mode < 0) && (factor < M)) {
      if (M % factor == 0) {
        subv_mode = search_subv_mode(key);
      }
      factor++;
    }
    SUBV_T subv = get_subv(subv_mode);
    Msubv_act = subv[0];
    Ksubv_act = subv[1];
    Nsubv_act = subv[2];
  }

  int N_w = N;
  // if (N_w < Nsubv * 2) {
  //   N_w = Nsubv * 2; // This is the miminum N
  // }
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ks, Ns};
  std::vector<size_t> qdq_shape = {Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};
  std::vector<size_t> aie_out_shape = {Ms, Ns};

  int P;
  if (shape_format == 1) { // mimic the tensor 4D shape
    P = sqrt(M);
    size_t Ps = static_cast<size_t>(P);
    a_shape = {1, Ps, Ps, Ks};
    aie_out_shape = {1, Ps, Ps, Ns};
  }

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

  // for (int i = 0; i < X.num_rows; ++i) {
  //   for (int j = 0; j < X.num_cols; ++j) {
  //     X.at(i, j) = 0;
  //     if (i == j){
  //       X.at(i, j) = 3;
  //     }
  //   }
  // }
  // for (size_t r = 0; r < K; ++r){
  //   for (size_t c = 0; c < N; ++c) {
  //     b.at(r * N + c) = 0;
  //     if (r == c){
  //       b.at(r * N + c) = 2;
  //     }
  //   }
  // }
  // for (size_t i = 0; i < N; i++) {
  //   qdq.at(i) = 0;
  // }
  init_random(X, 0, 32);
  initialize_random<WgT>(b, K * N, 32, 0);
  initialize_random<int64_t>(qdq, 1 * N, 32, 0);

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
    initialize_random<int64_t>(qdq, 1 * N, 10, 0);
    c0 = 0;
    C1 = -11;
    SQb = 0;
    Sout = 16;
    Stdm = 2; // round(log2(K)) - 8;
    C2 = 3;   // 2 << Stdm;
  }
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
  qdq_params[qdq_isint16_idx] =
      isint16; // for mxgan, user needs to set it based on Q datatype

  RowMajorMatrix<WgT> W(K, N_w, b.data());
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
  string fld_name = "//bin_files//m3uec_Matmul0";
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
                               "//shift_matmul.txt",
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
#endif
  std::map<std::string, std::any> attr;

  if (model_name == "4x4mzdk5") {
    attr["design_param"] = std::vector<string>{"4x4"};
    attr["input_shape"] = std::vector<int>{1, P, P, N};
  }

  ryzenai::matmul_tile matmul_ = ryzenai::matmul_tile<InT, WgT, OuT>(
      a_dtype, b_dtype, c_dtype, false, attr);

  matmul_.debug(debug);
  std::vector<size_t> param_shape = {Ms, Ks, Ns};
  matmul_.set_params(model_name, param_shape);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{b.data(), b_shape, b_dtype},
                  {qdq.data(), qdq_shape, "int64"},
                  {qdq_params.data(), qdq_params_shape, "int32"}};

  matmul_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

  // #ifdef UNIT_TEST_PERF
  //   LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
  //   PROFILE_THIS(matmul_.execute(input_Tensor, output_Tensor));
  // #else
  matmul_.execute(input_Tensor, output_Tensor);
  // #endif
  // read_bin_file(golden_out_name,
  // reinterpret_cast<char*>(cpu_out_qdq.data()));
  err_count = check_result(cpu_Y_qdq, aie_Y);

  return err_count;
}

// GEMMT a8w8
TEST(mdsqrv1_0_GEMMTILE_Testa8w8, Kernel1) {
  int err_count = test_matmul_tile<uint8_t, uint8_t, uint8_t>(
      1024, 1152, 1152, 0, false, "uint8", "uint8", "uint8", "mdsqr");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mdsqrv1_0_GEMMTILE_Testa8w8, Kernel2) {
  int err_count = test_matmul_tile<uint8_t, uint8_t, uint8_t>(
      2048, 768, 768, 0, false, "uint8", "uint8", "uint8", "mdsqr");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mdsqrv1_0_GEMMTILE_Testa8w8, Kernel3) {
  int err_count = test_matmul_tile<uint8_t, uint8_t, uint8_t>(
      2048, 3072, 768, 0, false, "uint8", "uint8", "uint8", "mdsqr");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mdsqrv1_0_GEMMTILE_Testa8w8, Kernel4) {
  int err_count = test_matmul_tile<uint8_t, uint8_t, uint8_t>(
      1024, 768, 26, 0, false, "uint8", "uint8", "uint8", "mdsqr");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mdsqrv1_0_GEMMTILE_Testa8w8, Kernel5) {
  int err_count = test_matmul_tile<uint8_t, uint8_t, uint8_t>(
      512, 768, 256, 0, false, "uint8", "uint8", "uint8", "mdsqr");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mdsqrv1_0_GEMMTILE_Testa8w8, Kernel6) {
  int err_count = test_matmul_tile<uint8_t, uint8_t, uint8_t>(
      1024, 768, 256, 0, false, "uint8", "uint8", "uint8", "mdsqr");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// mxpzi
TEST(mxpzi_GEMMTILE_Testa16w8, Kernel1) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      256, 1152, 1152, 0, false, "uint16", "uint8", "uint16", "mxpzi");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mxpzi_GEMMTILE_Testa16w8, Kernel2) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      256, 768, 256, 0, false, "uint16", "uint8", "uint16", "mxpzi");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mxpzi_GEMMTILE_Testa16w8, Kernel3) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      128, 768, 256, 0, false, "uint16", "uint8", "uint16", "mxpzi");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// mxgan

TEST(mxgan_GEMMTILE_Testa16w8, Kernel1) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      2048, 512, 1152, 0, false, "uint16", "uint8", "uint16", "mxgan");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mxganv1_1_GEMMTILE_Testa16w8, Kernel2) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      512, 768, 1536, 0, false, "uint16", "uint8", "uint16", "mxgan");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mxganv1_1_GEMMTILE_Testa16w8, Kernel3) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      1024, 768, 1536, 0, false, "uint16", "uint8", "uint16", "mxgan");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// m3uec
TEST(m3uec_GEMMTILE_Testa16w8, Kernel1) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      196, 1024, 3072, 0, false, "uint16", "uint8", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_GEMMTILE_Testa16w8, Kernel2) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      392, 2048, 512, 0, false, "uint16", "uint8", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_GEMMTILE_Testa16w8, Kernel3) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      1568, 1024, 256, 0, false, "uint16", "uint8", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_GEMMTILE_Testa16w8, Kernel4) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      3136, 512, 128, 0, false, "uint16", "uint8", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m3uec_GEMMTILE_Testa16w8, Kernel5) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      392, 2048, 1024, 0, false, "uint16", "uint8", "uint16", "m3uec");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// m7h4xjg

TEST(m7h4xjg_GEMMTILE_Testa16w8, Kernel1) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      154, 1024, 1024, 0, true, "uint16", "uint8", "uint16", "m7h4xjg");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(m7h4xjg_GEMMTILE_Testa16w8, Kernel2) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      154, 1024, 2048, 0, true, "uint16", "uint8", "uint16", "m7h4xjg");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// mzdk5 4x2

// TEST(mzdk5_GEMMTILE_Testa16w8, Kernel1) {
//   int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
//       256, 5120, 1280, 1, false, "uint16", "uint8", "uint16", "mzdk5");
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }
//
// TEST(mzdk5_GEMMTILE_Testa16w8, Kernel2) {
//   int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
//       4096, 640, 640, 1, false, "uint16", "uint8", "uint16", "mzdk5");
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }
//
// TEST(mzdk5_GEMMTILE_Testa16w8, Kernel3) {
//   int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
//       16384, 320, 2560, 1, false, "uint16", "uint8", "uint16", "mzdk5");
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }
//
// TEST(mzdk5_GEMMTILE_Testa16w8, Kernel4) {
//   int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
//       256, 5120, 2560, 1, false, "uint16", "uint8", "uint16", "mzdk5");
//   EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
// }

// mzdk5 4x4
TEST(C4mzdk5_GEMMTILE_Testa16w8, Kernel1) {
  int err_count = test_matmul_tile<uint16_t, uint8_t, uint16_t>(
      256, 1280, 10240, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// GEMMTILE a8w8
TEST(mdsqrv1_1_GEMMTILE_Testa8w8, Kernel1) {
  int err_count = test_matmul_tile<uint8_t, uint8_t, uint8_t>(
      1024, 512, 1152, 0, false, "uint8", "uint8", "uint8", "mdsqrv1.1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mdsqrv1_1_GEMMTILE_Testa8w8, Kernel2) {
  int err_count = test_matmul_tile<uint8_t, uint8_t, uint8_t>(
      1024, 768, 768, 0, false, "uint8", "uint8", "uint8", "mdsqrv1.1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mdsqrv1_1_GEMMTILE_Testa8w8, Kernel3) {
  int err_count = test_matmul_tile<uint8_t, uint8_t, uint8_t>(
      1024, 3072, 1536, 0, false, "uint8", "uint8", "uint8", "mdsqrv1.1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(mdsqrv1_1_GEMMTILE_Testa8w8, Kernel4) {
  int err_count = test_matmul_tile<uint8_t, uint8_t, uint8_t>(
      512, 768, 1536, 0, false, "uint8", "uint8", "uint8", "mdsqrv1.1");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
