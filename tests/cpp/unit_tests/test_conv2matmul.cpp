/*
 Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 Licensed under the MIT License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/conv2matmul/conv2matmul.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_conv2matmul(int H, int W, int C, int N, int vec_coeffs,
                     bool debug = false, const std::string &a_dtype = "int16",
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
  std::vector<int32_t> cpu_out(M * N);
  std::vector<OuT> cpu_out_qdq(M * N);
  std::vector<OuT> aie_out(M * N, garbage_value);

  RowMajorMatrix<InT> X(M, K, a.data());
  RowMajorMatrix<int32_t> cpu_Y(M, N, cpu_out.data());
  RowMajorMatrix<OuT> cpu_Y_qdq(M, N, cpu_out_qdq.data());
  RowMajorMatrix<OuT> aie_Y(M, N, aie_out.data());

  srand(0xABCD);
  init_random(X, 0, 2048);
  initialize_random<WgT>(b, b.size(), 128, 0);
  initialize_random<int64_t>(qdq, 1 * N, 10, 0);
  // initialize_random<int32_t>(c1_vec, 1 * N, 10, 0); // for PSU, zp_wgt = 0;
  initialize_random<int32_t>(c2_vec, 1 * N, 10, 0);
  uint32_t C1 = -11;
  if (model_name == "4x4PSU") {
    C1 = 0; // for PSU, zp_wgt = 0;
  }
  uint32_t C2 = 3;
  uint32_t SQb = 0;
  uint32_t Sout = 16;
  uint32_t Stdm = 2;
  int64_t *C0_vec = (int64_t *)qdq.data();
  int64_t c0 = 0;
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
#else
  std::string data_folder = OpInterface::get_dd_base_dir() + "//bin//PSU//";
  std::string wgt_filename = data_folder + "wgt.bin";
  std::string c0_filename = data_folder + "c0.bin";
  std::string qdq_filename = data_folder + "qdq.bin";
  std::string in_filename = data_folder + "in.bin";
  std::string golden_filename = data_folder + "golden.bin";

  read_bin_file(wgt_filename, reinterpret_cast<char *>(b.data()));
  read_bin_file(c0_filename, reinterpret_cast<char *>(qdq.data()));
  read_bin_file(qdq_filename, reinterpret_cast<char *>(qdq_params.data()));
  read_bin_file(in_filename, reinterpret_cast<char *>(a.data()));
  c0 = *(int64_t *)(&qdq_params[qdq_c0_idx]);
  C1 = qdq_params[qdq_c1_idx];
  C2 = qdq_params[qdq_c2_idx];
  SQb = qdq_params[qdq_SQb_idx];
  Sout = qdq_params[qdq_Sout_idx];
  Stdm = qdq_params[qdq_Stdm_idx];
  // qdq_params[qdq_veccoeffs_idx] = vec_coeffs;
#endif

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
                                    C0_vec, SQb, Sout, cpu_Y_qdq, c_dtype);
  } else {
    qdq_golden<RowMajorMatrix<InT>, RowMajorMatrix<int32_t>,
               RowMajorMatrix<OuT>>(X, cpu_Y, C2, C1, C0_vec, SQb, Sout,
                                    cpu_Y_qdq, c_dtype);
  }

#if 0
  std::string fld_name =
      OpInterface::get_dd_base_dir() + "//..//GemmV_1x768x8//";

  read_bin_file(fld_name + "input.bin", (char *)a.data());            // Input
  read_bin_file(fld_name + "weight.bin", (char *)b.data());           // wgt.bin
  read_bin_file(fld_name + "output.bin", (char *)cpu_out_qdq.data()); // ofm.bin
  read_bin_file(fld_name + "C0.bin", (char *)qdq.data());             // C0.bin

  *(int64_t *)(&qdq_params[qdq_c0_idx]) = 0; // qdq_params[0] = c0;
  qdq_params[qdq_c1_idx] = -744090460;       // C1
  qdq_params[qdq_c2_idx] = 46871840;         // C2
  qdq_params[qdq_c3_idx] = 0;                // C3
  // qdq_params[5] = Msubv;          // M
  // qdq_params[6] = Nsubv;          // N
  qdq_params[qdq_SQb_idx] = 0;   // Shift_Qb
  qdq_params[qdq_Sout_idx] = 36; // Shift_ou
  qdq_params[qdq_Stdm_idx] = 3;
  // for mxgan, user needs to set it based on Q datatype
  qdq_params[qdq_isint16_idx] = 1;
#endif

  std::map<std::string, std::any> attr;
  attr["input_format"] = std::vector<string>{"NCHW"};

  if (model_name == "4x4mzdk5" || model_name == "4x4PSW1.0") {
    attr["design_param"] = std::vector<string>{"4x4"};
    attr["input_shape"] = std::vector<int>{1, C, H, W};
  } else if (model_name == "4x4PSU") {
    attr["design_param"] = std::vector<string>{"4x4PSU"};
    attr["input_shape"] = std::vector<int>{1, C, H, W};
  }
  ryzenai::conv2matmul conv2matmul_ = ryzenai::conv2matmul<InT, WgT, OuT>(
      a_dtype, b_dtype, c_dtype, false, attr);

  conv2matmul_.debug(debug);
  std::vector<size_t> param_shape = {static_cast<size_t>(M), Cs, Ns};
  conv2matmul_.set_params(model_name, param_shape);

  std::vector<Tensor> const_Tensor;
  if (vec_coeffs == 1) {
    const_Tensor = {{b.data(), b_shape, b_dtype},
                    {qdq.data(), qdq_shape, "int64"},
                    {qdq_params.data(), qdq_params_shape, "int32"}};
  } else {
    const_Tensor = {{b.data(), b_shape, b_dtype},
                    {qdq.data(), qdq_shape, "int64"},
                    {qdq_params.data(), qdq_params_shape, "int32"},
                    {c1_vec.data(), qdq_shape, "int32"},
                    {c2_vec.data(), qdq_shape, "int32"}};
  }

  conv2matmul_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K << ", N = " << N);
  PROFILE_THIS(conv2matmul_.execute(input_Tensor, output_Tensor));
#else
  conv2matmul_.execute(input_Tensor, output_Tensor);
#endif
#ifndef RANDOM_DATA
  read_bin_file(golden_filename, reinterpret_cast<char *>(cpu_out_qdq.data()));
#endif
  err_count = check_result(cpu_Y_qdq, aie_Y);

  return err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_64_64_320_64) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      64, 64, 320, 64, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_1_77_1024_64) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      1, 77, 1024, 64, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_32_32_320_640) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 320, 640, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_32_32_640_64) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 640, 64, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_16_16_640_1280) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      16, 16, 640, 1280, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_16_16_1280_64) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      16, 16, 1280, 64, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_8_8_1280_64) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      8, 8, 1280, 64, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_8_8_2560_1280) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      8, 8, 2560, 1280, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_16_16_2560_1280) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      16, 16, 2560, 1280, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_16_16_1920_1280) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      16, 16, 1920, 1280, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_32_32_1920_640) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 1920, 640, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_32_32_1280_640) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 1280, 640, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_32_32_960_640) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      32, 32, 960, 640, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_64_64_960_320) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      64, 64, 960, 320, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_64_64_640_320) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      64, 64, 640, 320, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// mzdk5 GemmV with transB = 1
TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_1_1_1280_320) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      1, 1, 1280, 320, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_1_1_1280_640) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      1, 1, 1280, 640, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_CONV2GEMM_Testa16w8, Kernel_1_1_1280_1280) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      1, 1, 1280, 1280, 1, false, "uint16", "uint8", "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSW GemmV with transB = 1
TEST(PSW_CONV2GEMM_Testa16w8, Kernel_a16_w8_acc16_1_1_768_8) {
  int err_count = test_conv2matmul<uint16_t, uint8_t, uint16_t>(
      1, 1, 768, 8, 1, false, "uint16", "uint8", "uint16", "4x4PSW1.0");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSU0/1 v1.2
TEST(PSU_CONV2GEMM_Testa16w8, Kernel_1_1_3072_3072) {
  int err_count = test_conv2matmul<uint16_t, int8_t, uint16_t>(
      1, 1, 3072, 3072, 1, false, "uint16", "int8", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSU_CONV2GEMM_Testa16w8, Kernel_1_1_8192_3072) {
  int err_count = test_conv2matmul<uint16_t, int8_t, uint16_t>(
      1, 1, 8192, 3072, 1, false, "uint16", "int8", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSU_CONV2GEMM_Testa16w8, Kernel_1_64_3072_3072) {
  int err_count = test_conv2matmul<uint16_t, int8_t, uint16_t>(
      1, 64, 3072, 3072, 1, false, "uint16", "int8", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSU_CONV2GEMM_Testa16w8, Kernel_1_64_8192_3072) {
  int err_count = test_conv2matmul<uint16_t, int8_t, uint16_t>(
      1, 64, 8192, 3072, 1, false, "uint16", "int8", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSU0/1 v1.2 channelwise qdq
TEST(PSU_CONV2GEMM_Testa16w4, Kernel_1_64_3072_9216) {
  int err_count = test_conv2matmul<uint16_t, int8_t, uint16_t>(
      1, 64, 3072, 9216, 9216, false, "uint16", "int4", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSU_CONV2GEMM_Testa16w4, Kernel_1_1_3072_9216) {
  int err_count = test_conv2matmul<uint16_t, int8_t, uint16_t>(
      1, 1, 3072, 9216, 9216, false, "uint16", "int4", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSU_CONV2GEMM_Testa16w4, Kernel_1_1_3072_8192) {
  int err_count = test_conv2matmul<uint16_t, int8_t, uint16_t>(
      1, 1, 3072, 8192, 8192, false, "uint16", "int4", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSU_CONV2GEMM_Testa16w4, Kernel_1_64_3072_8192) {
  int err_count = test_conv2matmul<uint16_t, int8_t, uint16_t>(
      1, 64, 3072, 8192, 8192, false, "uint16", "int4", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSU_CONV2GEMM_Testa16w4, Kernel_1_64_3072_3072) {
  int err_count = test_conv2matmul<uint16_t, int8_t, uint16_t>(
      1, 64, 3072, 3072, 3072, false, "uint16", "int4", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSU_CONV2GEMM_Testa16w4, Kernel_1_1_3072_3072) {
  int err_count = test_conv2matmul<uint16_t, int8_t, uint16_t>(
      1, 1, 3072, 3072, 3072, false, "uint16", "int4", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSU_CONV2GEMM_Testa16w4, Kernel_1_64_8192_3072) {
  int err_count = test_conv2matmul<uint16_t, int8_t, uint16_t>(
      1, 64, 8192, 3072, 3072, false, "uint16", "int4", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSU_CONV2GEMM_Testa16w4, Kernel_1_1_8192_3072) {
  int err_count = test_conv2matmul<uint16_t, int8_t, uint16_t>(
      1, 1, 8192, 3072, 3072, false, "uint16", "int4", "uint16", "4x4PSU");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
