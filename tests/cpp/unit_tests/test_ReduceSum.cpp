/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/ReduceSum/ReduceSum.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename InT = int8_t, typename WgT = int8_t, typename OutT = int8_t>
int test_ReduceSum(int M, int N, bool debug = false,
                   const std::string &a_dtype = "int16",
                   const std::string &b_dtype = "int16",
                   const std::string &c_dtype = "int16",
                   const std::string &model_name = "mdsqr") {

  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ns};
  std::vector<size_t> aie_out_shape = {1, Ns};

  std::vector<InT> a(M * N);
  std::vector<OutT> cpu_out(1 * N);

  std::vector<OutT> aie_out(1 * N);

  RowMajorMatrix<OutT> cpu_Y(1, N, cpu_out.data());
  RowMajorMatrix<OutT> aie_Y(1, N, aie_out.data());

#ifdef RANDOM_DATA

  srand(0xABCD);
  initialize_random_bfloat16(a, M * N, -0.2, 0.2);

  for (int c = 0; c < N; c++) {
    float r_sum = 0.0;
    for (int r = 0; r < M; r++) {
      r_sum += bfloat16_to_float(a[r * N + c]);
    }
    cpu_out[c] = float_to_bfloat16(r_sum);
  }

#else
  std::string fld_name;
  fld_name = "//bins//reducesum//";

  read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "ifm.bin",
                reinterpret_cast<char *>(a.data()));

  read_bin_file(OpInterface::get_dd_base_dir() + fld_name + "ofm.bin",
                reinterpret_cast<char *>(cpu_out.data()));
#endif

  // run aie
  std::map<std::string, std::any> attr;

  if (model_name == "4x4mzdk5") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }
  ryzenai::ReduceSum ReduceSum_ = ryzenai::ReduceSum<InT, WgT, OutT>(
      a_dtype, b_dtype, c_dtype, false, attr);

  ReduceSum_.debug(debug);
  ReduceSum_.set_params(model_name, a_shape);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", N = " << N);
  PROFILE_THIS(ReduceSum_.execute(input_Tensor, output_Tensor));
#else
  ReduceSum_.execute(input_Tensor, output_Tensor);
#endif

  // compare results
  err_count = check_result_bfloat(cpu_Y, aie_Y, 0.1);
  return err_count;
}

TEST(START_TAIL_mxgan_ReduceSum_Testa16, Kernel1) {
  int err_count = test_ReduceSum<uint16_t, uint16_t, uint16_t>(
      512, 768, false, "bfloat16", "bfloat16", "bfloat16", "START_TAIL_PS");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
