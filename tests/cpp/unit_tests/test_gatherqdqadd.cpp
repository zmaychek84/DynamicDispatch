/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include <ops/dmacompiler/gather_qdq_add/gather_qdq_add.hpp>

#include "test_common.hpp"
// #define RANDOM_DATA
using namespace matmul_matrix;
using namespace std;

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_gatherqdqadd(size_t M, size_t K, size_t N, bool debug = false,
                      const std::string &a_dtype = "uint16",
                      const std::string &b_dtype = "uint16",
                      const std::string &c_dtype = "uint16",
                      const std::string &model_name = "mdsqr") {
  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);

  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ms, Ks};
  std::vector<size_t> c_shape = {Ms, Ns};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};

  std::vector<InT> ab(2 * (M * K)); // only used with model data
  std::vector<InT> a(M * K);
  std::vector<WgT> b(M * K);
  std::vector<OuT> cpu_out(M * N);
  std::vector<OuT> cpu_q_out(M * N);
  std::vector<OuT> aie_out(M * N);
  std::vector<int32_t> qdq_params(QDQparam_size);
  OutMatrix<OuT, 1, 1> aie_Y(M, N, aie_out.data());
  OutMatrix<OuT, 1, 1> cpu_Y(M, N, cpu_out.data());
  OutMatrix<OuT, 1, 1> cpu_Q_Y(M, N, cpu_q_out.data());

  std::string fld_name =
      OpInterface::get_dd_base_dir() + "//gatherqdqadd_bins//";

  read_bin_file(fld_name + "ifm1.bin", (char *)a.data());
  read_bin_file(fld_name + "ifm2.bin", (char *)b.data());
  read_bin_file(fld_name + "ofm.bin", (char *)cpu_out.data());
  read_bin_file(fld_name + "wgt.bin", (char *)qdq_params.data());

  std::map<std::string, std::any> attr;

  if (model_name.find("4x4") != std::string::npos) {
    attr["design_param"] = std::vector<string>{"4x4"};
    if (Ms == 12 && Ks == 32768 && Ns == 4096) {
      attr["input_shape"] = std::vector<int>{12, 64, 512};
      attr["output_shape"] = std::vector<int>{12, 64, 64};
    }
  }
  ryzenai::gather_qdq_add gatherqdqadd_ =
      ryzenai::gather_qdq_add<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype, false,
                                             attr);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{qdq_params.data(), qdq_params_shape, "int32"}};

  std::vector<Tensor> input_Tensor;
  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor b_T = {b.data(), a_shape, b_dtype};
  struct Tensor c_T = {aie_out.data(), c_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(b_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  gatherqdqadd_.debug(debug);
  gatherqdqadd_.set_params(model_name, a_shape);
  gatherqdqadd_.initialize_const_params(const_Tensor);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(gatherqdqadd_.execute(input_Tensor, output_Tensor));
#else
  gatherqdqadd_.execute(input_Tensor, output_Tensor);
#endif
  if (c_dtype == "uint16") {
    err_count = check_add_result(cpu_Y, aie_Y, 0.01);
  } else {
    err_count = check_add_result_bfloat16<OuT>(cpu_out, aie_out, a_shape, 0.01);
  }
  return err_count;
}

// GatherQdqAdd
TEST(PSW_GATHERQDQADD_Testa16a16, Kernel_uint16_uint16_uint16_12_32768_4096) {
  int err_count = test_gatherqdqadd<uint16_t, uint16_t, uint16_t>(
      12, 32768, 4096, false, "uint16", "uint16", "uint16", "4x4PSW1.0");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
