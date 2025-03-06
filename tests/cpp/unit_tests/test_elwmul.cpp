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
#include <tuple>

#include "../src/ops/ops_common/matmul_matrix.hpp"
#include <ops/elwmul/elwmul.hpp>
#include <stdexcept>

#include "enable_perf.hpp"

#include "test_common.hpp"

using namespace matmul_matrix;
template <typename LhsT = int16_t, typename RhsT = int16_t,
          typename OuT = int16_t>
int test_elwmul(size_t M, size_t K, bool debug = false,
                const std::string &a_dtype = "bfloat16",
                const std::string &b_dtype = "bfloat16",
                const std::string &c_dtype = "bfloat16",
                const std::string &model_name = "LLAMA2",
                const std::string &op_version = "v1") {
  int err_count = 0;
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);

  std::vector<size_t> a_shape = {Ms, Ks};
  std::vector<size_t> b_shape = {Ms, Ks};

  std::vector<LhsT> a(M * K);
  std::vector<LhsT> b(M * K);
  std::vector<float> cpu_out(M * K);
  std::vector<OuT> aie_out(M * K, garbage_value);

  dd::initialize_random_bfloat16(a, 40);
  dd::initialize_random_bfloat16(b, 40);

  // compute golden
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < K; c++) {
      cpu_out.at(r * K + c) = bfloat16_to_float(a.at(r * K + c)) *
                              bfloat16_to_float(b.at(r * K + c));
    }
  }

  std::map<std::string, std::any> attr;
  std::vector<int> size_matmul_M{1, 128, 256, 512, 1024, 2048};
  std::vector<std::vector<int>> shape_list;
  for (auto m : size_matmul_M) {
    if (K <= 14336) {
      shape_list.push_back({m, (int)K});
    } else {
      shape_list.push_back({m, 14336});
    }
  }
  // attr["shapes"] = shape_list;
  attr["op_version"] = op_version;
  ryzenai::elw_mul elwmul_ =
      ryzenai::elw_mul<LhsT, RhsT, OuT>(a_dtype, true, attr);

  std::vector<Tensor> const_Tensor;
  std::vector<Tensor> input_Tensor;

  struct Tensor a_T = {a.data(), a_shape, a_dtype};
  struct Tensor b_T = {b.data(), a_shape, a_dtype};
  struct Tensor c_T = {aie_out.data(), a_shape, c_dtype};
  input_Tensor.push_back(a_T);
  input_Tensor.push_back(b_T);

  std::vector<Tensor> output_Tensor;
  output_Tensor.push_back(c_T);

  elwmul_.debug(debug);

#ifdef UNIT_TEST_PERF
  LOG_THIS("M = " << M << ", K = " << K);
  PROFILE_THIS(elwmul_.execute(input_Tensor, output_Tensor));
#else
  elwmul_.execute(input_Tensor, output_Tensor);
#endif

  err_count = dd::count_errors_floatvsbfloat16(cpu_out, aie_out, a_shape, 4);

  return err_count;
}

TEST(LLAMA2_ELWMUL_V1, AutoRunAllTxnShapes) {
  // Create an operator instance that can parse transaction files
  // and fill its supported_shapes_.
  using ElwMulOp = ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>;
  ElwMulOp shapeFinderOp(
      /*a_dtype=*/"bfloat16",
      /*load_xrt=*/true,
      /*attr=*/std::map<std::string, std::any>());

  // Retrieve the discovered shapes.
  auto shapes = std::vector(shapeFinderOp.get_supported_shapes());

  // // Build a skip set for elwmul shapes (if needed).
  // // The skip set should contain keys generated from M and K.
  // auto skipSet = buildSkipSet_elwmul();

  // // Remove shapes that are in the skip set.
  // shapes.erase(
  //     std::remove_if(shapes.begin(), shapes.end(), [&](const auto &s) {
  //       std::string key = shapeToKey(s.M, s.K);
  //       return skipSet.find(key) != skipSet.end();
  //     }),
  //     shapes.end());

  // Loop over each discovered shape and run the elwmul test.
  for (const auto &s : shapes) {
    int M = std::get<0>(s);
    int K = std::get<1>(s);
    int err_count =
        test_elwmul<uint16_t, uint16_t, uint16_t>(M, K,
                                                  /*debug=*/false,
                                                  /*a_dtype=*/"bfloat16",
                                                  /*b_dtype=*/"bfloat16",
                                                  /*c_dtype=*/"bfloat16",
                                                  /*model_name=*/"LLAMA2",
                                                  /*op_version=*/"v1");

    EXPECT_EQ(err_count, 0) << "[test_elwmul] Error count = " << err_count
                            << " for shape M=" << M << ", K=" << K;
  }
}
