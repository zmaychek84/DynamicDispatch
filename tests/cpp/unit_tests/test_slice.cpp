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

#include <ops/slice/slice.hpp>

#include "enable_perf.hpp"

#include "test_common.hpp"
using namespace std;

template <typename InT = int8_t, typename OuT = int16_t>
int test_slice(size_t M, size_t K, int sIdx, bool debug = false,
               const std::string &a_dtype = "int16",
               const std::string &c_dtype = "int32",
               const std::string &model_name = "mdsqr") {

  int err_count = 0;

  size_t Mo = M;
  size_t Ko = K / 2;

  std::vector<size_t> a_shape = {M, K};
  std::vector<size_t> c_shape = {Mo, Ko};

  std::vector<InT> a(M * K);
  std::vector<OuT> cpu_out(Mo * Ko);
  std::vector<OuT> aie_out(Mo * Ko);

  initialize_random<InT>(a, M * K, 128, 0);

  // compute golden
  int Wout_start = sIdx * Ko;
  for (int i = 0; i < Mo; ++i) {
    for (int j = 0; j < Ko; ++j) {
      cpu_out[(i * Ko) + j] = a[(i * K) + Wout_start + j];
    }
  }
  // run aie
  std::map<std::string, std::any> attr;

  if (model_name == "4x4mzdk5") {
    attr["design_param"] = std::vector<string>{"4x4"};
    attr["slice_idx"] = std::vector<int>{sIdx};
  }
  ryzenai::slice slice_ =
      ryzenai::slice<InT, OuT>(a_dtype, c_dtype, false, attr);
  slice_.debug(debug);
  std::vector<size_t> param_shape = {M, K};
  slice_.set_params(model_name, param_shape);

  std::vector<Tensor> const_Tensor;
  slice_.initialize_const_params(
      const_Tensor); // passing empty const tensor. call used only to initialize
                     // control packet related items

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), c_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  PROFILE_THIS(slice_.execute(input_Tensor, output_Tensor));
#else
  slice_.execute(input_Tensor, output_Tensor);
#endif
  for (int i = 0; i < Mo; ++i) {
    for (int j = 0; j < Ko; ++j) {
      InT ref = cpu_out[(i * Ko) + j];
      InT act = aie_out[(i * Ko) + j];
      if (ref != act) {
        std::cout << "ERROR: [" << i << ", " << j << "]: "
                  << "Expected: " << ref << ", "
                  << "Received: " << act << "\n";
        err_count += 1;
      }
    }
  }

  return err_count;
}

// NNI 4x4
TEST(C4mzdk5_SLICE_a16, Kernel_64_10240_0) {
  int err_count = test_slice<uint16_t, uint16_t>(64, 10240, 0, false, "uint16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_SLICE_a16, Kernel_256_10240_0) {
  int err_count = test_slice<uint16_t, uint16_t>(256, 10240, 0, false, "uint16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_SLICE_a16, Kernel_1024_5120_0) {
  int err_count = test_slice<uint16_t, uint16_t>(1024, 5120, 0, false, "uint16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_SLICE_a16, Kernel_4096_2560_0) {
  int err_count = test_slice<uint16_t, uint16_t>(4096, 2560, 0, false, "uint16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_SLICE_a16, Kernel_64_10240_1) {
  int err_count = test_slice<uint16_t, uint16_t>(64, 10240, 1, false, "uint16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_SLICE_a16, Kernel_256_10240_1) {
  int err_count = test_slice<uint16_t, uint16_t>(256, 10240, 1, false, "uint16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_SLICE_a16, Kernel_1024_5120_1) {
  int err_count = test_slice<uint16_t, uint16_t>(1024, 5120, 1, false, "uint16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4mzdk5_SLICE_a16, Kernel_4096_2560_1) {
  int err_count = test_slice<uint16_t, uint16_t>(4096, 2560, 1, false, "uint16",
                                                 "uint16", "4x4mzdk5");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
