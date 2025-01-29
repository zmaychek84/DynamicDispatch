// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
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

#include <gtest/gtest.h>
#include <iostream>

#include <ops/experimental/square.hpp>

#include "test_common.hpp"

#define NUM_ELEMS 32

template <typename InT, typename OutT> int32_t test_square() {

  std::vector<InT> input(NUM_ELEMS, 0);
  std::vector<OutT> output(NUM_ELEMS, 0);
  std::vector<OutT> golden(NUM_ELEMS, 0);

  for (int i = 0; i < NUM_ELEMS; i++) {
    input.at(i) = i;
    golden.at(i) = input.at(i) * input.at(i);
  }

  ryzenai::square square_ = ryzenai::square<InT, OutT>(true);
  std::vector<Tensor> input_tensors;
  input_tensors = {{input.data(), {32}, "int32"}};
  std::vector<Tensor> output_tensors;
  output_tensors = {{output.data(), {32}, "int32"}};
  std::vector<Tensor> const_tensors;
  square_.initialize_const_params(const_tensors);
  square_.execute(input_tensors, output_tensors);

  int32_t err_count = 0;
  for (int i = 0; i < NUM_ELEMS; i++) {
    std::cout << "Golden: " << golden.at(i) << " , Actual: " << output.at(i)
              << std::endl;
    if (golden.at(i) != output.at(i)) {
      err_count++;
    }
  }

  return err_count;
}

TEST(EXPTL_square, square1) {
  int32_t err_count = test_square<int32_t, int32_t>();
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
