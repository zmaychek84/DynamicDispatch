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

#include <array>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

#include "ops/op_builder.hpp"

#include "test_common.hpp"

struct OpConfig {
  std::string name;
  std::vector<std::string> types;
  std::map<std::string, std::any> attr;

  friend void PrintTo(const OpConfig &config, std::ostream *os) {
    *os << config.name << "(";
    for (const auto &type : config.types) {
      *os << type << ",";
    }
    *os << ")";
  }
};

class IsSupportedFixture : public testing::TestWithParam<OpConfig> {};
class IsNotSupportedFixture : public testing::TestWithParam<OpConfig> {};

TEST_P(IsSupportedFixture, IsSupported) {
  const auto &config = GetParam();
  EXPECT_TRUE(OpsFusion::OpBuilder::is_supported(config.name, config.types,
                                                 config.attr));
}

TEST_P(IsNotSupportedFixture, IsNotSupported) {
  const auto &config = GetParam();
  EXPECT_FALSE(OpsFusion::OpBuilder::is_supported(config.name, config.types,
                                                  config.attr));
}

// not exhaustive but could be
const std::array kSupportedConfigs{
    OpConfig{"MatMul", {"uint8", "uint8", "uint8"}, {}},
    OpConfig{"MatMul", {"uint16", "uint8", "uint16"}, {}},
    OpConfig{"LRN", {"bfloat16", "uint16", "uint8"}, {}},
    OpConfig{"LayerNorm", {"bfloat16", "uint16", "uint16"}, {}},
    OpConfig{"MatMulAdd", {"uint8", "uint8", "uint8"}, {}},
    OpConfig{"MatMulAdd", {"uint16", "uint8", "uint16"}, {}},
    OpConfig{"ADD", {"uint8", "", "bfloat16"}, {}},
    OpConfig{"Add", {"uint16", "", "bfloat16"}, {}},
    OpConfig{"MHAGRPB", {"uint8", "uint8", "uint8"}, {}},
    OpConfig{"MHAGRPB", {"uint16", "uint8", "uint16"}, {}},
    OpConfig{"MatMulAddGelu", {"uint8", "uint8", "uint8"}, {}},
    OpConfig{"MatMulAddGelu", {"uint16", "uint8", "uint16"}, {}},
    OpConfig{"square", {}, {}},
    OpConfig{"cube", {}, {}},
    OpConfig{"PM_LOAD", {}, {}},
};

const std::array kNotSupportedConfigs{
    OpConfig{"MatMul", {"float", "float", "float"}},
    OpConfig{"MatMul", {"uint32", "uint32", "uint32"}},
};

INSTANTIATE_TEST_CASE_P(IsSupported, IsSupportedFixture,
                        testing::ValuesIn(kSupportedConfigs));
INSTANTIATE_TEST_CASE_P(IsNotSupported, IsNotSupportedFixture,
                        testing::ValuesIn(kNotSupportedConfigs));
