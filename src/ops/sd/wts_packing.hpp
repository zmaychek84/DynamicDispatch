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
#include <any>
#include <assert.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "ops/sd/conv2d.hpp"
#include "ops/sd/gemm.hpp"
#include "ops/sd/matmul.hpp"

namespace nb = nanobind;

namespace ryzenai {
namespace sd {

nb::ndarray<nb::numpy, const uint8_t> conv_to_bfp16(nb::ndarray<> in_array,
                                                    nb::ndarray<> bias,
                                                    const std::string &op_name,
                                                    nb::ndarray<> input_shape) {
  if (op_name != "SDConv") {
    throw std::runtime_error("Only SDConv op is supported");
  }
  // support SDMatMul op
  auto nDim = in_array.ndim();
  size_t YI, XI, CI;
  size_t CO, KY, KX;
  size_t YO, XO;
  std::vector<size_t> weight_shape(nDim);
  std::vector<size_t> bias_shape(2);

  int wgt_size = 0;
  int *inshape_p = (int *)input_shape.data();

  YI = (size_t)inshape_p[0];
  XI = (size_t)inshape_p[1];
  CO = (size_t)in_array.shape(0);
  KY = (size_t)in_array.shape(1);
  KX = (size_t)in_array.shape(2);
  CI = (size_t)in_array.shape(3);
  YO = (size_t)inshape_p[2];
  XO = (size_t)inshape_p[3];
  weight_shape = {CO, KY, KX, CI}; // shape to c++ is co, ky, kx, ci
  bias_shape = {CO};
  wgt_size = int(CO * KY * KX * CI * 9 / 8 * 2); // inaccurate, changed later

  float *weight_p = (float *)in_array.data();
  float *bias_p = (float *)bias.data();

  std::vector<Tensor> const_tensors;
  const_tensors.push_back({weight_p, weight_shape, "float"});
  const_tensors.push_back({bias_p, bias_shape, "float"});

  size_t total_bytes = 0;
  std::vector<uint8_t> result_vec(wgt_size);
  std::map<std::string, std::any> attr;

  std::vector<Tensor> input;
  std::vector<Tensor> output;

  attr["input_shape"] = std::vector<int>{1, (int)YI, (int)XI, (int)CI};
  attr["output_shape"] = std::vector<int>{1, (int)YO, (int)XO, (int)CO};
  attr["weight_shape"] = std::vector<int>{(int)CO, (int)KY, (int)KX, (int)CI};
  ryzenai::sd::conv sd_conv =
      ryzenai::sd::conv<std::uint16_t, float, float, std::uint16_t>(
          "bfloat16", "float32", "float32", "bfloat16", false, attr);

  std::vector<uint16_t> act_p(YI * XI * CI);
  std::vector<uint16_t> out_p(YO * XO * CO);
  input = {{act_p.data(), {1, YI, XI, CI}, "bfloat16"}};
  std::vector<OpArgMap> arg_map = sd_conv.get_buffer_reqs(input, output, attr);
  for (auto arg : arg_map) {
    if (arg.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
      total_bytes = arg.size;
    }
  }
  result_vec.resize(total_bytes);
  auto bo_const = BoConst(result_vec.data());
  sd_conv.initialize_const_params(bo_const, const_tensors);

  input.clear();
  output.clear();
  const_tensors.clear();
  uint8_t *out_array = new uint8_t[total_bytes];
  std::memcpy(out_array, result_vec.data(), total_bytes);
  size_t shape[1] = {total_bytes};
  return nb::ndarray<nb::numpy, const uint8_t>(
      /* data = */ out_array,
      /* ndim = */ 1,
      /* shape pointer = */ shape,
      /* owner = */ nb::handle());
}

nb::ndarray<nb::numpy, const uint16_t>
matmul_to_bf16(const nb::ndarray<float> &float_wts, int B, int M) {
  // assume K, N is in float_wts dims
  if (float_wts.ndim() != 2) {
    throw std::runtime_error("Only 2D float weights are supported");
  }
  auto K = static_cast<int>(float_wts.shape(0));
  auto N = static_cast<int>(float_wts.shape(1));
  // std::cerr << "B: " << B << " M: " << M << " K: " << K << " N: " << N
  //           << " float_wts.size() " << float_wts.size() << std::endl;

  std::map<std::string, std::any> attr;
  attr["input_shape"] = std::vector<int>{B, M, K};
  attr["output_shape"] = std::vector<int>{B, M, N};
  attr["weight_shape"] = std::vector<int>{K, N};
  ryzenai::sd::matmul sd_matmul =
      ryzenai::sd::matmul<std::uint16_t, std::uint16_t, std::uint16_t>(
          "bfloat16", "bfloat16", "bfloat16", false, attr);

  // we may need to pad wts later so keep this copy for now
  std::vector<float> raw_wts(float_wts.data(),
                             float_wts.data() + float_wts.size());
  auto bf16_wts = sd_matmul.shuffle_wts_bf16(raw_wts);

  uint16_t *out_array = new uint16_t[bf16_wts.size()];
  std::memcpy(out_array, bf16_wts.data(), bf16_wts.size() * sizeof(uint16_t));
  size_t shape[1] = {bf16_wts.size()};
  nb::capsule owner(out_array,
                    [](void *p) noexcept { delete[] (uint16_t *)p; });
  return nb::ndarray<nb::numpy, const uint16_t>(out_array, {bf16_wts.size()},
                                                owner);
}

nb::ndarray<nb::numpy, const uint8_t>
gemm_to_bfp16(nb::ndarray<float> float_wts, nb::ndarray<float> float_bias,
              const std::string &op_name, nb::ndarray<> ifm_shape,
              bool bias_enable) {

  if (op_name != "SDGemm") {
    throw std::runtime_error("Only SDGemm op is supported");
  }

  if (float_wts.ndim() != 2) {
    throw std::runtime_error("Only 2D float weights are supported");
  }

  std::vector<int> input_shape;
  int *ifm_shape_p = (int *)ifm_shape.data();
  for (size_t i = 0; i < ifm_shape.size(); ++i) {
    input_shape.push_back(ifm_shape_p[i]);
  }

  auto K = static_cast<int>(float_wts.shape(0));
  auto N = static_cast<int>(float_wts.shape(1));
  std::vector<int> out_shape = input_shape;
  out_shape.back() = N;

  std::map<std::string, std::any> attr;
  attr["input_shape"] = input_shape;
  attr["output_shape"] = out_shape;
  attr["weight_shape"] = std::vector<int>{K, N};
  attr["bias_enable"] = bias_enable;

  ryzenai::sd::gemm sd_gemm =
      ryzenai::sd::gemm<std::uint16_t, float, float, std::uint16_t>(
          "bfloat16", "float32", "float32", "bfloat16", false, attr);

  sd_gemm.debug(false);
  // sd_gemm.set_params("", "DPU");

  std::vector<float> raw_wts(float_wts.data(),
                             float_wts.data() + float_wts.size());

  std::vector<float> raw_bias(float_bias.data(),
                              float_bias.data() + float_bias.size());
  auto wts_bfp16 = sd_gemm.shuffle_wts_bfp16(raw_wts.data(), raw_bias.data());

  uint8_t *out_array = new uint8_t[wts_bfp16.size()];
  std::memcpy(out_array, wts_bfp16.data(), wts_bfp16.size() * sizeof(uint8_t));
  size_t shape[1] = {wts_bfp16.size()};
  nb::capsule owner(out_array, [](void *p) noexcept { delete[] (uint8_t *)p; });
  return nb::ndarray<nb::numpy, const uint8_t>(out_array, {wts_bfp16.size()},
                                               owner);
}

} // namespace sd
} // namespace ryzenai
