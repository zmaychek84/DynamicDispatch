/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */
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

namespace nb = nanobind;
// using namespace OpArgMap::OpArgType;
// namespace ryzenai{
// namespace matmul_bfp16 {

// template <typename InT, typename WtT, typename OutT>
nb::ndarray<nb::numpy, const uint8_t>
const_from_fp32_to_bfp16(nb::ndarray<> in_array, nb::ndarray<> bias,
                         const std::string &op_name,
                         nb::ndarray<> input_shape) {

  if (op_name != "SDConv") {
    throw std::runtime_error("Only SDConv op is supported");
  }
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

  int total_bytes = 0;
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

//  } //namespace matmul
//  } //namesapce ryzenai
