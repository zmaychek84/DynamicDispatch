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
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>
#include <utility>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include <utils/dpu_mdata.hpp>

#include <txn_container.hpp>
#include <utils/instruction_registry.hpp>
#include <xrt_context/xrt_context.hpp>

#include <ops/experimental/elwadd_tile.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include "xaiengine.h"

#include "ops/ops_common/matmul_matrix.hpp"
#include <txn/txn_utils.hpp>
#include <txn_helper/txn_tiling_util.hpp>

using namespace matmul_matrix;
using utils::txn_util;

namespace ryzenai {

static std::tuple<size_t, size_t>
extract_MK(const std::vector<Tensor> &inputs) {
  size_t M = 0;
  size_t K = 0;
  if (inputs.at(0).shape.size() == 2) {
    M = inputs.at(0).shape.at(0);
    K = inputs.at(0).shape.at(1);
  } else if (inputs.at(0).shape.size() == 3) {
    M = inputs.at(0).shape.at(1);
    K = inputs.at(0).shape.at(2);
  } else if (inputs.at(0).shape.size() == 4) {
    if (inputs.at(0).shape.at(1) == inputs.at(0).shape.at(2)) { // NHWC
      M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(1) *
          inputs.at(0).shape.at(2);
      K = inputs.at(0).shape.at(3);
    } else { // NCHW
      M = inputs.at(0).shape.at(2) * inputs.at(0).shape.at(3);
      K = inputs.at(0).shape.at(1);
    }
  }
  return std::make_tuple(M, K);
}

template <typename InT, typename WtT, typename OutT>
std::once_flag elw_add_tile<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t elw_add_tile<InT, WtT, OutT>::elw_add_tile_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag elw_add_tile<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
void elw_add_tile<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
std::tuple<size_t, size_t, size_t>
elw_add_tile<InT, WtT, OutT>::map_padded_shape(size_t M, size_t N) const {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<std::tuple<int, int>> &supported_shapes = iter->second;
  size_t Mo = M;
  size_t No = N;
  size_t fidx = 0;
  bool f_found = false;
  bool t_found = false;               // if possibvle to tile
  size_t min_num_tiles = UINTMAX_MAX; // tracking the smallest number of tiles
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes[i];
    auto mat_M = std::get<0>(mat);
    auto mat_N = std::get<1>(mat);
    if (M == mat_M && N == mat_N) {
      fidx = i;
      f_found = true;
      break;
    } else if ((M * N) % (mat_M * mat_N) == 0) {
      // checking if tiling along m / n dimension is possible
      // minimize the number of tiles
      size_t tiles = (M * N) / (mat_M * mat_N);
      if (tiles < min_num_tiles) {
        fidx = i;
        t_found = true;
        min_num_tiles = tiles;
      }
    }
  }
  if (!(f_found || t_found)) {
    throw std::runtime_error("Cannot find the shape");
  }
  iter = default_shapes_.find(txn_fname_prefix_);
  const std::vector<std::tuple<int, int>> &actual_shapes = iter->second;
  auto mat = actual_shapes[fidx];
  Mo = std::get<0>(mat);
  No = std::get<1>(mat);
  size_t tile_info = 1;

  if (f_found) {
    return std::make_tuple(Mo, No, tile_info);
  }

  if (t_found && (M * N > Mo * No)) {
    // if padding is big enough then we don't need tiling
    RYZENAI_LOG_TRACE("M: " + std::to_string(M) + ", N: " + std::to_string(N) +
                      " mapped to tiling kernel: " + "Mo: " +
                      std::to_string(Mo) + ", No: " + std::to_string(No));
    tile_info = min_num_tiles;
  }
  return std::make_tuple(Mo, No, tile_info);
}

template <typename InT, typename WtT, typename OutT>
std::string
elw_add_tile<InT, WtT, OutT>::get_instr_key(std::string prefix, size_t m,
                                            size_t k, size_t tile_info) const {
  if (tile_info > 1) {
    return "elwaddtile_" + prefix + "_" + std::to_string(m) + "_" +
           std::to_string(k) + "_tile_" + std::to_string(tile_info);
  }
  return "elwaddtile_" + prefix + "_" + std::to_string(m) + "_" +
         std::to_string(k);
}

template <typename InT, typename WtT, typename OutT>
std::string elw_add_tile<InT, WtT, OutT>::get_param_key(std::string prefix,
                                                        size_t m,
                                                        size_t k) const {
  return "elwadd_" + prefix + "_" + std::to_string(m) + "_" +
         std::to_string(k) + "_param";
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
elw_add_tile<InT, WtT, OutT>::get_split_bo_txn_bin(std::string prefix, size_t m,
                                                   size_t k) const {
  std::string source_txn_key =
      "elwadd_" + prefix + "_" + std::to_string(m) + "_" + std::to_string(k);
  auto param_key = get_param_key(prefix, m, k);
  Transaction &txn = Transaction::getInstance();
  std::vector<uint8_t> data = txn.get_txn_bvec(source_txn_key);
  size_t input_1_bo_size = (m * k * sizeof(InT));
  size_t input_2_bo_size = (m * k * sizeof(WtT));
  size_t output_bo_size = (m * k * sizeof(OutT));
  size_t const_params_bo_size = matmul_matrix::QDQparam_size * sizeof(int32_t);
  size_t super_kernel_size = txn.get_txn_bvec(param_key).size();
  std::vector<OpArgMap> dest_arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::INPUT, 2, 1, 0, input_2_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 3, 2, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, 3, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 4, 0, 0,
       super_kernel_size}};
  std::vector<OpArgMap> source_arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::INPUT, 1, 1, input_1_bo_size, input_2_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 2, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, 3, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size}};
  std::vector<uint8_t> seperated_txn_bin =
      txn_util::patch(data, source_arg_map, dest_arg_map);
  return seperated_txn_bin;
}

template <typename InT, typename WtT, typename OutT>
void elw_add_tile<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> layer_params;
  for (const auto &[mkey, value] : default_shapes_) {
    auto iter = default_shapes_.find(mkey);
    std::vector<std::tuple<int, int>> &supported_shapes = iter->second;
    for (size_t i = 0; i < supported_shapes.size(); i++) {
      auto mat = supported_shapes[i];
      size_t M = std::get<0>(mat);
      size_t K = std::get<1>(mat);
      auto key = get_instr_key(mkey, M, K);
      auto param_key = get_param_key(mkey, M, K);
      {
        std::lock_guard lock(instr_reg_mutex_);
        if (!xrt_ctx_->get_registry().instr_in_registry(key)) {
          std::vector<uint8_t> seperated_txn_bin =
              get_split_bo_txn_bin(mkey, M, K);
          RYZENAI_LOG_TRACE("Split buffer instruction generated.");
          auto instr = std::make_pair(key, false);
          xrt_ctx_->get_registry().insert_fused_instr_to_instruction_map(
              instr, seperated_txn_bin);
        }
      }
      layer_params.push_back(std::make_pair(param_key, false));
    }
  }
  xrt_ctx_->get_registry().add_layer_params(layer_params);
}

template <typename InT, typename WtT, typename OutT>
elw_add_tile<InT, WtT, OutT>::elw_add_tile(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"bfloat16", "abf16"}, {"uint16", "a16"}, {"uint8", "a8"}};
  txnbin_b_header = {{"bfloat16", "abf16"}, {"uint16", "a16"}, {"uint8", "a8"}};
  txnbin_c_header = {{"bfloat16", "accbf16"}, {"uint16", "acc16"}};

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_["elwadd_4x2_a8a8accbf16"] =
      std::vector<std::tuple<int, int>>();
  default_shapes_["elwadd_4x2_a16a16accbf16"] =
      std::vector<std::tuple<int, int>>();
  default_shapes_["elwadd_4x2_abf16a16accbf16"] =
      std::vector<std::tuple<int, int>>();
  default_shapes_["elwadd_4x2_abf16a16acc16"] =
      std::vector<std::tuple<int, int>>();
  default_shapes_["elwadd_4x2_a16a16acc16"] =
      std::vector<std::tuple<int, int>>();
  default_shapes_["elwadd_4x4_abf16a16accbf16"] =
      std::vector<std::tuple<int, int>>();
  default_shapes_["elwadd_4x4_a16a16accbf16"] =
      std::vector<std::tuple<int, int>>();
  default_shapes_["elwadd_4x4_abf16a16acc16"] =
      std::vector<std::tuple<int, int>>();
  default_shapes_["elwadd_4x4_a16a16acc16"] =
      std::vector<std::tuple<int, int>>();

  default_shapes_["elwadd_4x2_a8a8accbf16"].push_back(
      std::make_tuple(512, 768));
  default_shapes_["elwadd_4x2_a8a8accbf16"].push_back(
      std::make_tuple(256, 768));

  default_shapes_["elwadd_4x2_a16a16accbf16"].push_back(
      std::make_tuple(128, 768));
  default_shapes_["elwadd_4x2_a16a16accbf16"].push_back(
      std::make_tuple(128, 768));
  default_shapes_["elwadd_4x2_a16a16accbf16"].push_back(
      std::make_tuple(512, 768));
  default_shapes_["elwadd_4x2_a16a16accbf16"].push_back(
      std::make_tuple(64, 1024));
  default_shapes_["elwadd_4x2_a16a16accbf16"].push_back(
      std::make_tuple(224, 512));
  default_shapes_["elwadd_4x2_a16a16accbf16"].push_back(
      std::make_tuple(832, 256));
  default_shapes_["elwadd_4x2_a16a16accbf16"].push_back(
      std::make_tuple(3200, 128));

  default_shapes_["elwadd_4x2_abf16a16accbf16"].push_back(
      std::make_tuple(64, 1024));
  default_shapes_["elwadd_4x2_abf16a16accbf16"].push_back(
      std::make_tuple(224, 512));
  default_shapes_["elwadd_4x2_abf16a16accbf16"].push_back(
      std::make_tuple(832, 256));
  default_shapes_["elwadd_4x2_abf16a16accbf16"].push_back(
      std::make_tuple(3200, 128));
  default_shapes_["elwadd_4x2_abf16a16accbf16"].push_back(
      std::make_tuple(128, 1024));

  default_shapes_["elwadd_4x2_abf16a16acc16"].push_back(
      std::make_tuple(64, 1024));
  default_shapes_["elwadd_4x2_abf16a16acc16"].push_back(
      std::make_tuple(224, 512));
  default_shapes_["elwadd_4x2_abf16a16acc16"].push_back(
      std::make_tuple(832, 256));
  default_shapes_["elwadd_4x2_abf16a16acc16"].push_back(
      std::make_tuple(3200, 128));

  default_shapes_["elwadd_4x2_a16a16acc16"].push_back(
      std::make_tuple(64, 1024));
  default_shapes_["elwadd_4x2_a16a16acc16"].push_back(
      std::make_tuple(224, 512));
  default_shapes_["elwadd_4x2_a16a16acc16"].push_back(
      std::make_tuple(832, 256));
  default_shapes_["elwadd_4x2_a16a16acc16"].push_back(
      std::make_tuple(3200, 128));

  default_shapes_["elwadd_4x4_abf16a16accbf16"].push_back(
      std::make_tuple(64, 1280));
  default_shapes_["elwadd_4x4_abf16a16accbf16"].push_back(
      std::make_tuple(256, 1280));
  default_shapes_["elwadd_4x4_abf16a16accbf16"].push_back(
      std::make_tuple(1024, 640));
  default_shapes_["elwadd_4x4_abf16a16accbf16"].push_back(
      std::make_tuple(4096, 320));

  default_shapes_["elwadd_4x4_a16a16accbf16"].push_back(
      std::make_tuple(64, 1280));
  default_shapes_["elwadd_4x4_a16a16accbf16"].push_back(
      std::make_tuple(256, 1280));
  default_shapes_["elwadd_4x4_a16a16accbf16"].push_back(
      std::make_tuple(1024, 640));
  default_shapes_["elwadd_4x4_a16a16accbf16"].push_back(
      std::make_tuple(4096, 320));

  default_shapes_["elwadd_4x4_abf16a16acc16"].push_back(
      std::make_tuple(64, 1280));
  default_shapes_["elwadd_4x4_abf16a16acc16"].push_back(
      std::make_tuple(256, 1280));
  default_shapes_["elwadd_4x4_abf16a16acc16"].push_back(
      std::make_tuple(1024, 640));
  default_shapes_["elwadd_4x4_abf16a16acc16"].push_back(
      std::make_tuple(4096, 320));

  default_shapes_["elwadd_4x4_a16a16acc16"].push_back(
      std::make_tuple(64, 1280));
  default_shapes_["elwadd_4x4_a16a16acc16"].push_back(
      std::make_tuple(256, 1280));
  default_shapes_["elwadd_4x4_a16a16acc16"].push_back(
      std::make_tuple(1024, 640));
  default_shapes_["elwadd_4x4_a16a16acc16"].push_back(
      std::make_tuple(4096, 320));

  // raw shape is the actual shape from ONNX
  raw_shapes_["elwadd_4x2_a8a8accbf16"] = std::vector<std::tuple<int, int>>();
  raw_shapes_["elwadd_4x2_a16a16accbf16"] = std::vector<std::tuple<int, int>>();
  raw_shapes_["elwadd_4x2_abf16a16accbf16"] =
      std::vector<std::tuple<int, int>>();
  raw_shapes_["elwadd_4x2_abf16a16acc16"] = std::vector<std::tuple<int, int>>();
  raw_shapes_["elwadd_4x2_a16a16acc16"] = std::vector<std::tuple<int, int>>();
  raw_shapes_["elwadd_4x4_abf16a16accbf16"] =
      std::vector<std::tuple<int, int>>();
  raw_shapes_["elwadd_4x4_a16a16accbf16"] = std::vector<std::tuple<int, int>>();
  raw_shapes_["elwadd_4x4_abf16a16acc16"] = std::vector<std::tuple<int, int>>();
  raw_shapes_["elwadd_4x4_a16a16acc16"] = std::vector<std::tuple<int, int>>();

  raw_shapes_["elwadd_4x2_a8a8accbf16"].push_back(std::make_tuple(512, 768));
  raw_shapes_["elwadd_4x2_a8a8accbf16"].push_back(std::make_tuple(256, 768));

  raw_shapes_["elwadd_4x2_a16a16accbf16"].push_back(std::make_tuple(77, 768));
  raw_shapes_["elwadd_4x2_a16a16accbf16"].push_back(std::make_tuple(128, 768));
  raw_shapes_["elwadd_4x2_a16a16accbf16"].push_back(std::make_tuple(512, 768));
  raw_shapes_["elwadd_4x2_a16a16accbf16"].push_back(std::make_tuple(49, 1024));
  raw_shapes_["elwadd_4x2_a16a16accbf16"].push_back(std::make_tuple(196, 512));
  raw_shapes_["elwadd_4x2_a16a16accbf16"].push_back(std::make_tuple(784, 256));
  raw_shapes_["elwadd_4x2_a16a16accbf16"].push_back(std::make_tuple(3136, 128));

  raw_shapes_["elwadd_4x2_abf16a16accbf16"].push_back(
      std::make_tuple(49, 1024));
  raw_shapes_["elwadd_4x2_abf16a16accbf16"].push_back(
      std::make_tuple(196, 512));
  raw_shapes_["elwadd_4x2_abf16a16accbf16"].push_back(
      std::make_tuple(784, 256));
  raw_shapes_["elwadd_4x2_abf16a16accbf16"].push_back(
      std::make_tuple(3136, 128));
  raw_shapes_["elwadd_4x2_abf16a16accbf16"].push_back(
      std::make_tuple(77, 1024));

  raw_shapes_["elwadd_4x2_abf16a16acc16"].push_back(std::make_tuple(49, 1024));
  raw_shapes_["elwadd_4x2_abf16a16acc16"].push_back(std::make_tuple(196, 512));
  raw_shapes_["elwadd_4x2_abf16a16acc16"].push_back(std::make_tuple(784, 256));
  raw_shapes_["elwadd_4x2_abf16a16acc16"].push_back(std::make_tuple(3136, 128));

  raw_shapes_["elwadd_4x2_a16a16acc16"].push_back(std::make_tuple(49, 1024));
  raw_shapes_["elwadd_4x2_a16a16acc16"].push_back(std::make_tuple(196, 512));
  raw_shapes_["elwadd_4x2_a16a16acc16"].push_back(std::make_tuple(784, 256));
  raw_shapes_["elwadd_4x2_a16a16acc16"].push_back(std::make_tuple(3136, 128));

  raw_shapes_["elwadd_4x4_abf16a16accbf16"].push_back(
      std::make_tuple(64, 1280));
  raw_shapes_["elwadd_4x4_abf16a16accbf16"].push_back(
      std::make_tuple(256, 1280));
  raw_shapes_["elwadd_4x4_abf16a16accbf16"].push_back(
      std::make_tuple(1024, 640));
  raw_shapes_["elwadd_4x4_abf16a16accbf16"].push_back(
      std::make_tuple(4096, 320));

  raw_shapes_["elwadd_4x4_a16a16accbf16"].push_back(std::make_tuple(64, 1280));
  raw_shapes_["elwadd_4x4_a16a16accbf16"].push_back(std::make_tuple(256, 1280));
  raw_shapes_["elwadd_4x4_a16a16accbf16"].push_back(std::make_tuple(1024, 640));
  raw_shapes_["elwadd_4x4_a16a16accbf16"].push_back(std::make_tuple(4096, 320));

  raw_shapes_["elwadd_4x4_abf16a16acc16"].push_back(std::make_tuple(64, 1280));
  raw_shapes_["elwadd_4x4_abf16a16acc16"].push_back(std::make_tuple(256, 1280));
  raw_shapes_["elwadd_4x4_abf16a16acc16"].push_back(std::make_tuple(1024, 640));
  raw_shapes_["elwadd_4x4_abf16a16acc16"].push_back(std::make_tuple(4096, 320));

  raw_shapes_["elwadd_4x4_a16a16acc16"].push_back(std::make_tuple(64, 1280));
  raw_shapes_["elwadd_4x4_a16a16acc16"].push_back(std::make_tuple(256, 1280));
  raw_shapes_["elwadd_4x4_a16a16acc16"].push_back(std::make_tuple(1024, 640));
  raw_shapes_["elwadd_4x4_a16a16acc16"].push_back(std::make_tuple(4096, 320));

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);
  elw_add_tile_id_ = elw_add_tile_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dd_base_dir() + ryzenai::mdsqr_A8W8_QDQ_XCLBIN_PATH;

  if (a_dtype_ == "uint16") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mxpzi_A16W8_QDQ_XCLBIN_PATH;
  }

  design_param_ = "";
  if (attr.count("design_param") &&
      attr.at("design_param").type() == typeid(std::vector<std::string>)) {
    const auto &design_param_vector =
        std::any_cast<const std::vector<std::string> &>(
            attr.at("design_param"));

    if (design_param_vector.size() == 1) {
      design_param_ = design_param_vector[0];
    } else {
      std::cout
          << "Design Format attribute does not have the expected number of "
             "elements.Number of passed : design_param_vector.size(), "
             "Expected:1"
          << std::endl;
    }
    RYZENAI_LOG_TRACE("iConv: DesignFormat: " + design_param_);
  }

  txn_fname_prefix_ = "elwadd_4x2_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_c_header.at(c_dtype_);

  param_fname_prefix_ = "elwadd_4x2_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_c_header.at(c_dtype_);

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    txn_fname_prefix_ = "elwadd_4x4_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_c_header.at(c_dtype_);

    param_fname_prefix_ = "elwadd_4x4_" + txnbin_a_header.at(a_dtype_) +
                          txnbin_b_header.at(b_dtype_) +
                          txnbin_c_header.at(c_dtype_);
  }

  KERNEL_M_MAX = 512;

  w_shape_[0] = KERNEL_M_MAX;
  w_shape_[1] = 768;

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  std::call_once(logger_flag_, []() {
    std::string header =
        "elw_add_tile_id M K N kernel_m kernel_k kernel_n Execute"
        "time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[ADD] ID: " + std::to_string(elw_add_tile_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype_ + ", " +
                    b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void elw_add_tile<InT, WtT, OutT>::set_params(const std::string &model_name,
                                              std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;
  if (model_name == "mdsqr") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mdsqr_A8W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "mxpzi") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mxpzi_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "mxgan") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mxgan_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "m3uec") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::m3uec_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "m7h4xjg") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::m7h4xjg_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "mzdk5") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mzdk5_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "4x4mzdk5") {
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                   ryzenai::mzdk54x4_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "mdsqrv1.1") {
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                   ryzenai::mdsqrv1_1_A8W8_QDQ_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }
  // std::cout << XCLBIN_FNAME << std::endl;

  auto [M, K, tile_info] =
      map_padded_shape(input_shape.at(0), input_shape.at(1));
  tile_info_ = tile_info;
  w_shape_[0] = M;
  w_shape_[1] = K;
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void elw_add_tile<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("elwaddtile initialize_const_params(ptr) ...");

  DD_THROW_IF((const_params.size() != 1),
              OpsFusion::dd_format("Unsupported const spec for Elwadd\n") +
                  OpsFusion::dd_format("(Details : #const params == 1 ({})",
                                       const_params.size()));

  auto qdq_params = (int32_t *)const_params.at(0).data;
  auto temp = qdq_params[0];
  qdq_params[0] = qdq_params[1];
  qdq_params[1] = temp;
  temp = qdq_params[2];
  qdq_params[2] = qdq_params[3];
  qdq_params[3] = temp;
  temp = qdq_params[4];
  qdq_params[4] = qdq_params[5];
  qdq_params[5] = temp;
  auto qdq_params_size = matmul_matrix::QDQparam_size * sizeof(int32_t);
  io.write(0, (void *)qdq_params, qdq_params_size);

  RYZENAI_LOG_TRACE("elwaddtile initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void elw_add_tile<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {

  if (const_params.size() != 1) {
    throw std::runtime_error(
        "ELWADDTILE IPU Wrapper expect to have one constant.");
  }

  kernel_x_shape_[0] = w_shape_[0];
  kernel_x_shape_[1] = w_shape_[1];

  kernel_y_shape_[0] = 0;
  kernel_y_shape_[1] = 0;

  kernel_z_shape_[0] = w_shape_[0];
  kernel_z_shape_[1] = w_shape_[1];

  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  /* Create input/output BOs */
  const size_t B_BO_SIZE = matmul_matrix::QDQparam_size * sizeof(int32_t);
  const size_t A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_) * tile_info_;

  const size_t C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_) * tile_info_;
  RYZENAI_LOG_TRACE("ELWADDTILE: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));

  a1_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(8));

  a2_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(8));

  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));

  // copy b_bo
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;

  auto b_copy_start = GET_ELAPSED_TIME_NS();
  auto b_format_start = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = b_bo_.map<WtT *>();
  auto bo_const = BoConst(b_bo_map);
  initialize_const_params(bo_const, const_params);
  auto b_format_stop = GET_ELAPSED_TIME_NS();
  auto b_copy_stop = GET_ELAPSED_TIME_NS();
  b_format_time_ += static_cast<int64_t>(b_format_stop - b_format_start);
  b_copy_time_ = static_cast<int64_t>(b_copy_stop - b_copy_start);

  // sync b_bo
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();
  b_sync_time_ = static_cast<int64_t>(b_sync_stop - b_sync_start);
}

template <typename InT, typename WtT, typename OutT>
void elw_add_tile<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                           std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 2) {
    throw std::runtime_error(
        "ELWADDTILE IPU Wrapper expect to have two inputs.");
  }
  const int a_idx = 0;
  // The first data is a and second data is b
  InT *a = (InT *)input.at(a_idx).data;
  InT *b = (InT *)input.at(a_idx + 1).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;

  auto exec_start = GET_ELAPSED_TIME_NS();

  a_shape_[0] = input.at(a_idx).shape.at(0);
  a_shape_[1] = input.at(a_idx).shape.at(1);

  // each input needs to have M * K size
  auto [M, K, tile_info] = map_padded_shape(a_shape_[0], a_shape_[1]);
  if (tile_info > 1) {
    std::string txn_key = get_instr_key(txn_fname_prefix_, M, K, tile_info);
    {
      std::lock_guard lock(instr_reg_mutex_);
      if (!xrt_ctx_->get_registry().instr_in_registry(txn_key)) {
        RYZENAI_LOG_TRACE(
            "Tiling required, generating tiled transaction binary ...");
        auto txn_bin_vec = get_transaction_bin(input, output);
        auto instr = std::make_pair(txn_key, true);
        xrt_ctx_->get_registry().insert_fused_instr_to_instruction_map(
            instr, txn_bin_vec);
      }
    }
  }
  c_shape_[0] = a_shape_[0];
  c_shape_[1] = a_shape_[1];

  size_t a_size = a_shape_[0] * a_shape_[1] * sizeof(InT);
  RYZENAI_LOG_TRACE("a_shape_0: " + std::to_string(a_shape_[0]) +
                    ", a_shape_1: " + std::to_string(a_shape_[1]));
  RYZENAI_LOG_TRACE("ELWADDTILE: a_size: " + std::to_string(a_size));
  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a1_bo_map = a1_bo_.map<InT *>();
  memcpy((void *)a1_bo_map, (void *)a, a_size);
  InT *a2_bo_map = a2_bo_.map<InT *>();
  memcpy((void *)a2_bo_map, (void *)b, a_size);

  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a1_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  a2_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  // prepare inst_bo and param_bo
  auto instr_bo_key = get_instr_key(txn_fname_prefix_, M, K, tile_info);
  auto param_bo_key = get_param_key(param_fname_prefix_, M, K);
  // std::cout << instr_bo_key << std::endl;
  // std::cout << param_bo_key << std::endl;
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  const xrt::bo &param_bo =
      xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  xrt::run run;
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run = kernel_(2, instr_bo, instr_bo_words,
                c_bo_.address() + DDR_AIE_ADDR_OFFSET,        // 0
                a1_bo_.address() + DDR_AIE_ADDR_OFFSET,       // 1
                a2_bo_.address() + DDR_AIE_ADDR_OFFSET,       // 2
                b_bo_.address() + DDR_AIE_ADDR_OFFSET,        // 3
                param_bo.address() + DDR_AIE_ADDR_OFFSET, 0); // 4
  run.wait2();
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);
  num_run_aie_++;

  // sync output activation to host memory
  auto c_sync_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  auto c_sync_stop = GET_ELAPSED_TIME_NS();
  c_sync_time_ += static_cast<int64_t>(c_sync_stop - c_sync_start);

  // copy c_bo to host memory
  auto aie_out = (OutT *)output.at(0).data;
  auto c_copy_start = GET_ELAPSED_TIME_NS();
  OutT *c_bo_map = c_bo_.map<OutT *>();
  memcpy((void *)aie_out, (void *)c_bo_map,
         c_shape_[0] * c_shape_[1] * sizeof(OutT));
  auto c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = static_cast<int64_t>(c_copy_stop - c_copy_start);
  auto exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(elw_add_tile_id_) + " " + std::to_string(a_shape_[0]) +
      " " + std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) +
      " " + std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> elw_add_tile<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K] = extract_MK(input);
  auto [Mo, Ko, tile_info] = map_padded_shape(M, K);
  std::string txn_key = get_instr_key(txn_fname_prefix_, Mo, Ko);
  std::vector<uint8_t> data = get_split_bo_txn_bin(txn_fname_prefix_, Mo, Ko);
  if (tile_info > 1) {
    RYZENAI_LOG_TRACE(("Dynamic shape required for elewadd, num tiles: " +
                       std::to_string(tile_info)));
    size_t const_params_bo_size =
        matmul_matrix::QDQparam_size * sizeof(int32_t);
    size_t input_1_bo_size = (Mo * Ko * sizeof(InT));
    size_t input_2_bo_size = (Mo * Ko * sizeof(WtT));
    size_t output_bo_size = (Mo * Ko * sizeof(OutT));
    size_t super_kernel_size = get_super_kernel_params(input, output).size();

    std::vector<OpArgMap> arg_map{
        {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_1_bo_size},
        {OpArgMap::OpArgType::INPUT, 2, 1, 0, input_2_bo_size},
        {OpArgMap::OpArgType::CONST_INPUT, 3, 2, 0, const_params_bo_size},
        {OpArgMap::OpArgType::OUTPUT, 0, 3, 0, output_bo_size},
        {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 4, 0, 0,
         super_kernel_size}};
    std::vector<size_t> tiles = {(size_t)tile_info};
    data = binary_op_tile_transaction_bin(data, arg_map, tiles);
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("Fused Dynamically Shaped : {}", txn_key));
  }
  return data;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
elw_add_tile<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K] = extract_MK(input);
  auto [Mo, Ko, tile_info] = map_padded_shape(M, K);
  // TODO: Add check to validate tensor shapes
  std::string param_key = get_param_key(param_fname_prefix_, Mo, Ko);
  // std::cout << "Super kernel params name : " << fname << std::endl;
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(param_key);
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> elw_add_tile<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, N] = extract_MK(input);

  auto [Mo, No, tile_info] = map_padded_shape(M, N);

  size_t const_params_bo_size = matmul_matrix::QDQparam_size * sizeof(int32_t);
  size_t input_1_bo_size = (Mo * No * sizeof(InT)) * tile_info;
  size_t input_2_bo_size = (Mo * No * sizeof(WtT)) * tile_info;
  size_t output_bo_size = (Mo * No * sizeof(OutT)) * tile_info;
  size_t super_kernel_size = get_super_kernel_params(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::INPUT, 2, 1, 0, input_2_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 3, 2, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, 3, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 4, 0, 0,
       super_kernel_size}};
  return arg_map;
}

template <typename InT, typename WtT, typename OutT>
std::mutex elw_add_tile<InT, WtT, OutT>::instr_reg_mutex_;

template class elw_add_tile<uint8_t, uint8_t, uint16_t>;
template class elw_add_tile<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
