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
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
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
#include <xclbin_container.hpp>
#include <xrt_context/xrt_context.hpp>

#include <ops/mladfmharope/mladfmharope.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

namespace ryzenai {

namespace {
std::string getXCLBinName(std::string op_version) {
  if (op_version == "v1" || op_version == "flat") {
    return LLAMA2_MLADF_2x4x4_V1_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_NAME;
  } else if (op_version == "v2") {
    return LLAMA2_MLADF_2x4x4_V2_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_NAME;
  } else {
    return LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_NAME;
  }
}

std::tuple<size_t, size_t, size_t> extract_BMK(const Tensor &input) {
  size_t B = 0;
  size_t M = 0;
  size_t K = 0;
  if (input.shape.size() != 3) {
    throw std::runtime_error(
        "mharope expects a rank 3 tensor [Batch,Rows,Cols]");
  }
  B = input.shape.at(0);
  M = input.shape.at(1);
  K = input.shape.at(2);
  return std::make_tuple(B, M, K);
}
} // namespace

template <typename LhsT, typename TrigT, typename OutT>
bool mha_rope<LhsT, TrigT, OutT>::isSupportedShape(const Tensor &operand) {
  const auto &supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  const auto shape_operand = extract_BMK(operand);
  for (const auto &supported : supported_shapes) {
    if (supported == shape_operand) {
      return true;
    }
  }
  return false;
}

template <typename LhsT, typename TrigT, typename OutT>
std::once_flag mha_rope<LhsT, TrigT, OutT>::logger_flag_;

template <typename LhsT, typename TrigT, typename OutT>
uint64_t mha_rope<LhsT, TrigT, OutT>::mha_rope_count = 0;

template <typename LhsT, typename TrigT, typename OutT>
std::once_flag mha_rope<LhsT, TrigT, OutT>::instr_reg_flag_;

template <typename LhsT, typename TrigT, typename OutT>
std::once_flag mha_rope<LhsT, TrigT, OutT>::instr_reg_v1_flag_;

template <typename LhsT, typename TrigT, typename OutT>
void mha_rope<LhsT, TrigT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename LhsT, typename TrigT, typename OutT>
std::string mha_rope<LhsT, TrigT, OutT>::get_instr_key(std::string prefix,
                                                       size_t batch, size_t m,
                                                       size_t k) const {
  return "mladfmharope_" + prefix + "_" + std::to_string(batch) + "_" +
         std::to_string(m) + "_" + std::to_string(k);
}

template <typename LhsT, typename TrigT, typename OutT>
void mha_rope<LhsT, TrigT, OutT>::set_kernel_shape(std::vector<size_t> shape) {

  /*
    if (thresholds_.size()) {
      for (const auto &threshold : thresholds_) {
        if (shape.at(1) > threshold.first) {
          kernel_x_shape_[1] = threshold.second;
          break;
        }
      }
    }
    kernel_x_shape_[0] = 32;
    kernel_x_shape_[2] = 128;
  */

  kernel_x_shape_[0] = shape.at(0);
  kernel_x_shape_[1] = shape.at(1);
  kernel_x_shape_[2] = shape.at(2);

  if (load_xrt_) {
    instr_bo_key_ = get_instr_key(txn_fname_prefix_, kernel_x_shape_[0],
                                  kernel_x_shape_[1], kernel_x_shape_[2]);
  }
}

template <typename LhsT, typename TrigT, typename OutT>
void mha_rope<LhsT, TrigT, OutT>::setup_instr_init() {}

template <typename LhsT, typename TrigT, typename OutT>
void mha_rope<LhsT, TrigT, OutT>::setup_instr_registry(
    const std::map<std::string, std::any> &attr) {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  std::vector<std::tuple<int, int, int>> supported_shapes;
  std::map<int, int, std::greater<int>> m_shape_list;

  if (attr.find("shapes") != attr.end()) {
    auto shapes = std::any_cast<std::vector<std::vector<int>>>(
        attr.find("shapes")->second);
    for (auto sh : shapes) {
      supported_shapes.push_back(
          std::tuple<int, int, int>{sh[0], sh[1], sh[2]});
    }
  } else {
    supported_shapes = default_shapes_.find(txn_fname_prefix_)->second;
  }

  for (auto &sh : supported_shapes) {
    m_shape_list.emplace(std::get<0>(sh), 1);

    auto key = get_instr_key(txn_fname_prefix_, std::get<0>(sh),
                             std::get<1>(sh), std::get<2>(sh));
    instructions.push_back(std::make_pair(key, false));
  }
  if (m_shape_list.size()) {
    if (m_shape_list.size() > 1) {
      int i = 0;
      for (auto iter = m_shape_list.begin(); iter != m_shape_list.end();
           iter++) {
        if (i == (m_shape_list.size() - 1)) {
          break;
        }
        int val = std::next(iter)->first;
        thresholds_.push_back(std::make_pair(val, iter->first));
        i++;
      }
    }
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
  xrt_ctx_->get_registry().add_layer_params(layer_params);
}

template <typename LhsT, typename TrigT, typename OutT>
mha_rope<LhsT, TrigT, OutT>::mha_rope(
    const std::string &operand_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr)
    : load_xrt_(load_xrt) {
  if (operand_dtype != "bfloat16") {
    throw std::runtime_error(
        "mharope only supports homogeneous bfloat16 data type "
        "for activation, trig. matrices and result");
  }
  operand_dtype_ = operand_dtype;
  operand_dtype_size_ = sizeof(LhsT);
  txnbin_operand_header = {{"bfloat16", "a16"}};

  op_version_ = "v1";
  if (attr.find("op_version") != attr.end()) {
    op_version_ = std::any_cast<std::string>(attr.find("op_version")->second);
    if (op_version_ != "v1" && op_version_ != "v2" && op_version_ != "flat") {
      throw std::runtime_error("The selected op version does not exist");
    }
  }

  if (attr.find("transpose") != attr.end()) {
    std::string transpose_str =
        std::any_cast<std::string>(attr.find("transpose")->second);
    if (transpose_attr.find(transpose_str) != transpose_attr.end()) {
      transpose_ = transpose_attr.at(transpose_str);
    }
  }

  if (attr.find("model_name") != attr.end()) {
    std::string model_str =
        std::any_cast<std::string>(attr.find("model_name")->second);
    model_ = model_string_attr.at(model_str);
  }

  txn_fname_prefix_ = "mharope_" + op_version_ + model_ +
                      transpose_txn_suffix.at(transpose_) + "_" +
                      txnbin_operand_header.at(operand_dtype_);

  default_shapes_[txn_fname_prefix_] = std::vector<std::tuple<int, int, int>>();

  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 4096, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 3072, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 2048, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 1920, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 1792, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 1664, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 1536, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 1408, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 1280, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 1152, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 1024, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 896, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 768, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 640, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 512, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 384, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 256, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 128, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 64, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 1, 128));

  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 4096, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 3072, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 2048, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1920, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1792, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1664, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1536, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1408, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1280, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1152, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1024, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 896, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 768, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 640, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 512, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 384, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 256, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 128, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 8, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1, 128));

  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 4096, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 3072, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 2048, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1920, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1792, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1664, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1536, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1408, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1280, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1152, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1024, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 896, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 768, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 640, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 512, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 384, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 256, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 8, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1, 128));

  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 4096, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 3072, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 2048, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1920, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1792, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1664, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1536, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1408, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1280, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1152, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1024, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 896, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 768, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 640, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 512, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 384, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 256, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 128, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 8, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1, 96));

  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 4096, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 3072, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 2048, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1920, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1792, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1664, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1536, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1408, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1280, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1152, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1024, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 896, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 768, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 640, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 512, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 384, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 256, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 8, 96));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1, 96));

  // new shapes for llama-3.2
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 4096, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 3072, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 2048, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1920, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1792, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1664, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1536, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1408, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1280, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1152, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1024, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 896, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 768, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 640, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 512, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 384, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 256, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 8, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1, 64));

  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 4096, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 3072, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 2048, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1920, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1792, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1664, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1536, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1408, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1280, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1152, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1024, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 896, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 768, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 640, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 512, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 384, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 256, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 128, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 8, 64));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(8, 1, 64));

  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 4096, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 3072, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 2048, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 1920, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 1792, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 1664, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 1536, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 1408, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 1280, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 1152, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 1024, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 896, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 768, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 640, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 512, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 384, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 256, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 128, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 8, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(24, 1, 128));

  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(4, 1, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(4, 64, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(4, 128, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(4, 256, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(4, 512, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(4, 1024, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(4, 2048, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(4, 3072, 128));

  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(12, 1, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(12, 64, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(12, 128, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(12, 256, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(12, 512, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(12, 1024, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(12, 2048, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(12, 3072, 128));

  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(28, 1, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(28, 64, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(28, 128, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(28, 256, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(28, 512, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(28, 1024, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(28, 2048, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(28, 3072, 128));

  mha_rope_id_ = mha_rope_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = getXCLBinName(op_version_);

  // use the max shape to initialize instr and data BOs

  kernel_x_shape_[0] = 32;
  kernel_x_shape_[1] = 4096;
  kernel_x_shape_[2] = 128;
  operand_size_in_bytes_ = kernel_x_shape_[0] * kernel_x_shape_[1] *
                           kernel_x_shape_[2] * operand_dtype_size_;
  trig_size_in_bytes_ =
      (model_ == "_glm")
          ? kernel_x_shape_[1] * kernel_x_shape_[2] * operand_dtype_size_
          : 2 * kernel_x_shape_[1] * kernel_x_shape_[2] * operand_dtype_size_;

  if (load_xrt) {

    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(
        XCLBIN_FNAME, 0, {},
        XclbinContainer::getInstance().get_xclbin_content(XCLBIN_FNAME));

    if (op_version_ == "v1") {
      std::call_once(instr_reg_v1_flag_,
                     [this, &attr]() { setup_instr_init(); });
    } else {
      std::call_once(instr_reg_flag_, [this, &attr]() { setup_instr_init(); });
    }
    setup_instr_registry(attr);
    if (attr.find("skip_create_input") == attr.end()) {
      a_bo_ =
          xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
      b_bo_ =
          xrt::bo(xrt_ctx_->get_device(), trig_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    }
    if (attr.find("skip_create_output") == attr.end()) {
      c_bo_ =
          xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    }
  }

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  num_run_aie_ = 0;

  std::call_once(logger_flag_, []() {
    std::string header = "mha_rope_id Batch M N Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "Trig_copy_time(ns) Trig_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[mharope] ID: " + std::to_string(mha_rope_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (operand_dtype, b_dtype, c_dtype): (" + operand_dtype_ +
                    ", " + operand_dtype_ + ", " + operand_dtype_ + ")");
}

template <typename LhsT, typename TrigT, typename OutT>
std::vector<xrt::bo> mha_rope<LhsT, TrigT, OutT>::get_inputs() {
  return {a_bo_, b_bo_};
}

template <typename LhsT, typename TrigT, typename OutT>
std::vector<xrt::bo> mha_rope<LhsT, TrigT, OutT>::get_outputs() {
  return {c_bo_};
}

template <typename LhsT, typename TrigT, typename OutT>
void mha_rope<LhsT, TrigT, OutT>::set_params(const std::string &model_name,
                                             std::vector<size_t> input_shape) {

  if (kernel_x_shape_[0] != input_shape[0] ||
      kernel_x_shape_[1] != input_shape[1] ||
      kernel_x_shape_[2] != input_shape[2]) {
    kernel_x_shape_[0] = input_shape[0];
    kernel_x_shape_[1] = input_shape[1];
    kernel_x_shape_[2] = input_shape[2];
    operand_size_in_bytes_ = kernel_x_shape_[0] * kernel_x_shape_[1] *
                             kernel_x_shape_[2] * operand_dtype_size_;
    trig_size_in_bytes_ =
        (model_ == "_glm")
            ? kernel_x_shape_[1] * kernel_x_shape_[2] * operand_dtype_size_
            : 2 * kernel_x_shape_[1] * kernel_x_shape_[2] * operand_dtype_size_;

    instr_bo_key_ = get_instr_key(txn_fname_prefix_, input_shape[0],
                                  input_shape[1], input_shape[2]);
  }
  return;
}

template <typename LhsT, typename TrigT, typename OutT>
void mha_rope<LhsT, TrigT, OutT>::execute(std::vector<xrt::bo> &input,
                                          std::vector<xrt::bo> &output,
                                          bool wait, int64_t offset) {
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key_);
  auto instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, input[0], input[1],
                                            output[0], 0, 0, wait, false);
  return;
}

template <typename LhsT, typename TrigT, typename OutT>
void mha_rope<LhsT, TrigT, OutT>::execute(std::vector<Tensor> &input,
                                          std::vector<Tensor> &output) {

  // The first data is a and second data is b
  LhsT *a = (LhsT *)input.at(0).data;
  TrigT *b = (TrigT *)input.at(1).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  num_run_aie_ = 0;

  auto exec_start = GET_ELAPSED_TIME_NS();

  if (!isSupportedShape(input.at(0))) {
    throw std::runtime_error("Unsupported shape for mharope");
  }

  set_kernel_shape(input.at(0).shape);

  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  LhsT *a_bo_map = a_bo_.map<LhsT *>();
  memcpy((void *)a_bo_map, (void *)a, operand_size_in_bytes_);
  auto a_copy_stop = GET_ELAPSED_TIME_NS();
  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  // b_bo copy
  auto b_copy_start = GET_ELAPSED_TIME_NS();
  TrigT *b_bo_map = b_bo_.map<TrigT *>();
  memcpy((void *)b_bo_map, (void *)b, trig_size_in_bytes_);
  auto b_copy_stop = GET_ELAPSED_TIME_NS();
  b_copy_time_ = static_cast<int64_t>(b_copy_stop - b_copy_start);

  // b_bo sync
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();
  b_sync_time_ = static_cast<int64_t>(b_sync_stop - b_sync_start);

  // launch the kernel

  auto run_aie_start = GET_ELAPSED_TIME_NS();
  std::vector<xrt::bo> inputs = {a_bo_, b_bo_};
  std::vector<xrt::bo> outputs = {c_bo_};
  execute(inputs, outputs);
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
  memcpy((void *)aie_out, (void *)c_bo_map, operand_size_in_bytes_);
  auto c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = static_cast<int64_t>(c_copy_stop - c_copy_start);
  auto exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(mha_rope_id_) + " " + std::to_string(kernel_x_shape_[0]) +
      " " + std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_x_shape_[2]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(b_copy_time_) + " " + std::to_string(b_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename LhsT, typename TrigT, typename OutT>
const std::vector<uint8_t> mha_rope<LhsT, TrigT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  const auto [B, M, K] = extract_BMK(input.at(0));
  std::string txn_key = get_instr_key(txn_fname_prefix_, B, M, K);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename TrigT, typename OutT>
const std::vector<uint8_t> mha_rope<InT, TrigT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return {};
}

template <typename LhsT, typename TrigT, typename OutT>
std::vector<OpArgMap> mha_rope<LhsT, TrigT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  const auto shape_operand = extract_BMK(input.at(0));
  const auto shape_trig = extract_BMK(input.at(1));
  const auto shape_result = extract_BMK(output.at(0));

  if ((shape_operand != shape_result)) {
    throw std::runtime_error("mismatch shape of activation and result not "
                             "supported for mharope\n");
  }
  if (std::get<1>(shape_result) != std::get<1>(shape_trig) ||
      std::get<2>(shape_result) != std::get<2>(shape_trig)) {
    throw std::runtime_error(
        "Mismatched shape of trig. matrices and activation/result "
        "not supported for mharope");
  }

  const auto num_elem_operand =
      utils::running_product_with_skips(utils::tuple_to_vector(shape_operand));
  const auto num_elem_trig =
      utils::running_product_with_skips(utils::tuple_to_vector(shape_trig));
  size_t input_1_bo_size = (num_elem_operand * sizeof(LhsT));
  size_t input_2_bo_size = (num_elem_trig * sizeof(TrigT));
  size_t output_bo_size = (num_elem_operand * sizeof(OutT));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::INPUT, 1, 1, 0, input_2_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, output_bo_size},
  };
  return arg_map;
}

template class mha_rope<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
