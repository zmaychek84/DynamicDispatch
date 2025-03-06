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
#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include <utils/dpu_mdata.hpp>

#include "ops/ops_common/lrn_matrix.hpp"
#include "txn/txn_utils.hpp"
#include "utils/ctrl_pkt_utils.hpp"
#include <ops/groupnorm/groupnorm.hpp>
#include <ops/op_interface.hpp>
#include <ops/ops_common/ctrlpkt.hpp>
#include <txn_container.hpp>
#include <txn_helper/txn_helper.hpp>
#include <utils/instruction_registry.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>
#include <xrt_context/xrt_context.hpp>

#include "txn/txn_utils.hpp"

// AIE Driver header
#include <xaiengine.h>

using namespace lrn_matrix;

namespace ryzenai {
static std::tuple<size_t, size_t, size_t>
extract_MK(const std::vector<Tensor> &inputs) {
  size_t M = 0;
  size_t N = 0;
  size_t G = 0;
  if (inputs.at(0).shape.size() == 4) { // NHWC format mzdk5
    M = inputs.at(0).shape.at(1) * inputs.at(0).shape.at(2); // H*W
    N = inputs.at(0).shape.at(3);
    G = inputs.at(0).shape.at(3);
  }

  return std::make_tuple(M, N, G);
}

template <typename InT, typename WtT, typename OutT>
std::tuple<size_t, size_t>
groupnorm<InT, WtT, OutT>::map_padded_shape(size_t M, size_t N) const {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<std::tuple<int, int, int>> &supported_shapes = iter->second;
  size_t Mo = M;
  size_t No = N;
  size_t fidx = 0;
  bool f_found = false;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes[i];
    if (M == std::get<0>(mat) && N == std::get<1>(mat)) {
      fidx = i;
      f_found = true;
      break;
    }
  }
  if (f_found) {
    iter = default_shapes_.find(txn_fname_prefix_);
    const std::vector<std::tuple<int, int, int>> &actual_shapes = iter->second;
    auto mat = actual_shapes[fidx];
    Mo = std::get<0>(mat);
    No = std::get<1>(mat);
  } else {
    throw std::runtime_error("Cannot find the shape");
  }
  return std::make_tuple(Mo, No);
}

template <typename InT, typename WtT, typename OutT>
std::once_flag groupnorm<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t groupnorm<InT, WtT, OutT>::groupnorm_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag groupnorm<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
void groupnorm<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
std::string groupnorm<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                                     size_t m, size_t k,
                                                     size_t g) const {
  // return prefix + "_" + std::to_string(m) + "_" + std::to_string(k) + ".bin";
  return "gpn_" + prefix + "_" + std::to_string(m) + "_" + std::to_string(k) +
         "_" + std::to_string(g);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> groupnorm<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, G] = extract_MK(input);
  auto [Mo, Ko] = map_padded_shape(M, K);
  auto Go = Ko;
  std::string txn_key = get_instr_key(txn_fname_prefix_, Mo, Ko, Go);
  Transaction &txn = Transaction::getInstance();
  std::vector<uint8_t> data = txn.get_txn_bvec(txn_key);

  // assume input.at(3) is qdq_params
  int32_t *qdq_param;
  if (is_generic_fusion) {
    float *out_scale = (float *)input.at(7).data;
    uint16_t *out_zero_point = (uint16_t *)input.at(8).data;
    float input_scale =
        (std::any_cast<std::vector<float>>(attr.at("input_scale")))[0];
    uint16_t input_zp = static_cast<uint16_t>(
        (std::any_cast<std::vector<float>>(attr.at("input_zp")))[0]);
    std::tie(qdq_tensor[0], qdq_tensor[1]) =
        OpsFusion::coeffs::calc_lrn_coeff((1 / (*out_scale)), *out_zero_point);
    std::tie(qdq_tensor[3], qdq_tensor[4]) =
        OpsFusion::coeffs::calc_lrn_coeff(input_scale, input_zp);
    qdq_tensor[2] = 1;
    qdq_tensor[5] = 1;

    // this logic was present in vaip mzdk5 pass
    std::vector<std::string> in_dtypes =
        std::any_cast<std::vector<std::string>>(attr.at("in_dtypes"));
    std::vector<std::string> out_dtypes =
        std::any_cast<std::vector<std::string>>(attr.at("out_dtypes"));
    if (in_dtypes[0] == "bfloat16") {
      qdq_tensor[5] = 0;
      qdq_tensor[4] = 0;
    } else {
      qdq_tensor[5] = 1;
    }

    if (out_dtypes[0] == "bfloat16") {
      qdq_tensor[2] = 0;
    } else {
      qdq_tensor[2] = 1;
    }

    qdq_param = qdq_tensor;
  } else {
    qdq_param = (int32_t *)input.at(3).data;
  }
  uint32_t zp = uint16_t(qdq_param[lrn_qdq_ifm_zp_idx]);
  uint32_t pad_val = zp | (zp << 16);
  std::vector<uint8_t> txn_w_pad;
  if (design_param_.find("4x4") != std::string::npos) { // mzdk5 4x4 design
    txn_w_pad = prepend_mtile_const_pad_txn(data, pad_val, 6, 4);
  } else {
    txn_w_pad = prepend_mtile_const_pad_txn(data, pad_val, 6, 2);
  }
  return txn_w_pad;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> groupnorm<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, G] = extract_MK(input);
  auto [Mo, Ko] = map_padded_shape(M, K);
  auto Go = Ko;
  // TODO: Add check to validate tensor shapes
  std::string param_key =
      get_instr_key(param_fname_prefix_, Mo, Ko, Go) + "_param";
  // std::cout << "Super kernel params name : " << fname << std::endl;
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(param_key);
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> groupnorm<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  // input --> [input, gamma, beta, output]
  // Check if IO buffers have batch.
  auto [M, N, G] = extract_MK(input);
  auto [Mo, No] = map_padded_shape(M, N);

  int64_t gamma_dim, beta_dim;
  if (is_generic_fusion) {
    gamma_dim = input.at(1).shape.at(0);
    beta_dim = input.at(4).shape.at(0);
  } else {
    gamma_dim = input.at(1).shape.at(0);
    beta_dim = input.at(2).shape.at(0);
  }

  int repeat_factor;
  if (design_param_.find("4x4") != std::string::npos) {     // mzdk5 4x4 design
    repeat_factor = (int)(std::lcm(N / 32, 16) / (N / 32)); // 32 groups
  } else {
    repeat_factor = 8;
  }
  size_t const_params_bo_size =
      ((gamma_dim + beta_dim) * sizeof(WtT) * repeat_factor +
       lrn_matrix::QDQparam_size * sizeof(int32_t));
  size_t input_bo_size = (Mo * No * sizeof(InT));
  size_t output_bo_size = (Mo * No * sizeof(OutT));
  size_t super_kernel_size = get_super_kernel_params(input, output).size();
  size_t ctrl_pkt_size = get_ctrl_pkts(input, output).size();

  size_t output_idx = is_generic_fusion ? 9 : 4;

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, output_idx, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size},
      {OpArgMap::OpArgType::CTRL_PKT_BIN, 4, 0, 0, ctrl_pkt_size}};
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
void groupnorm<InT, WtT, OutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;

  for (const auto &[mkey, value] : default_shapes_) {
    auto iter = default_shapes_.find(mkey);
    std::vector<std::tuple<int, int, int>> &supported_shapes = iter->second;
    for (size_t i = 0; i < supported_shapes.size(); i++) {
      auto mat = supported_shapes[i];
      auto key = get_instr_key(mkey, std::get<0>(mat), std::get<1>(mat),
                               std::get<2>(mat));
      auto param_key = get_instr_key(mkey, std::get<0>(mat), std::get<1>(mat),
                                     std::get<2>(mat)) +
                       "_param";
      instructions.push_back(std::make_pair(key, false));
      layer_params.push_back(std::make_pair(param_key, false));
    }
  }

  xrt_ctx_->get_registry().add_instructions(instructions);
  xrt_ctx_->get_registry().add_layer_params(layer_params);
}

template <typename InT, typename WtT, typename OutT>
groupnorm<InT, WtT, OutT>::groupnorm(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"bfloat16", "abf16"}, {"uint16", "a16"}};

  txnbin_acc_header = {{"bfloat16", "accbf16"}, {"uint16", "acc16"}};

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_["gpn_4x4_abf16accbf16"] =
      std::vector<std::tuple<int, int, int>>();
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(64, 1280, 1280));
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(64, 2560, 2560));
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(256, 1280, 1280));
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(256, 2560, 2560));
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(256, 1920, 1920));
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(1024, 640, 640));
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(4096, 320, 320));
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(1024, 1920, 1920));
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(1024, 960, 960));
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(1024, 1280, 1280));
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(4096, 640, 640));
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(4096, 960, 960));
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(1024, 320, 320));
  default_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(256, 640, 640));

  default_shapes_["gpn_4x4_a16accbf16"] =
      std::vector<std::tuple<int, int, int>>();
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(64, 1280, 1280));
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(64, 2560, 2560));
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(256, 1280, 1280));
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(256, 2560, 2560));
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(256, 1920, 1920));
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(1024, 640, 640));
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(4096, 320, 320));
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(1024, 1920, 1920));
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(1024, 960, 960));
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(1024, 1280, 1280));
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(4096, 640, 640));
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(4096, 960, 960));
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(1024, 320, 320));
  default_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(256, 640, 640));

  // raw shape is the actual shape from ONNX, sequence needs to match with
  // default shape
  raw_shapes_["gpn_4x4_abf16accbf16"] =
      std::vector<std::tuple<int, int, int>>();
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(64, 1280, 1280));
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(64, 2560, 2560));
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(256, 1280, 1280));
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(256, 2560, 2560));
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(256, 1920, 1920));
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(1024, 640, 640));
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(4096, 320, 320));
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(1024, 1920, 1920));
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(1024, 960, 960));
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(1024, 1280, 1280));
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(4096, 640, 640));
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(4096, 960, 960));
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(
      std::make_tuple(1024, 320, 320));
  raw_shapes_["gpn_4x4_abf16accbf16"].push_back(std::make_tuple(256, 640, 640));

  raw_shapes_["gpn_4x4_a16accbf16"] = std::vector<std::tuple<int, int, int>>();
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(std::make_tuple(64, 1280, 1280));
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(std::make_tuple(64, 2560, 2560));
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(std::make_tuple(256, 1280, 1280));
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(std::make_tuple(256, 2560, 2560));
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(std::make_tuple(256, 1920, 1920));
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(std::make_tuple(1024, 640, 640));
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(std::make_tuple(4096, 320, 320));
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(1024, 1920, 1920));
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(std::make_tuple(1024, 960, 960));
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(
      std::make_tuple(1024, 1280, 1280));
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(std::make_tuple(4096, 640, 640));
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(std::make_tuple(4096, 960, 960));
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(std::make_tuple(1024, 320, 320));
  raw_shapes_["gpn_4x4_a16accbf16"].push_back(std::make_tuple(256, 640, 640));

  is_generic_fusion = OpsFusion::check_generic_fusion(attr);

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  groupnorm_id_ = groupnorm_count++;
  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dd_base_dir() + ryzenai::mdsqr_A8W8_QDQ_XCLBIN_PATH;

  if (c_dtype_ == "uint16") {
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

  txn_fname_prefix_ = "gpn_4x2_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_acc_header.at(c_dtype_);

  param_fname_prefix_ = "gpn_4x2_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_acc_header.at(c_dtype_);

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    txn_fname_prefix_ = "gpn_4x4_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_acc_header.at(c_dtype_);

    param_fname_prefix_ = "gpn_4x4_" + txnbin_a_header.at(a_dtype_) +
                          txnbin_acc_header.at(c_dtype_);
  }

  KERNEL_M_MAX = 32;

  w_shape_[0] = KERNEL_M_MAX;

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
  is_ctrl_pkt_ = 0;

  std::call_once(logger_flag_, []() {
    std::string header = "groupnorm_id M K N kernel_m kernel_k kernel_n Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[GEMM] ID: " + std::to_string(groupnorm_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype + ", " +
                    b_dtype + ", " + c_dtype + ")");
}

template <typename InT, typename WtT, typename OutT>
void groupnorm<InT, WtT, OutT>::set_params(const std::string &model_name,
                                           std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;
  if (model_name == "mzdk5") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mzdk5_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "4x4mzdk5") {
    is_ctrl_pkt_ = 1;
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                   ryzenai::mzdk54x4_A16W8_QDQ_XCLBIN_PATH;
    ;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }

  // for memory allocation
  w_shape_[0] = input_shape.at(0);
  w_shape_[1] = input_shape.at(1);
  w_shape_[2] = 32;

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void groupnorm<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Groupnorm initialize_const_params(ptr) ...");

  // DD_THROW_IF((const_params.size() != 3) ||
  //                  (const_params.at(0).shape.size() != 1) ||
  //                  (const_params.at(1).shape.size() != 1),
  //              OpsFusion::dd_format("Unsupported const spec for
  //              Groupnorm\n") +
  //                  OpsFusion::dd_format(
  //                      "(Details : #const params == 3 ({}), Const param1 dim
  //                      "
  //                      "== 1 ({}), Const param2 dim == 1 ({})",
  //                      const_params.size(), const_params.at(0).shape.size(),
  //                      const_params.at(1).shape.size()));

  int32_t *qdq_params;
  WtT *gamma;
  WtT *beta;
  std::vector<uint16_t> gamma_bf16, beta_bf16;
  std::vector<size_t> shape;

  if (is_generic_fusion) {

    std::vector<uint16_t> mul_const_vec =
        OpsFusion::get_tensor_as_uint16_t_vec(const_params.at(0));
    float *mul_const_scale = (float *)const_params.at(1).data;
    uint16_t *mul_const_zp = (uint16_t *)const_params.at(2).data;
    gamma_bf16 = OpsFusion::coeffs::dq_vec_to_bf16(
        mul_const_vec, *mul_const_scale, *mul_const_zp);
    gamma = reinterpret_cast<WtT *>(gamma_bf16.data());
    shape = const_params.at(0).shape;

    std::vector<uint16_t> add_const_vec =
        OpsFusion::get_tensor_as_uint16_t_vec(const_params.at(3));
    float *add_const_scale = (float *)const_params.at(4).data;
    uint16_t *add_const_zp = (uint16_t *)const_params.at(5).data;
    beta_bf16 = OpsFusion::coeffs::dq_vec_to_bf16(
        add_const_vec, *add_const_scale, *add_const_zp);
    beta = reinterpret_cast<WtT *>(beta_bf16.data());

    float *out_scale = (float *)const_params.at(6).data;
    uint16_t *out_zero_point = (uint16_t *)const_params.at(7).data;
    float input_scale =
        (std::any_cast<std::vector<float>>(attr.at("input_scale")))[0];
    uint16_t input_zp = static_cast<uint16_t>(
        (std::any_cast<std::vector<float>>(attr.at("input_zp")))[0]);
    std::tie(qdq_tensor[0], qdq_tensor[1]) =
        OpsFusion::coeffs::calc_lrn_coeff((1 / (*out_scale)), *out_zero_point);
    std::tie(qdq_tensor[3], qdq_tensor[4]) =
        OpsFusion::coeffs::calc_lrn_coeff(input_scale, input_zp);
    qdq_tensor[2] = 1;
    qdq_tensor[5] = 1;

    // this logic was present in vaip mzdk5 pass
    std::vector<std::string> in_dtypes =
        std::any_cast<std::vector<std::string>>(attr.at("in_dtypes"));
    std::vector<std::string> out_dtypes =
        std::any_cast<std::vector<std::string>>(attr.at("out_dtypes"));
    if (in_dtypes[0] == "bfloat16") {
      qdq_tensor[5] = 0;
      qdq_tensor[4] = 0;
    } else {
      qdq_tensor[5] = 1;
    }

    if (out_dtypes[0] == "bfloat16") {
      qdq_tensor[2] = 0;
    } else {
      qdq_tensor[2] = 1;
    }

    qdq_params = qdq_tensor;

  } else {

    const int gamma_idx = 0, beta_idx = 1, qdq_params_idx = 2;
    // The first data is Gamma
    gamma = (WtT *)const_params.at(gamma_idx).data;
    shape = const_params.at(gamma_idx).shape;
    beta = (WtT *)const_params.at(beta_idx).data;
    qdq_params = (int32_t *)const_params.at(qdq_params_idx).data;
  }
  // Group Norm kernel requires Gamma and Beta to be repeated few times
  int repeat_factor;
  if (design_param_.find("4x4") != std::string::npos) { // mzdk5 4x4 design
    repeat_factor =
        (int)(std::lcm(shape[0] / 32, 16) / (shape[0] / 32)); // 32 groups
  } else {
    repeat_factor = 8;
  }
  // SW convert scale to 1/scale and bfloat16 for Q
  auto count = repeat_factor * (int)shape[0];
  auto offset = lrn_matrix::BiasVector<WtT, 1>::size(count);
  auto qdq_params_size = lrn_matrix::QDQparam_size * sizeof(int32_t);
  {
    auto buffer = io.get_buffer(0, offset);
    lrn_matrix::BiasVector<WtT, 1> bias(count, buffer->ptr());
    for (size_t k = 0; k < count; ++k) {
      bias.gamma((int)k) = gamma[k % shape[0]];
      bias.beta((int)k) = beta[k % shape[0]];
    }
  }
  io.write(offset, (void *)qdq_params, qdq_params_size);
  const_pad_ = uint16_t(qdq_params[lrn_qdq_ifm_zp_idx]);
  RYZENAI_LOG_TRACE("Groupnorm initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void groupnorm<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  // Check the number of inputs
  if (!is_generic_fusion && const_params.size() != 3) {
    throw std::runtime_error("GPN IPU Wrapper expect to have three constants.");
  }

  std::vector<size_t> shape;
  if (is_generic_fusion) {
    shape = const_params.at(0).shape;
  } else {
    const int gamma_idx = 0, beta_idx = 1;
    shape = const_params.at(gamma_idx).shape;
  }

  // Init the BO size
  int repeat_factor;
  if (design_param_.find("4x4") != std::string::npos) { // mzdk5 4x4 design
    repeat_factor =
        (int)(std::lcm(shape[0] / 32, 16) / (shape[0] / 32)); // 32 groups
  } else {
    repeat_factor = 8;
  }
  kernel_x_shape_[0] = w_shape_[0];
  kernel_x_shape_[1] = w_shape_[1];
  kernel_y_shape_[0] =
      repeat_factor * 2; // Bo has Gamma and Beta repeated few times
  kernel_y_shape_[1] = shape[0];
  kernel_z_shape_[0] = w_shape_[0];
  kernel_z_shape_[1] = w_shape_[1];

  // Create input/output BOs
  const size_t B_BO_SIZE =
      (kernel_y_shape_[0] * kernel_y_shape_[1] * b_dtype_size_) +
      lrn_matrix::QDQparam_size * sizeof(int32_t);
  const size_t A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_);
  const size_t C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_);
  RYZENAI_LOG_TRACE("GPN: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));

  auto M = kernel_x_shape_[0];
  auto K = kernel_x_shape_[1];
  auto [Mo, Ko] = map_padded_shape(M, K);
  auto Go = Ko;

  if (is_ctrl_pkt_) {
    // Based on the mapped_shape to get the meta json file
    std::vector<uint8_t> json_data;
    try {
      auto json_key =
          get_instr_key(param_fname_prefix_, Mo, Ko, Go) + "_ctrl_meta";
      Transaction &txn = Transaction::getInstance();
      json_data = txn.get_txn_bvec(json_key);
    } catch (...) {
      is_ctrl_pkt_ = 0;
    }

    if (is_ctrl_pkt_) {
      std::cout << "ctrlpkt patching" << std::endl;
      RYZENAI_LOG_TRACE("groupnorm patch ctrlpkt ... START");
      // get param_bo address
      auto param_bo_key =
          get_instr_key(param_fname_prefix_, Mo, Ko, Go) + "_param";
      const xrt::bo &param_bo =
          xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;

      // Get ctrl pkt patch info from json
      std::vector<CtrlPktPatchInfo> ctrlpkt_info;
      ctrlpkt_info = json_str_to_ctrlpkt_patch_info(json_data);

      // Get the ctrl pkt
      auto ctrl_bo_key =
          get_instr_key(param_fname_prefix_, Mo, Ko, Go) + "_ctrl";
      std::string ctrl_params =
          Transaction::getInstance().get_txn_str(ctrl_bo_key);
      std::vector<char> ctrl_buffer(ctrl_params.begin(), ctrl_params.end());

      // ctrl pkt patch
      std::vector<char> ctrl_pkt_new;
      std::vector<uint64_t> buffer_addrs = {
          uint64_t(c_bo_.address() + DDR_AIE_ADDR_OFFSET),
          uint64_t(a_bo_.address() + DDR_AIE_ADDR_OFFSET),
          uint64_t(b_bo_.address() + DDR_AIE_ADDR_OFFSET),
          uint64_t(param_bo.address() + DDR_AIE_ADDR_OFFSET)};
      ctrl_pkt_new = patch_ctrl_bin(ctrl_buffer, ctrlpkt_info, buffer_addrs);

      size_t ctrl_bo_words = ctrl_pkt_new.size();
      ctrl_bo_ =
          xrt::bo(xrt_ctx_->get_device(), ctrl_bo_words, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
      ctrl_bo_.write(ctrl_pkt_new.data());
      ctrl_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      RYZENAI_LOG_TRACE("GPN patch ctrlpkt ... DONE");
    }
  }

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
std::vector<uint8_t> groupnorm<InT, WtT, OutT>::get_ctrl_pkts(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  auto [M, K, G] = extract_MK(input);
  auto [Mo, Ko] = map_padded_shape(M, K);
  auto Go = Ko;
  // TODO: Add check to validate tensor shapes
  std::string ctrl_key =
      get_instr_key(param_fname_prefix_, Mo, Ko, Go) + "_ctrl";
  // std::cout << "Super kernel params name : " << fname << std::endl;
  try {
    Transaction &txn = Transaction::getInstance();
    return txn.get_txn_bvec(ctrl_key);
  } catch (...) {
    return {};
  }
}

template <typename InT, typename WtT, typename OutT>
std::vector<CtrlPktPatchInfo>
groupnorm<InT, WtT, OutT>::get_ctrl_pkt_patch_info(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, G] = extract_MK(input);
  auto [Mo, Ko] = map_padded_shape(M, K);
  auto Go = Ko;
  // TODO: Add check to validate tensor shapes
  try {
    auto ctrl_pkt_meta =
        get_instr_key(param_fname_prefix_, Mo, Ko, Go) + "_ctrl_meta";
    Transaction &txn = Transaction::getInstance();
    return json_str_to_ctrlpkt_patch_info(txn.get_txn_bvec(ctrl_pkt_meta));
  } catch (...) {
    // throw std::runtime_error(
    //     "groupnorm : Can not file the ctrl_meta.json file");
    return {};
  }
}

template <typename InT, typename WtT, typename OutT>
void groupnorm<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                        std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 1) {
    throw std::runtime_error("GPN IPU Wrapper expect to have one input.");
  }
  const int a_idx = 0;
  // The first data is a
  InT *a = (InT *)input.at(a_idx).data;

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

  c_shape_[0] = a_shape_[0];
  c_shape_[1] = a_shape_[1];

  kernel_x_rows = a_shape_[0];

  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  size_t a_size = a_shape_[0] * a_shape_[1] * sizeof(InT);
  RYZENAI_LOG_TRACE("GPN: a_size:" + std::to_string(a_size));
  memcpy((void *)a_bo_map, (void *)a, a_size);
  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  // prepare inst_bo and param_bo
  auto M = a_shape_[0];
  auto K = a_shape_[1];
  auto [Mo, Ko] = map_padded_shape(M, K);
  auto GB = Ko;
  auto instr_bo_key = get_instr_key(txn_fname_prefix_, Mo, Ko, GB);
  auto param_bo_key = get_instr_key(param_fname_prefix_, Mo, Ko, GB) + "_param";
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  const xrt::bo &param_bo =
      xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;
  size_t instr_bo_words = instr_bo.size() / sizeof(int);

  // Ignore instruction key from registry since const padding instruction is
  // required.

  Transaction &txn = Transaction::getInstance();
  std::vector<uint8_t> data = txn.get_txn_bvec(instr_bo_key);

  uint32_t zp = uint16_t(const_pad_);
  uint32_t pad_val = zp | (zp << 16);
  std::vector<uint8_t> txn_w_pad;
  if (design_param_.find("4x4") != std::string::npos) { // mzdk5 4x4 design
    txn_w_pad = prepend_mtile_const_pad_txn(data, pad_val, 6, 4);
  } else {
    txn_w_pad = prepend_mtile_const_pad_txn(data, pad_val, 6, 2);
  }

  auto i_buf = transaction_op(txn_w_pad);
  instr_bo_words = i_buf.get_txn_instr_size();
  xrt::bo i_bo =
      xrt::bo(xrt_ctx_->get_context(), instr_bo_words,
              xrt::bo::flags::cacheable, xrt_ctx_->get_kernel().group_id(1));
  i_bo.write(i_buf.get_txn_op().data());
  i_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto i_bo_words = i_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();

  // launch the kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, i_bo, i_bo_words, c_bo_,
                                            a_bo_, b_bo_, param_bo, ctrl_bo_,
                                            true, is_ctrl_pkt_);
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
      std::to_string(groupnorm_id_) + " " + std::to_string(a_shape_[0]) + " " +
      std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) + " " +
      std::to_string(kernel_x_rows) + " " + std::to_string(kernel_x_shape_[1]) +
      " " + std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template class groupnorm<int16_t, int16_t, uint16_t>;
} // namespace ryzenai
