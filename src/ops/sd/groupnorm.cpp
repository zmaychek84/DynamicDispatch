/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

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
#include <ops/op_interface.hpp>
#include <ops/sd/groupnorm.hpp>
#include <txn_container.hpp>
#include <txn_helper/txn_helper.hpp>
#include <utils/instruction_registry.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>
#include <xrt_context/xrt_context.hpp>

// AIE Driver header
#include <xaiengine.h>
using namespace lrn_matrix;

namespace ryzenai {
namespace sd {
template <typename InT, typename WtT, typename OutT>
void groupnorm<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  for (const auto &[mkey, value] : default_shapes_) {
    auto iter = default_shapes_.find(mkey);
    std::vector<std::tuple<int, int, int, int>> &supported_shapes =
        iter->second;
    for (size_t i = 0; i < supported_shapes.size(); i++) {
      auto mat = supported_shapes[i];
      auto key = get_instr_key(mkey, std::get<0>(mat), std::get<1>(mat),
                               std::get<2>(mat), std::get<3>(mat));
      instructions.push_back(std::make_pair(key, false));
    }
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
std::string groupnorm<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                                     size_t b, size_t h,
                                                     size_t w, size_t c) const {
  return prefix + "_" + std::to_string(b) + "_" + std::to_string(h) + "_" +
         std::to_string(w) + "_" + std::to_string(c);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> groupnorm<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  std::string txn_key = get_instr_key(txn_fname_prefix_, a_shape_[0],
                                      a_shape_[1], a_shape_[2], a_shape_[3]);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
groupnorm<InT, WtT, OutT>::groupnorm(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"bfloat16", "a16bf"}};
  txnbin_b_header = {{"bfloat16", "w16bf"}};
  txnbin_acc_header = {{"bfloat16", "acc16bf"}};
  /*select xclbin based on the input/output types*/
  XCLBIN_FNAME_ =
      OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\SDGroupNorm.xclbin";
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME_));
  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  txn_fname_prefix_ = sd_gn_key_ + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_[txn_fname_prefix_] =
      std::vector<std::tuple<int, int, int, int>>();
  // sd1.5
  default_shapes_[txn_fname_prefix_].push_back(
      std::make_tuple(1, 128, 128, 512));
  default_shapes_[txn_fname_prefix_].push_back(
      std::make_tuple(1, 256, 256, 256));
  default_shapes_[txn_fname_prefix_].push_back(
      std::make_tuple(1, 256, 256, 512));
  default_shapes_[txn_fname_prefix_].push_back(
      std::make_tuple(1, 512, 512, 128));
  default_shapes_[txn_fname_prefix_].push_back(
      std::make_tuple(1, 512, 512, 256));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(1, 64, 64, 512));
  default_shapes_[txn_fname_prefix_].push_back(
      std::make_tuple(2, 16, 16, 1280));
  default_shapes_[txn_fname_prefix_].push_back(
      std::make_tuple(2, 16, 16, 1920));
  default_shapes_[txn_fname_prefix_].push_back(
      std::make_tuple(2, 16, 16, 2560));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 16, 16, 640));
  default_shapes_[txn_fname_prefix_].push_back(
      std::make_tuple(2, 32, 32, 1280));
  default_shapes_[txn_fname_prefix_].push_back(
      std::make_tuple(2, 32, 32, 1920));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 32, 32, 320));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 32, 32, 640));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 32, 32, 960));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 64, 64, 320));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 64, 64, 640));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 64, 64, 960));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 8, 8, 1280));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 8, 8, 2560));

  // sd3.0
  default_shapes_[txn_fname_prefix_].push_back(
      std::make_tuple(1, 512, 512, 512));
  default_shapes_[txn_fname_prefix_].push_back(
      std::make_tuple(1, 1024, 1024, 128));
  default_shapes_[txn_fname_prefix_].push_back(
      std::make_tuple(1, 1024, 1024, 256));
  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);
  groupnorm_id_ = groupnorm_count++;
  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));

    if (input_shape_vector.size() == 4) {
      a_shape_[0] = input_shape_vector[0];
      a_shape_[1] = input_shape_vector[1];
      a_shape_[2] = input_shape_vector[2];
      a_shape_[3] = input_shape_vector[3];
    } else {
      RYZENAI_LOG_INFO(
          "Input Shape attribute does not have the expected number of "
          "elements.Number of passed : " +
          std::to_string(input_shape_vector.size()) + ", Expected:4");
    }
    RYZENAI_LOG_TRACE(
        "GroupNorm: InputShape: " + std::to_string(input_shape_vector[0]) +
        ", " + std::to_string(input_shape_vector[1]) + ", " +
        std::to_string(input_shape_vector[2]) + ", " +
        std::to_string(input_shape_vector[3]));
  } else {
    RYZENAI_LOG_INFO("Input Shape attribute not found or not of correct type.");
  }

  if (attr.count("output_shape") &&
      attr.at("output_shape").type() == typeid(std::vector<int>)) {
    const auto &output_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("output_shape"));

    if (output_shape_vector.size() == 4) {
      c_shape_[0] = output_shape_vector[0];
      c_shape_[1] = output_shape_vector[1];
      c_shape_[2] = output_shape_vector[2];
      c_shape_[3] = output_shape_vector[3];

    } else {
      RYZENAI_LOG_INFO(
          "Input Shape attribute does not have the expected number of "
          "elements.Number of passed : " +
          std::to_string(output_shape_vector.size()) + ", Expected:4");
    }
    RYZENAI_LOG_TRACE(
        "GroupNorm: OutputShape: " + std::to_string(output_shape_vector[0]) +
        ", " + std::to_string(output_shape_vector[1]) + ", " +
        std::to_string(output_shape_vector[2]) + ", " +
        std::to_string(output_shape_vector[3]));
  } else {
    RYZENAI_LOG_INFO(
        "Output Shape attribute not found or not of correct type.");
  }

  if (attr.count("wts_shape") &&
      attr.at("wts_shape").type() == typeid(std::vector<int>)) {
    const auto &wts_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("wts_shape"));

    if (wts_shape_vector.size() == 1) {
      wts_size_ = wts_shape_vector[0];
      b_bo_size_ = wts_size_ * b_dtype_size_;

    } else {
      RYZENAI_LOG_INFO(
          "Input Shape attribute does not have the expected number of "
          "elements.Number of passed : " +
          std::to_string(wts_shape_vector.size()) + ", Expected:1");
    }
    RYZENAI_LOG_TRACE("GroupNorm: weight shape: " +
                      std::to_string(wts_shape_vector[0]));
  } else {
    RYZENAI_LOG_INFO(
        "Weight Shape attribute not found or not of correct type.");
  }

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
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
    std::string header = "groupnorm_id B H W C Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[GEMM] ID: " + std::to_string(groupnorm_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME_ +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype + ", " +
                    b_dtype + ", " + c_dtype + ")");
}

template <typename InT, typename WtT, typename OutT>
void groupnorm<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Groupnorm initialize_const_params(ptr) ...");
  io.write(0, const_params.at(0).data, b_bo_size_);
  RYZENAI_LOG_TRACE("Groupnorm initialize_const_params(ptr) ... DONE");
  return;
}

template <typename InT, typename WtT, typename OutT>
void groupnorm<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {

  const size_t A_BO_SIZE =
      (a_shape_[0] * a_shape_[1] * a_shape_[2] * a_shape_[3] * a_dtype_size_);
  const size_t C_BO_SIZE =
      (c_shape_[0] * c_shape_[1] * c_shape_[2] * c_shape_[3] * c_dtype_size_);
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), b_bo_size_, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  WtT *b_bo_map = b_bo_.map<WtT *>();
  auto bo_const = BoConst(b_bo_map);
  initialize_const_params(bo_const, const_params);
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  return;
}

template <typename InT, typename WtT, typename OutT>
void groupnorm<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                        std::vector<Tensor> &output) {

  a_bo_.write(input.at(0).data);
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto instr_bo_key =
      get_instr_key(txn_fname_prefix_, size_t(a_shape_[0]), size_t(a_shape_[1]),
                    size_t(a_shape_[2]), size_t(a_shape_[3]));
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  size_t instr_bo_words = instr_bo.size() / sizeof(int);

  // launch the kernel
  auto kernel_ = xrt_ctx_->get_kernel();

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, a_bo_, b_bo_, c_bo_,
                                            0, 0, true, false);
  // sync output activation to host memory
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  c_bo_.read(output.at(0).data);
}

template <typename InT, typename WtT, typename OutT>
void groupnorm<InT, WtT, OutT>::set_params() {

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> groupnorm<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  const size_t A_BO_SIZE =
      (a_shape_[0] * a_shape_[1] * a_shape_[2] * a_shape_[3] * a_dtype_size_);
  const size_t C_BO_SIZE =
      (c_shape_[0] * c_shape_[1] * c_shape_[2] * c_shape_[3] * c_dtype_size_);

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, A_BO_SIZE},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, b_bo_size_},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, C_BO_SIZE}};
  return arg_map;
};

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

template class groupnorm<uint16_t, uint16_t, uint16_t>;
} // namespace sd
} // namespace ryzenai
