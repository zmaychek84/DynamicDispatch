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

// #include "ops/ops_common/lrn_matrix.hpp"clear
#include <ops/op_interface.hpp>
#include <ops/sd/layernorm.hpp>
#include <txn_container.hpp>
#include <txn_helper/txn_helper.hpp>
#include <utils/instruction_registry.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>
#include <xrt_context/xrt_context.hpp>

// AIE Driver header
#include <xaiengine.h>
// using namespace lrn_matrix;

namespace ryzenai {
namespace sd {

template <typename InT, typename WtT, typename OutT>
void layernorm<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  for (const auto &[mkey, value] : default_shapes_) {
    auto iter = default_shapes_.find(mkey);
    std::vector<std::tuple<int, int, int>> &supported_shapes = iter->second;
    for (size_t i = 0; i < supported_shapes.size(); i++) {
      auto mat = supported_shapes[i];
      auto key = get_instr_key(mkey, std::get<0>(mat), std::get<1>(mat),
                               std::get<2>(mat));
      instructions.push_back(std::make_pair(key, false));
    }
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
std::string layernorm<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                                     size_t b, size_t m,
                                                     size_t k) const {
  return prefix + "_" + std::to_string(b) + "_" + std::to_string(m) + "_" +
         std::to_string(k);
}

template <typename InT, typename WtT, typename OutT>
layernorm<InT, WtT, OutT>::layernorm(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {
  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;

  txnbin_a_header = {{"bfloat16", "a16bf"}};
  txnbin_b_header = {{"bfloat16", "w16bf"}};
  txnbin_acc_header = {{"bfloat16", "acc16bf"}};

  XCLBIN_FNAME_ =
      OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\SDLayerNorm.xclbin";
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME_));
  txn_fname_prefix_ = sd_lrn_key_ + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);

  default_shapes_[txn_fname_prefix_] = std::vector<std::tuple<int, int, int>>();
  // sd1.5
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 4096, 320));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 1024, 640));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 256, 1280));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 64, 1280));
  // sd3.0
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 154, 1536));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 1024, 1536));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 4096, 1536));
  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  layernorm_id_ = layernorm_count++;
  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));
    DD_ASSERT(input_shape_vector.size() == 3,
              OpsFusion::dd_format(
                  "The input shape for layernorm must have 3 dimensions, but "
                  "the given shape has {} dimensions",
                  input_shape_vector.size()));
    a_shape_[0] = input_shape_vector[0];

    a_shape_[1] = input_shape_vector[1];
    a_shape_[2] = input_shape_vector[2];
    b_shape_[0] = input_shape_vector[2];
    b_bo_size_ = b_shape_[0] * b_dtype_size_;
    b_ = input_shape_vector[0];
    m_ = input_shape_vector[1];
    k_ = input_shape_vector[2];
  }

  if (attr.count("output_shape") &&
      attr.at("output_shape").type() == typeid(std::vector<int>)) {
    const auto &output_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("output_shape"));
    DD_ASSERT(output_shape_vector.size() == 3,
              OpsFusion::dd_format(
                  "The output shape for layernorm must have 3 dimensions, but "
                  "the given shape has {} dimensions",
                  output_shape_vector.size()));
    a_shape_[0] = output_shape_vector[0];
    a_shape_[1] = output_shape_vector[1];
    a_shape_[2] = output_shape_vector[2];
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
    std::string header = "layernorm_id B M K  kernel_m kernel_k  Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[GEMM] ID: " + std::to_string(layernorm_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME_ +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype + ", " +
                    b_dtype + ", " + c_dtype + ")");
}

template <typename InT, typename WtT, typename OutT>
void layernorm<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Layernorm initialize_const_params(ptr) ...");

  DD_THROW_IF((const_params.size() != 2) ||
                  (const_params.at(0).shape.size() != 1) ||
                  (const_params.at(1).shape.size() != 1),
              OpsFusion::dd_format("Unsupported const spec for Layernorm\n") +
                  OpsFusion::dd_format(
                      "(Details : #const params == 2 ({}), Const param1 dim "
                      "== 1 ({}), Const param2 dim == 1 ({})",
                      const_params.size(), const_params.at(0).shape.size(),
                      const_params.at(1).shape.size()));
  // gamma
  io.write(0, const_params.at(0).data, b_bo_size_);
  // beta
  io.write(b_bo_size_, const_params.at(1).data, b_bo_size_);

  RYZENAI_LOG_TRACE("Layernorm initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void layernorm<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  // Check the number of inputs
  if (const_params.size() != 2) {
    throw std::runtime_error("LRN IPU Wrapper expect to have 2 constants.");
  }

  // Create input/output BOs
  const size_t A_BO_SIZE = size_t(b_) * size_t(m_) * size_t(k_) * a_dtype_size_;
  const size_t C_BO_SIZE = size_t(b_) * size_t(m_) * size_t(k_) * c_dtype_size_;

  RYZENAI_LOG_TRACE("LRN: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(b_bo_size_ * 2) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), b_bo_size_ * 2,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));

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
void layernorm<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                        std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 1) {
    throw std::runtime_error("LRN IPU Wrapper expect to have one input.");
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

  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  a_bo_.write(input.at(0).data);
  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  auto instr_bo_key = get_instr_key(txn_fname_prefix_, b_, m_, k_);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));

  // launch the kernel
  auto kernel_ = xrt_ctx_->get_kernel();

  auto run_aie_start = GET_ELAPSED_TIME_NS();

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, a_bo_, b_bo_, c_bo_,
                                            0, 0, true, false);
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
  c_bo_.read(output.at(0).data);
  auto c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = static_cast<int64_t>(c_copy_stop - c_copy_start);
  auto exec_end = GET_ELAPSED_TIME_NS();
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> layernorm<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  std::string txn_key = get_instr_key(txn_fname_prefix_, b_, m_, k_);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
void layernorm<InT, WtT, OutT>::set_params() {
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> layernorm<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  size_t const_params_bo_size = b_bo_size_ * 2;
  size_t input_bo_size = (b_ * m_ * k_ * sizeof(InT));
  size_t output_bo_size = (b_ * m_ * k_ * sizeof(OutT));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 3, 0, output_bo_size}};

  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
void layernorm<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
std::once_flag layernorm<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t layernorm<InT, WtT, OutT>::layernorm_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag layernorm<InT, WtT, OutT>::instr_reg_flag_;

template class layernorm<uint16_t, uint16_t, uint16_t>;
} // namespace sd
} // namespace ryzenai
