/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>

#ifndef _WIN32
#include <cmath>
#endif

#include <iomanip>
#include <iterator>
#include <string>

#include <ops/op_interface.hpp>
#include <ops/ops_common/help_file.hpp>
#include <ops/sd/resize.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>

#include "txn_helper/txn_helper.hpp"
#include "utils/dpu_mdata.hpp"

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

using std::vector;

namespace ryzenai {

namespace sd {

inline int get_shape_ele_num(const std::vector<int> &shape) {
  int total_num = 1;
  for (int dim : shape) {
    total_num *= dim;
  }
  return total_num;
}

template <typename InT, typename OutT>
void resize<InT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  auto supported_shapes = default_shapes_.find(txn_fname_prefix_)->second;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto shape = supported_shapes.at(i);
    auto key = get_key(txn_fname_prefix_, shape.first, shape.second);
    instructions.push_back(std::make_pair(key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename OutT>
std::string resize<InT, OutT>::get_key(std::string prefix,
                                       const std::vector<int> &a_shape,
                                       const std::vector<int> &c_shape) const {
  auto key = prefix;
  for (int i = 0; i < a_shape.size(); i++) {
    key += "_" + std::to_string(a_shape[i]);
  }
  key += "__";
  for (int i = 0; i < c_shape.size(); i++) {
    key += "_" + std::to_string(c_shape[i]);
  }
  return key;
}

// resize class constructor
template <typename InT, typename OutT>
resize<InT, OutT>::resize(const std::string &a_dtype,
                          const std::string &c_dtype, bool load_xrt,
                          const std::map<std::string, std::any> &attr)
    : attr_(attr) {

  if (Utils::get_env_var("DEBUG_SD", "0") != "0") {
    this->debug_ = true;
  }
  txnbin_a_header = {{"bfloat16", "a16bf"}};
  txnbin_c_header = {{"bfloat16", "acc16bf"}};

  a_dtype_ = a_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  c_dtype_size_ = sizeof(OutT);
  resize_id_ = resize_count++;
  XCLBIN_FNAME_ =
      OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\SDResize.xclbin";
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME_));
  txn_fname_prefix_ =
      "sd_resize_" + txnbin_a_header.at(a_dtype) + txnbin_c_header.at(c_dtype);
  // default shape is pair of a_shape + c_shape, ifm shape always first
  default_shapes_["sd_resize_a16bfacc16bf"].emplace_back(std::make_pair(
      std::vector<int>{1, 64, 64, 512}, std::vector<int>{1, 128, 128, 512}));
  default_shapes_["sd_resize_a16bfacc16bf"].emplace_back(std::make_pair(
      std::vector<int>{1, 128, 128, 512}, std::vector<int>{1, 256, 256, 512}));
  default_shapes_["sd_resize_a16bfacc16bf"].emplace_back(std::make_pair(
      std::vector<int>{1, 256, 256, 256}, std::vector<int>{1, 512, 512, 256}));
  default_shapes_["sd_resize_a16bfacc16bf"].emplace_back(std::make_pair(
      std::vector<int>{2, 16, 16, 1280}, std::vector<int>{2, 32, 32, 1280}));
  default_shapes_["sd_resize_a16bfacc16bf"].emplace_back(std::make_pair(
      std::vector<int>{2, 32, 32, 640}, std::vector<int>{2, 64, 64, 640}));
  default_shapes_["sd_resize_a16bfacc16bf"].emplace_back(std::make_pair(
      std::vector<int>{2, 8, 8, 1280}, std::vector<int>{2, 16, 16, 1280}));
  // SD3
  default_shapes_["sd_resize_a16bfacc16bf"].emplace_back(std::make_pair(
      std::vector<int>{1, 256, 256, 512}, std::vector<int>{1, 512, 512, 512}));
  default_shapes_["sd_resize_a16bfacc16bf"].emplace_back(
      std::make_pair(std::vector<int>{1, 512, 512, 256},
                     std::vector<int>{1, 1024, 1024, 256}));
  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  if (attr.count("a_shape") &&
      attr.at("a_shape").type() == typeid(std::vector<int>)) {
    a_shape_ = std::any_cast<const std::vector<int> &>(attr.at("a_shape"));
  } else {
    RYZENAI_LOG_INFO(
        "Input A Shape attribute not found or not of correct type.");
  }

  if (attr.count("c_shape") &&
      attr.at("c_shape").type() == typeid(std::vector<int>)) {
    c_shape_ = std::any_cast<const std::vector<int> &>(attr.at("c_shape"));
  } else {
    RYZENAI_LOG_INFO(
        "Output Shape attribute not found or not of correct type.");
  }

  std::call_once(logger_flag_, []() {
    std::string header =
        "sd_resize_id | Execute time | total time | Avg_time_per_aie_run\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[SD_Resize] ID: " + std::to_string(resize_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME_ + ", (a_dtype, c_dtype): (" +
                    a_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename OutT>
void resize<InT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("resize initialize_const_params ...");
  RYZENAI_LOG_TRACE("resize initialize_const_params ... DONE");
}

template <typename InT, typename OutT>
void resize<InT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  std::vector<Tensor> input;
  std::vector<Tensor> output;
  size_t A_BO_SIZE, B_BO_SIZE, C_BO_SIZE;
  A_BO_SIZE = B_BO_SIZE = C_BO_SIZE = 0;
  auto args_map_list = this->get_buffer_reqs(input, output, attr);
  for (const auto &args_map : args_map_list) {
    if (args_map.arg_type == OpArgMap::OpArgType::INPUT) {
      A_BO_SIZE = args_map.size;
    }
    if (args_map.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
      B_BO_SIZE = args_map.size;
    }
    if (args_map.arg_type == OpArgMap::OpArgType::OUTPUT) {
      C_BO_SIZE = args_map.size;
    }
  }

  RYZENAI_LOG_TRACE("SD_Resize: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  dummy_b_bo_ =
      xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  RYZENAI_LOG_TRACE("resize initialize_const_params ... DONE");
}

template <typename InT, typename OutT>
void resize<InT, OutT>::execute(std::vector<Tensor> &input,
                                std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("resize execute ...");
  a_bo_.write(input.at(0).data);
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto instr_bo_key = get_key(txn_fname_prefix_, a_shape_, c_shape_);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  size_t instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();

  auto run_aie_start = GET_ELAPSED_TIME_NS();
  // TODO: figure out the Bo order

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, a_bo_, dummy_b_bo_,
                                            c_bo_, 0, 0, true, false);
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);
  /* sync output activation to host memory */
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  c_bo_.read(output.at(0).data);

  RYZENAI_LOG_INFO(std::to_string(resize_id_) + " " +
                   std::to_string(num_run_aie_) + " " +
                   std::to_string(run_aie_time_) + " " +
                   std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename OutT>
void resize<InT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename OutT>
const std::vector<uint8_t> resize<InT, OutT>::get_transaction_bin() const {
  std::string txn_key = get_key(txn_fname_prefix_, a_shape_, c_shape_);
  Transaction &txn = Transaction::getInstance();
  std::vector<uint8_t> txnData = txn.get_txn_bvec(txn_key);
  return txnData;
}

template <typename InT, typename OutT>
const std::vector<uint8_t> resize<InT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return this->get_transaction_bin();
}

template <typename InT, typename OutT>
void resize<InT, OutT>::set_params(const std::string &modelName,
                                   const std::vector<int> &a_shape,
                                   const std::vector<int> &c_shape) {
  a_shape_ = a_shape;
  c_shape_ = c_shape;
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename OutT>
std::vector<OpArgMap> resize<InT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  size_t a_bo_size = (get_shape_ele_num(a_shape_) * a_dtype_size_);
  size_t c_bo_size = (get_shape_ele_num(c_shape_) * c_dtype_size_);

  RYZENAI_LOG_TRACE("SD_Resize: A_BO_SIZE:" + std::to_string(a_bo_size) +
                    " C_BO_SIZE:" + std::to_string(c_bo_size));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, a_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0,
       b_bo_size_}, // Dummy allocation
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, c_bo_size}};

  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("SD_Resize Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename OutT>
const std::vector<uint8_t> resize<InT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return {};
}

template <typename InT, typename OutT>
std::once_flag resize<InT, OutT>::logger_flag_;

template <typename InT, typename OutT>
uint64_t resize<InT, OutT>::resize_count = 0;

template <typename InT, typename OutT>
std::once_flag resize<InT, OutT>::instr_reg_flag_;

template class resize<std::uint16_t, std::uint16_t>;

} // namespace sd

} // namespace ryzenai
