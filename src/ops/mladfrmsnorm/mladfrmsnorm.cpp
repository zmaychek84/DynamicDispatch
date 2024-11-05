/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */
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
#include <xrt_context/xrt_context.hpp>

#include <ops/mladfrmsnorm/mladfrmsnorm.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

namespace ryzenai {

namespace {
std::string getXCLBinName(std::string op_version) {
  return (op_version == "v1")
             ? OpInterface::get_dd_base_dir() +
                   LLAMA2_MLADF_2x4x4_V1_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_PATH
             : OpInterface::get_dd_base_dir() +
                   LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_PATH;
}

std::tuple<size_t, size_t> extract_MK(const Tensor &input) {
  size_t M = 0;
  size_t K = 0;
  if (input.shape.size() != 2) {
    throw std::runtime_error("rmsnorm expects a rank 2 tensor [Rows,Cols]");
  }

  M = input.shape.at(0);
  K = input.shape.at(1);
  return std::make_tuple(M, K);
}
} // namespace

template <typename LhsT, typename WtsT, typename OutT>
bool rms_norm<LhsT, WtsT, OutT>::isSupportedShape(const Tensor &operand) const {
  const auto &supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  const auto shape_operand = extract_MK(operand);
  for (const auto &supported : supported_shapes) {
    if (supported == shape_operand) {
      return true;
    }
  }
  return false;
}

template <typename LhsT, typename WtsT, typename OutT>
std::once_flag rms_norm<LhsT, WtsT, OutT>::logger_flag_;

template <typename LhsT, typename WtsT, typename OutT>
uint64_t rms_norm<LhsT, WtsT, OutT>::rms_norm_count = 0;

template <typename LhsT, typename WtsT, typename OutT>
std::once_flag rms_norm<LhsT, WtsT, OutT>::instr_reg_flag_;

template <typename LhsT, typename WtsT, typename OutT>
std::once_flag rms_norm<LhsT, WtsT, OutT>::instr_reg_v1_flag_;

template <typename LhsT, typename WtsT, typename OutT>
void rms_norm<LhsT, WtsT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename LhsT, typename WtsT, typename OutT>
std::string rms_norm<LhsT, WtsT, OutT>::get_instr_key(std::string prefix,
                                                      size_t m,
                                                      size_t k) const {
  return "mladfrmsnorm_" + prefix + "_" + std::to_string(m) + "_" +
         std::to_string(k);
}

template <typename LhsT, typename WtsT, typename OutT>
void rms_norm<LhsT, WtsT, OutT>::setup_instr_init() {}
template <typename LhsT, typename WtsT, typename OutT>
void rms_norm<LhsT, WtsT, OutT>::setup_instr_registry(
    const std::map<std::string, std::any> &attr) {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  std::vector<std::tuple<int, int>> supported_shapes;
  std::map<int, int, std::greater<int>> m_shape_list;

  if (attr.find("shapes") != attr.end()) {
    auto shapes = std::any_cast<std::vector<std::vector<int>>>(
        attr.find("shapes")->second);
    for (auto sh : shapes) {
      supported_shapes.push_back(std::tuple<int, int>{sh[0], sh[1]});
    }
  } else {
    supported_shapes = default_shapes_.find(txn_fname_prefix_)->second;
  }

  for (auto &sh : supported_shapes) {
    m_shape_list.emplace(std::get<0>(sh), 1);

    auto key =
        get_instr_key(txn_fname_prefix_, std::get<0>(sh), std::get<1>(sh));
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
    auto iter = m_shape_list.end();
    iter--;
    thresholds_.push_back(std::make_pair(0, iter->first));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
  xrt_ctx_->get_registry().add_layer_params(layer_params);
}

template <typename LhsT, typename WtsT, typename OutT>
void rms_norm<LhsT, WtsT, OutT>::set_kernel_shape(
    const std::vector<size_t> &shape) {
  if (thresholds_.size()) {
    for (const auto &threshold : thresholds_) {
      if (shape.at(0) > threshold.first) {
        kernel_x_shape_[0] = threshold.second;
        break;
      }
    }
  }
  // kernel_x_shape_[1] = 4096;
  kernel_x_shape_[1] = shape.at(1);

  if (load_xrt_) {
    instr_bo_key_ = get_instr_key(txn_fname_prefix_, kernel_x_shape_[0],
                                  kernel_x_shape_[1]);
  }
}

template <typename LhsT, typename WtsT, typename OutT>
rms_norm<LhsT, WtsT, OutT>::rms_norm(
    const std::string &operand_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr)
    : load_xrt_(load_xrt) {
  if (operand_dtype != "bfloat16") {
    throw std::runtime_error(
        "rmsnorm only supportes homogeneous bfloat16 data type "
        "for activation, weights matrices and result");
  }
  operand_dtype_ = operand_dtype;
  operand_dtype_size_ = sizeof(LhsT);
  txnbin_operand_header = {{"bfloat16", "a16"}};

  op_version_ = "v1";
  if (attr.find("op_version") != attr.end()) {
    op_version_ = std::any_cast<std::string>(attr.find("op_version")->second);
    if (op_version_ != "v1") {
      throw std::runtime_error("The selected op version does not exist");
    }
  }

  txn_fname_prefix_ =
      "rmsnorm_" + op_version_ + "_" + txnbin_operand_header.at(operand_dtype_);

  default_shapes_[txn_fname_prefix_] = std::vector<std::tuple<int, int>>();
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2048, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(1920, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(1792, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(1664, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(1536, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(1408, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(1280, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(1152, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(1024, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(800, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(768, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(640, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(512, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(384, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(256, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(128, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 4096));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(1, 4096));
  // default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(256, 3072));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2048, 3072));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(1024, 3072));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(512, 3072));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(256, 3072));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(128, 3072));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(1, 3072));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(3072, 4096));

  rms_norm_id_ = rms_norm_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = getXCLBinName(op_version_);

  // preempting bo creation with largest shape
  std::vector<std::vector<int>> shape_vector;
  for (const auto &entry : default_shapes_[txn_fname_prefix_]) {
    shape_vector.push_back(utils::tuple_to_vector(entry));
  }
  const auto operand_size_in_bytes =
      utils::max_element_count_with_skips(shape_vector) * operand_dtype_size_;

  const auto wts_size_in_bytes =
      utils::max_element_count_with_skips(shape_vector, {0}) *
      operand_dtype_size_;

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);

    if (op_version_ == "v1") {
      std::call_once(instr_reg_v1_flag_,
                     [this, &attr]() { setup_instr_init(); });
    } else {
      std::call_once(instr_reg_flag_, [this, &attr]() { setup_instr_init(); });
    }
    setup_instr_registry(attr);
    if (attr.find("skip_create_input") == attr.end()) {
      a_bo_ =
          xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
      b_bo_ =
          xrt::bo(xrt_ctx_->get_device(), wts_size_in_bytes,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    }
    if (attr.find("skip_create_output") == attr.end()) {
      c_bo_ =
          xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes,
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
    std::string header =
        "rms_norm_id M K Execute_time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "Wts_copy_time(ns) Wts_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[rmsnorm] ID: " + std::to_string(rms_norm_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (operand_dtype, b_dtype, c_dtype): (" + operand_dtype_ +
                    ", " + operand_dtype_ + ", " + operand_dtype_ + ")");
}

template <typename LhsT, typename WtsT, typename OutT>
void rms_norm<LhsT, WtsT, OutT>::execute(std::vector<Tensor> &input,
                                         std::vector<Tensor> &output) {

  // The first data is a and second data is b
  LhsT *a = (LhsT *)input.at(0).data;
  WtsT *b = (WtsT *)input.at(1).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;

  auto exec_start = GET_ELAPSED_TIME_NS();

  if (!isSupportedShape(input.at(0))) {
    throw std::runtime_error("Unsupported shape for rmsnorm");
  }
  // a_bo copy
  operand_size_in_bytes_ =
      utils::running_product_with_skips(input.at(0).shape) *
      operand_dtype_size_;
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  LhsT *a_bo_map = a_bo_.map<LhsT *>();
  memcpy((void *)a_bo_map, (void *)a, operand_size_in_bytes_);
  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  // b_bo copy
  wts_size_in_bytes_ = utils::running_product_with_skips(input.at(1).shape) *
                       operand_dtype_size_;
  auto b_copy_start = GET_ELAPSED_TIME_NS();
  WtsT *b_bo_map = b_bo_.map<WtsT *>();
  memcpy((void *)b_bo_map, (void *)b, wts_size_in_bytes_);
  auto b_copy_stop = GET_ELAPSED_TIME_NS();

  // b_bo sync
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();

  b_copy_time_ = static_cast<int64_t>(b_copy_stop - b_copy_start);
  b_sync_time_ = static_cast<int64_t>(b_sync_stop - b_sync_start);

  // prepare inst_bo and param_bo
  set_kernel_shape(input.at(0).shape);

  // launch the kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  // do we really need to sync before? c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
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
      std::to_string(rms_norm_id_) + " " + std::to_string(kernel_x_shape_[0]) +
      " " + std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(b_copy_time_) + " " + std::to_string(b_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename LhsT, typename WtsT, typename OutT>
std::vector<xrt::bo> rms_norm<LhsT, WtsT, OutT>::get_inputs() {
  return {a_bo_, b_bo_};
}

template <typename LhsT, typename WtsT, typename OutT>
std::vector<xrt::bo> rms_norm<LhsT, WtsT, OutT>::get_outputs() {
  return {c_bo_};
}

template <typename LhsT, typename WtsT, typename OutT>
void rms_norm<LhsT, WtsT, OutT>::execute(std::vector<uint64_t> &input,
                                         std::vector<uint64_t> &output,
                                         bool wait) {
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key_);
  auto instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();
  auto run = kernel_(
      2, instr_bo, instr_bo_words, input[0] + DDR_AIE_ADDR_OFFSET,
      input[1] + DDR_AIE_ADDR_OFFSET, output[0] + DDR_AIE_ADDR_OFFSET, 0, 0);

  if (wait) {
    run.wait2();
  }
  return;
}

template <typename LhsT, typename WtsT, typename OutT>
void rms_norm<LhsT, WtsT, OutT>::execute(std::vector<xrt::bo> &input,
                                         std::vector<xrt::bo> &output,
                                         bool wait) {
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key_);
  auto instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();
  auto run = kernel_(2, instr_bo, instr_bo_words,
                     input[0].address() + DDR_AIE_ADDR_OFFSET,
                     input[1].address() + DDR_AIE_ADDR_OFFSET,
                     output[0].address() + DDR_AIE_ADDR_OFFSET, 0, 0);
  if (wait) {
    run.wait2();
  }
  return;
}

template <typename LhsT, typename WtsT, typename OutT>
const std::vector<uint8_t> rms_norm<LhsT, WtsT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  const auto [M, K] = extract_MK(input.at(0));
  std::string txn_key = get_instr_key(txn_fname_prefix_, M, K);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtsT, typename OutT>
const std::vector<uint8_t> rms_norm<InT, WtsT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return {};
}

template <typename LhsT, typename WtsT, typename OutT>
std::vector<OpArgMap> rms_norm<LhsT, WtsT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  const auto shape_operand = extract_MK(input.at(0));
  const auto shape_wts = input.at(1).shape.at(0);
  const auto shape_result = extract_MK(output.at(0));

  if ((shape_operand != shape_result)) {
    throw std::runtime_error("mismatch shape of activation and result not "
                             "supported for rmsnorm\n");
  }
  if (std::get<1>(shape_result) != shape_wts) {
    throw std::runtime_error(
        "Mismatched shape between rmsnorm weights and activation/result "
        "not supported for rmsnorm");
  }
  const auto num_elem_operand =
      utils::running_product_with_skips(utils::tuple_to_vector(shape_operand));
  const auto num_elem_wts = shape_wts;
  size_t input_1_bo_size = (num_elem_operand * sizeof(LhsT));
  size_t input_2_bo_size = (num_elem_wts * sizeof(WtsT));
  size_t output_bo_size = (num_elem_operand * sizeof(OutT));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::INPUT, 1, 1, 0, input_2_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, output_bo_size},
  };
  return arg_map;
}

template class rms_norm<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
