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

#include <ops/maskedsoftmax/maskedsoftmax.hpp>
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

} // namespace

static std::tuple<size_t, size_t, size_t> extract_BMK(const Tensor &input) {
  size_t B = 0;
  size_t M = 0;
  size_t K = 0;
  if (input.shape.size() != 3) {
    throw std::runtime_error(
        "Masked SoftMax expects a rank 3 tensor [Batch,Rows,Cols]");
  }
  B = input.shape.at(0);
  M = input.shape.at(1);
  K = input.shape.at(2);
  return std::make_tuple(B, M, K);
}

template <typename LhsT, typename MaskT, typename OutT>
bool masked_softmax<LhsT, MaskT, OutT>::isSupportedShape(
    const Tensor &operand) const {
  const auto &supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  const auto shapeOperand = extract_BMK(operand);
  for (const auto &supported : supported_shapes) {
    if (supported == shapeOperand) {
      return true;
    }
  }
  return false;
}

template <typename LhsT, typename MaskT, typename OutT>
std::once_flag masked_softmax<LhsT, MaskT, OutT>::logger_flag_;

template <typename LhsT, typename MaskT, typename OutT>
uint64_t masked_softmax<LhsT, MaskT, OutT>::masked_softmax_count = 0;

template <typename LhsT, typename MaskT, typename OutT>
std::once_flag masked_softmax<LhsT, MaskT, OutT>::instr_reg_flag_;

template <typename LhsT, typename MaskT, typename OutT>
std::once_flag masked_softmax<LhsT, MaskT, OutT>::instr_reg_v1_flag_;

template <typename LhsT, typename MaskT, typename OutT>
void masked_softmax<LhsT, MaskT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename LhsT, typename MaskT, typename OutT>
std::string masked_softmax<LhsT, MaskT, OutT>::get_instr_key(std::string prefix,
                                                             size_t batch,
                                                             size_t m,
                                                             size_t k) const {
  return "maskedsoftmax_" + prefix + "_" + std::to_string(batch) + "_" +
         std::to_string(m) + "_" + std::to_string(k);
}

template <typename LhsT, typename MaskT, typename OutT>
void masked_softmax<LhsT, MaskT, OutT>::setup_instr_registry(
    const std::map<std::string, std::any> &attr) {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  std::vector<std::tuple<int, int, int>> supported_shapes;

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
    auto key = get_instr_key(txn_fname_prefix_, std::get<0>(sh),
                             std::get<1>(sh), std::get<2>(sh));
    instructions.push_back(std::make_pair(key, false));
  }

  xrt_ctx_->get_registry().add_instructions(instructions);
  xrt_ctx_->get_registry().add_layer_params(layer_params);
}

template <typename LhsT, typename MaskT, typename OutT>
void masked_softmax<LhsT, MaskT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  return;
}

template <typename LhsT, typename MaskT, typename OutT>
masked_softmax<LhsT, MaskT, OutT>::masked_softmax(
    const std::string &operand_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {
  if (operand_dtype != "bfloat16") {
    throw std::runtime_error(
        "MaskedSoftMax only supportes homogeneous bfloat16 data type "
        "for activation, mask and result");
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

  txn_fname_prefix_ = "maskedsoftmax_" + op_version_ + "_" +
                      txnbin_operand_header.at(operand_dtype_);

  default_shapes_[txn_fname_prefix_] = std::vector<std::tuple<int, int, int>>();

  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 128));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 256));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 384));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 512));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 640));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 768));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 896));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 1024));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 1152));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 1280));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 1408));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 1536));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 1664));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 1792));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 1920));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 2048));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 256, 256));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 384, 384));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 512, 512));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 640, 640));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 768, 768));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 896, 896));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1024, 1024));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1152, 1152));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1280, 1280));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1408, 1408));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1536, 1536));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1664, 1664));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1792, 1792));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 1920, 1920));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 2048, 2048));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 128, 3072));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(32, 3072, 3072));

  masked_softmax_id_ = masked_softmax_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = getXCLBinName(op_version_);

  // TODO not really sure we need this member
  kernel_x_shape_[0] = 32;
  kernel_x_shape_[1] = 3072;
  kernel_x_shape_[2] = 3072;
  auto shapeOperand = std::make_tuple(32, 3072, 3072);
  auto maskshapeOperand = std::make_tuple(1, 3072, 3072);
  operand_size_in_bytes_ =
      utils::running_product_with_skips(utils::tuple_to_vector(shapeOperand)) *
      sizeof(LhsT);
  mask_size_in_bytes_ = utils::running_product_with_skips(
                            utils::tuple_to_vector(maskshapeOperand)) *
                        sizeof(MaskT);

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    if (op_version_ == "v1") {
      std::call_once(instr_reg_v1_flag_,
                     [this, &attr]() { setup_instr_registry(attr); });
    } else {
      std::call_once(instr_reg_flag_,
                     [this, &attr]() { setup_instr_registry(attr); });
    }
    if (attr.find("skip_create_input") == attr.end()) {
      a_bo_ =
          xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    }
    if (attr.find("skip_create_mask") == attr.end()) {
      b_bo_ =
          xrt::bo(xrt_ctx_->get_device(), mask_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    }
    if (attr.find("skip_create_output") == attr.end()) {
      c_bo_ =
          xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes_,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    }
    // prepare inst_bo and param_bo
    instr_bo_key_ = get_instr_key(txn_fname_prefix_, kernel_x_shape_[0],
                                  kernel_x_shape_[1], kernel_x_shape_[2]);
  }

  std::call_once(logger_flag_, []() {
    std::string header = "masked_softmax_id Batch M N Execute"
                         "time(ns) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "Mask_copy_time(ns) Mask_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE(
      "[MASKEDSOFTMAX] ID: " + std::to_string(masked_softmax_id_) +
      ", XCLBIN: " + XCLBIN_FNAME + ", (operand_dtype, b_dtype, c_dtype): (" +
      operand_dtype_ + ", " + operand_dtype_ + ", " + operand_dtype_ + ")");
}

template <typename LhsT, typename MaskT, typename OutT>
void masked_softmax<LhsT, MaskT, OutT>::set_params(
    const std::string &model_name, std::vector<size_t> input_shape) {
  if (kernel_x_shape_[1] == input_shape.at(1) &&
      kernel_x_shape_[2] == input_shape.at(2)) {
    return;
  }

  kernel_x_shape_[0] = input_shape.at(0);
  kernel_x_shape_[1] = input_shape.at(1);
  kernel_x_shape_[2] = input_shape.at(2);
  std::vector<size_t> mask_shape = {1, input_shape.at(1), input_shape.at(2)};
  auto shapeOperand =
      std::make_tuple(input_shape.at(0), input_shape.at(1), input_shape.at(2));
  operand_size_in_bytes_ =
      utils::running_product_with_skips(input_shape) * operand_dtype_size_;
  mask_size_in_bytes_ =
      utils::running_product_with_skips(mask_shape) * operand_dtype_size_;

  instr_bo_key_ = get_instr_key(txn_fname_prefix_, kernel_x_shape_[0],
                                kernel_x_shape_[1], kernel_x_shape_[2]);

  return;
}

template <typename LhsT, typename MaskT, typename OutT>
void masked_softmax<LhsT, MaskT, OutT>::execute(std::vector<Tensor> &input,
                                                std::vector<Tensor> &output) {

  // The first data is a and second data is b
  LhsT *a = (LhsT *)input.at(0).data;
  MaskT *b = (MaskT *)input.at(1).data;

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
    throw std::runtime_error("Unsupported shape for masked softmax");
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
  mask_size_in_bytes_ = utils::running_product_with_skips(input.at(1).shape) *
                        operand_dtype_size_;
  auto b_copy_start = GET_ELAPSED_TIME_NS();
  MaskT *b_bo_map = b_bo_.map<MaskT *>();
  memcpy((void *)b_bo_map, (void *)b, mask_size_in_bytes_);
  auto b_copy_stop = GET_ELAPSED_TIME_NS();

  // b_bo sync
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();

  b_copy_time_ = static_cast<int64_t>(b_copy_stop - b_copy_start);
  b_sync_time_ = static_cast<int64_t>(b_sync_stop - b_sync_start);
  std::vector<xrt::bo> inputs = {a_bo_, b_bo_};
  std::vector<xrt::bo> outputs = {c_bo_};
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  // the following execute needs these member variables to be set
  kernel_x_shape_[0] = input.at(0).shape.at(0);
  kernel_x_shape_[1] = input.at(0).shape.at(1);
  kernel_x_shape_[2] = input.at(0).shape.at(2);
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
      std::to_string(masked_softmax_id_) + " " +
      std::to_string(kernel_x_shape_[0]) + " " +
      std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_x_shape_[2]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(b_copy_time_) + " " + std::to_string(b_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename LhsT, typename MaskT, typename OutT>
void masked_softmax<LhsT, MaskT, OutT>::execute(std::vector<xrt::bo> &input,
                                                std::vector<xrt::bo> &output,
                                                bool wait) {

  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key_);
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  // do we really need to sync before? c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto run = kernel_(2, instr_bo, instr_bo_words,
                     input[0].address() + DDR_AIE_ADDR_OFFSET,
                     input[1].address() + DDR_AIE_ADDR_OFFSET,
                     output[0].address() + DDR_AIE_ADDR_OFFSET, 0, 0);
  if (wait) {
    run.wait2();
  }
}
template <typename LhsT, typename MaskT, typename OutT>
std::vector<xrt::bo> masked_softmax<LhsT, MaskT, OutT>::get_inputs() {

  return {a_bo_, b_bo_};
}

template <typename LhsT, typename MaskT, typename OutT>
const std::vector<uint8_t>
masked_softmax<LhsT, MaskT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  const auto [B, M, K] = extract_BMK(input.at(0));
  std::string txn_key = get_instr_key(txn_fname_prefix_, B, M, K);

  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename MaskT, typename OutT>
const std::vector<uint8_t>
masked_softmax<InT, MaskT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return {};
}

template <typename LhsT, typename MaskT, typename OutT>
std::vector<OpArgMap> masked_softmax<LhsT, MaskT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  const auto shapeOperand = extract_BMK(input.at(0));
  const auto shapeMask = extract_BMK(input.at(1));
  const auto shapeResult = extract_BMK(output.at(0));

  if ((shapeOperand != shapeResult)) {
    throw std::runtime_error("mismatch shape of activation and result not "
                             "supported for masked softmax\n");
  }
  if (std::get<1>(shapeResult) != std::get<1>(shapeMask) ||
      std::get<2>(shapeResult) != std::get<2>(shapeMask)) {
    throw std::runtime_error("Mismatched shape of mask and activation/result "
                             "not supported for masked softmax");
  }
  const auto numElementsOperand =
      utils::running_product_with_skips(input.at(0).shape);
  const auto numElementsMask =
      utils::running_product_with_skips(input.at(1).shape);
  size_t input_1_bo_size = (numElementsOperand * sizeof(LhsT));
  size_t input_2_bo_size = (numElementsMask * sizeof(MaskT));
  size_t output_bo_size = (numElementsOperand * sizeof(OutT));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::INPUT, 1, 1, 0, input_2_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, output_bo_size},
  };
  return arg_map;
}

template class masked_softmax<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
