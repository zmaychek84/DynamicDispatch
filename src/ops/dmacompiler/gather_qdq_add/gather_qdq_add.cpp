/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */
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

#include "utils/ctrl_pkt_utils.hpp"

#include <ops/dmacompiler/gather_qdq_add/gather_qdq_add.hpp>
#include <ops/op_interface.hpp>
#include <ops/ops_common/ctrlpkt.hpp>
#include <txn_container.hpp>
#include <utils/instruction_registry.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>
#include <xrt_context/xrt_context.hpp>
// AIE Driver header
#include "xaiengine.h"

#include "ops/ops_common/kernel_structs.hpp"
#include "ops/ops_common/matmul_matrix.hpp"

using namespace matmul_matrix;
using namespace kernel_structs;

namespace ryzenai {

static std::tuple<size_t, size_t, size_t>
extract_MKN(const std::vector<Tensor> &inputs) {
  size_t M = 0;
  size_t K = 0;
  size_t N = 0;

  if (inputs.at(0).shape.size() == 3) {
    M = inputs.at(0).shape.at(0);
    K = inputs.at(0).shape.at(1) * inputs.at(0).shape.at(2);
    N = inputs.at(3).shape.at(1) * inputs.at(3).shape.at(2);
  }

  return std::make_tuple(M, K, N);
}

template <typename InT, typename WtT, typename OutT>
std::once_flag gather_qdq_add<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t gather_qdq_add<InT, WtT, OutT>::gather_qdq_add_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag gather_qdq_add<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
void gather_qdq_add<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
std::tuple<size_t, size_t, size_t>
gather_qdq_add<InT, WtT, OutT>::map_padded_shape(size_t M, size_t K,
                                                 size_t N) const {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<std::tuple<int, int, int>> &supported_shapes = iter->second;
  size_t Mo = M;
  size_t Ko = K;
  size_t No = N;
  size_t fidx = 0;
  bool f_found = false;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes[i];
    if (M == std::get<0>(mat) && K == std::get<1>(mat) &&
        N == std::get<2>(mat)) {
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
    Ko = std::get<1>(mat);
    No = std::get<2>(mat);
  } else {
    throw std::runtime_error("Cannot find the shape");
  }
  return std::make_tuple(Mo, Ko, No);
}

template <typename InT, typename WtT, typename OutT>
std::string gather_qdq_add<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                                          size_t m, size_t k,
                                                          size_t n) const {
  return "gatherqdqadd_" + prefix + "_" + std::to_string(m) + "_" +
         std::to_string(k) + "_" + std::to_string(n);
}

template <typename InT, typename WtT, typename OutT>
void gather_qdq_add<InT, WtT, OutT>::setup_instr_registry() {

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
gather_qdq_add<InT, WtT, OutT>::gather_qdq_add(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"bfloat16", "abf16"}, {"uint16", "a16"}, {"uint8", "a8"}};
  txnbin_b_header = {{"bfloat16", "abf16"}, {"uint16", "a16"}, {"uint8", "a8"}};
  txnbin_c_header = {{"bfloat16", "accbf16"}, {"uint16", "acc16"}};

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_["gatherqdqadd_4x4_a16a16acc16"] =
      std::vector<std::tuple<int, int, int>>();
  default_shapes_["gatherqdqadd_4x4_a16a16acc16"].push_back(
      std::make_tuple(12, 32768, 4096));

  // raw shape is the actual shape from ONNX
  raw_shapes_["gatherqdqadd_4x4_a16a16acc16"] =
      std::vector<std::tuple<int, int, int>>();
  raw_shapes_["gatherqdqadd_4x4_a16a16acc16"].push_back(
      std::make_tuple(12, 32768, 4096));

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);
  gather_qdq_add_id_ = gather_qdq_add_count++;

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
    RYZENAI_LOG_TRACE("Gather qdq add: DesignFormat: " + design_param_);
  }

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    txn_fname_prefix_ = "gatherqdqadd_4x4_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_c_header.at(c_dtype_);

    param_fname_prefix_ = "gatherqdqadd_4x4_" + txnbin_a_header.at(a_dtype_) +
                          txnbin_b_header.at(b_dtype_) +
                          txnbin_c_header.at(c_dtype_);
  }

  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));

    if (input_shape_vector.size() == 3) {

      a_shape_[0] = input_shape_vector[0];
      a_shape_[1] = input_shape_vector[1];
      a_shape_[2] = input_shape_vector[2];

      b_shape_[0] = input_shape_vector[0];
      b_shape_[1] = input_shape_vector[1];
      b_shape_[2] = input_shape_vector[2];

    } else {
      RYZENAI_LOG_INFO(
          "Input Shape attribute does not have the expected number of "
          "elements.Number of passed : " +
          std::to_string(input_shape_vector.size()) + ", Expected:3");
    }
    RYZENAI_LOG_TRACE(
        "GatherQdqAdd: InputShape: " + std::to_string(input_shape_vector[0]) +
        ", " + std::to_string(input_shape_vector[1]) + ", " +
        std::to_string(input_shape_vector[2]));
  } else {
    RYZENAI_LOG_INFO("Input Shape attribute not found or not of correct type.");
  }

  if (attr.count("output_shape") &&
      attr.at("output_shape").type() == typeid(std::vector<int>)) {
    const auto &output_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("output_shape"));

    if (output_shape_vector.size() == 3) {
      c_shape_[0] = output_shape_vector[0];
      c_shape_[1] = output_shape_vector[1];
      c_shape_[2] = output_shape_vector[2];
    } else {
      RYZENAI_LOG_INFO(
          "Output Shape attribute does not have the expected number of "
          "elements.Number of passed : " +
          std::to_string(output_shape_vector.size()) + ", Expected:3");
    }
    RYZENAI_LOG_TRACE("Gather Qdq Add: OutputShape: " +
                      std::to_string(output_shape_vector[0]) + ", " +
                      std::to_string(output_shape_vector[1]) + ", " +
                      std::to_string(output_shape_vector[2]));
  } else {
    RYZENAI_LOG_INFO(
        "Output Shape attribute not found or not of correct type.");
  }

  KERNEL_M_MAX = 512;

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
    std::string header =
        "gather_qdq_add_id M K N kernel_m kernel_k kernel_n Execute"
        "time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[GatherADD] ID: " + std::to_string(gather_qdq_add_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype_ + ", " +
                    b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void gather_qdq_add<InT, WtT, OutT>::set_params(
    const std::string &model_name, std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;
  if (model_name == "4x4PSW1.0") {
    is_ctrl_pkt_ = 1;
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::PSW1_0_A16W8_QDQ_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void gather_qdq_add<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("gatherqdqadd initialize_const_params(ptr) ...");

  DD_THROW_IF(
      (const_params.size() != 1),
      OpsFusion::dd_format("Unsupported const spec for gatherqdqadd\n") +
          OpsFusion::dd_format("(Details : #const params == 1 ({})",
                               const_params.size()));

  auto qdq_params = (int32_t *)const_params.at(0).data;
  auto qdq_params_size = matmul_matrix::QDQparam_size * sizeof(int32_t);
  io.write(0, (void *)qdq_params, qdq_params_size);

  RYZENAI_LOG_TRACE("Gatherqdqadd initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void gather_qdq_add<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {

  if (const_params.size() != 1) {
    throw std::runtime_error(
        "GATHERQDQADD IPU Wrapper expect to have one constant.");
  }

  kernel_x_shape_[0] = a_shape_[0];
  kernel_x_shape_[1] = a_shape_[1] * a_shape_[2];

  kernel_y_shape_[0] = 0;
  kernel_y_shape_[1] = 0;

  kernel_z_shape_[0] = c_shape_[0];
  kernel_z_shape_[1] = c_shape_[1] * c_shape_[2];

  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  /* Create input/output BOs */
  const size_t B_BO_SIZE = matmul_matrix::QDQparam_size * sizeof(int32_t);
  const size_t A_BO_SIZE =
      2 * (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_);
  const size_t C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_);
  RYZENAI_LOG_TRACE("GATHERQDQADD: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
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

  if (is_ctrl_pkt_) {
    // Based on the mapped_shape to get the meta json file
    auto [M, K, N] = map_padded_shape(kernel_x_shape_[0], kernel_x_shape_[1],
                                      kernel_z_shape_[1]);

    std::vector<uint8_t> json_data;
    try {
      auto json_key =
          get_instr_key(param_fname_prefix_, M, K, N) + "_ctrl_meta";
      Transaction &txn = Transaction::getInstance();
      json_data = txn.get_txn_bvec(json_key);
    } catch (...) {
      is_ctrl_pkt_ = 0;
    }

    if (is_ctrl_pkt_) {
      std::cout << "ctrlpkt patching" << std::endl;
      RYZENAI_LOG_TRACE("Gatherqdqadd patch ctrlpkt ... START");
      // get param_bo address
      auto param_bo_key =
          get_instr_key(param_fname_prefix_, M, K, N) + "_param";
      const xrt::bo &param_bo =
          xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;

      // Get ctrl pkt patch info from json
      std::vector<CtrlPktPatchInfo> ctrlpkt_info;
      ctrlpkt_info = json_str_to_ctrlpkt_patch_info(json_data);

      // Get the ctrl pkt
      auto ctrl_bo_key = get_instr_key(param_fname_prefix_, M, K, N) + "_ctrl";
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
      RYZENAI_LOG_TRACE("gatherqdqadd patch ctrlpkt ... DONE");
    }
  }
}

template <typename InT, typename WtT, typename OutT>
void gather_qdq_add<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                             std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 2) {
    throw std::runtime_error(
        "GATHERQDQADD IPU Wrapper expect to have two inputs.");
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

  // each input needs to have M * K size
  auto [M, K, N] =
      map_padded_shape(input.at(a_idx).shape.at(0), input.at(a_idx).shape.at(1),
                       output.at(a_idx).shape.at(1));

  size_t a_size = M * K * sizeof(InT);
  RYZENAI_LOG_TRACE("GATHERQDQADD: a_size:" + std::to_string(a_size));
  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  memcpy((void *)a_bo_map, (void *)a, a_size);
  memcpy((void *)(reinterpret_cast<int8_t *>(a_bo_map) + M * K * sizeof(InT)),
         (void *)b, a_size);
  // memcpy((void*)&a_bo_map[a_size], (void*)b, a_size);

  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  // prepare inst_bo and param_bo
  auto instr_bo_key = get_instr_key(txn_fname_prefix_, M, K, N);
  auto param_bo_key = get_instr_key(param_fname_prefix_, M, K, N) + "_param";
  // std::cout << instr_bo_key << std::endl;
  // std::cout << param_bo_key << std::endl;
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  const xrt::bo &param_bo =
      xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel

  auto run_aie_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  ryzenai::dynamic_dispatch::execute_kernel(
      kernel_, 2, instr_bo, instr_bo_words, c_bo_, a_bo_, b_bo_, param_bo,
      ctrl_bo_, true, is_ctrl_pkt_);
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
         c_shape_[0] * c_shape_[1] * c_shape_[2] * sizeof(OutT));
  auto c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = static_cast<int64_t>(c_copy_stop - c_copy_start);
  auto exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(gather_qdq_add_id_) + " " + std::to_string(a_shape_[0]) +
      " " + std::to_string(a_shape_[1]) + " " + std::to_string(a_shape_[2]) +
      " " + std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> gather_qdq_add<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko, No] = map_padded_shape(M, K, N);
  std::string txn_key = get_instr_key(txn_fname_prefix_, Mo, Ko, No);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
gather_qdq_add<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko, No] = map_padded_shape(M, K, N);
  // TODO: Add check to validate tensor shapes
  std::string param_key =
      get_instr_key(param_fname_prefix_, Mo, Ko, No) + "_param";
  // std::cout << "Super kernel params name : " << fname << std::endl;
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(param_key);
}

template <typename InT, typename WtT, typename OutT>
std::vector<uint8_t> gather_qdq_add<InT, WtT, OutT>::get_ctrl_pkts(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko, No] = map_padded_shape(M, K, N);
  // TODO: Add check to validate tensor shapes
  std::string ctrl_key =
      get_instr_key(param_fname_prefix_, Mo, Ko, No) + "_ctrl";
  try {
    Transaction &txn = Transaction::getInstance();
    return txn.get_txn_bvec(ctrl_key);
  } catch (...) {
    return {};
  }
}

template <typename InT, typename WtT, typename OutT>
std::vector<CtrlPktPatchInfo>
gather_qdq_add<InT, WtT, OutT>::get_ctrl_pkt_patch_info(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko, No] = map_padded_shape(M, K, N);
  // TODO: Add check to validate tensor shapes
  try {
    auto ctrl_pkt_meta =
        get_instr_key(param_fname_prefix_, Mo, Ko, No) + "_ctrl_meta";
    Transaction &txn = Transaction::getInstance();
    return json_str_to_ctrlpkt_patch_info(txn.get_txn_bvec(ctrl_pkt_meta));
  } catch (...) {
    // throw std::runtime_error("gatherqdqadd : Can not file the ctrl_meta.json
    // file");
    return {};
  }
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> gather_qdq_add<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, N] = extract_MKN(input);

  auto [Mo, Ko, No] = map_padded_shape(M, K, N);

  size_t const_params_bo_size = matmul_matrix::QDQparam_size * sizeof(int32_t);
  size_t input_1_bo_size = (Mo * Ko * sizeof(InT));
  size_t input_2_bo_size = (Mo * Ko * sizeof(WtT));
  size_t output_bo_size = (Mo * No * sizeof(OutT));
  size_t super_kernel_size = get_super_kernel_params(input, output).size();
  size_t ctrl_pkt_size = get_ctrl_pkts(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::INPUT, 1, 1, input_1_bo_size, input_2_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 2, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, 3, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size},
      {OpArgMap::OpArgType::CTRL_PKT_BIN, 4, 0, 0, ctrl_pkt_size}};
  return arg_map;
}

template class gather_qdq_add<uint8_t, uint8_t, uint16_t>;
template class gather_qdq_add<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
