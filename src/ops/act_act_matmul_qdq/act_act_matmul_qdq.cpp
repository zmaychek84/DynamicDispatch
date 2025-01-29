/*
 Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 Licensed under the MIT License.
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

#include <ops/act_act_matmul_qdq/act_act_matmul_qdq.hpp>
#include <ops/ops_common/ctrlpkt.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include "ops/ops_common/mhagprb_matrix.hpp"
#include "utils/ctrl_pkt_utils.hpp"
#include "xaiengine.h"

// using namespace matmul_matrix;

namespace ryzenai {

static std::array<size_t, 2> extract_shape(const Tensor &tensor) {
  std::array<size_t, 2> res;
  if (tensor.shape.size() == 4) {
    if (tensor.shape.at(1) == tensor.shape.at(2)) { // NHWC
      res = {tensor.shape.at(1) * tensor.shape.at(2), tensor.shape.at(3)};
    } else { // NCHW
      res = {tensor.shape.at(1) * tensor.shape.at(3), tensor.shape.at(2)};
    }
  } else if (tensor.shape.size() == 3) {
    res = {tensor.shape.at(0) * tensor.shape.at(1), tensor.shape.at(2)};
  } else if (tensor.shape.size() == 2) {
    res = {tensor.shape.at(0), tensor.shape.at(1)};
  } else if (tensor.shape.size() == 1) {
    res = {tensor.shape.at(0)};
  } else {
    throw std::runtime_error(
        "Activation Matmul : Invalid shape received for Matrix");
  }
  return res;
}

template <typename InT, typename WtT, typename OutT>
std::once_flag act_act_matmul<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t act_act_matmul<InT, WtT, OutT>::act_act_matmul_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag act_act_matmul<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
void act_act_matmul<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
std::string
act_act_matmul<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                              std::vector<size_t> &mat) const {
  std::string out_str = "act_act_matmul_qdq_" + prefix;
  for (size_t i = 0; i < mat.size(); i++) {
    out_str += "_" + std::to_string(mat[i]);
  }
  return out_str;
}

template <typename InT, typename WtT, typename OutT>
void act_act_matmul<InT, WtT, OutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;

  std::vector<std::vector<size_t>> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key = get_instr_key(txn_fname_prefix_, mat);
    auto param_key = get_instr_key(param_fname_prefix_, mat) + "_param";
    instructions.push_back(std::make_pair(key, false));
    layer_params.push_back(std::make_pair(param_key, false));
  }

  xrt_ctx_->get_registry().add_instructions(instructions);
  xrt_ctx_->get_registry().add_layer_params(layer_params);
}

template <typename InT, typename WtT, typename OutT>
act_act_matmul<InT, WtT, OutT>::act_act_matmul(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"uint16", "a16"}, {"uint8", "a8"}};

  txnbin_b_header = {{"uint16", "w16"}, {"uint8", "w8"}};

  txnbin_c_header = {{"uint16", "acc16"}, {"uint8", "acc8"}};

  default_shapes_["act_act_matmul_qdq_4x4_a16w16acc16"] =
      std::vector<std::vector<size_t>>();
  default_shapes_["act_act_matmul_qdq_4x4_a16w16acc16"].push_back(
      std::vector<size_t>{256, 256, 64});
  default_shapes_["act_act_matmul_qdq_4x4_a16w16acc16"].push_back(
      std::vector<size_t>{1024, 1024, 64});
  default_shapes_["act_act_matmul_qdq_4x4_a16w16acc16"].push_back(
      std::vector<size_t>{4096, 4096, 64});

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  act_act_matmul_id_ = act_act_matmul_count++;

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
    RYZENAI_LOG_TRACE("act_act_matmul: DesignFormat: " + design_param_);
  }

  txn_fname_prefix_ = "act_act_matmul_qdq_4x2_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_c_header.at(c_dtype_);

  param_fname_prefix_ =
      "act_act_matmul_qdq_4x2_" + txnbin_a_header.at(a_dtype_) +
      txnbin_b_header.at(b_dtype_) + txnbin_c_header.at(c_dtype_);

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    txn_fname_prefix_ =
        "act_act_matmul_qdq_4x4_" + txnbin_a_header.at(a_dtype_) +
        txnbin_b_header.at(b_dtype_) + txnbin_c_header.at(c_dtype_);

    param_fname_prefix_ =
        "act_act_matmul_qdq_4x4_" + txnbin_a_header.at(a_dtype_) +
        txnbin_b_header.at(b_dtype_) + txnbin_c_header.at(c_dtype_);
  }

  KERNEL_M_MAX = 512;

  if (load_xrt == true) {
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
        "ipu_wrapper_id M K N kernel_m kernel_k kernel_n Execute"
        "time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE(
      "[Activation Matmul] ID: " + std::to_string(act_act_matmul_id_) +
      ", XCLBIN: " + XCLBIN_FNAME + ", (a_dtype, b_dtype, c_dtype): (" +
      a_dtype + ", " + b_dtype + ", " + c_dtype + ")");
}

template <typename InT, typename WtT, typename OutT>
void act_act_matmul<InT, WtT, OutT>::set_params(
    const std::string &model_name, std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;

  if (model_name == "mzdk5") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mzdk5_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "4x4mzdk5") {
    is_ctrl_pkt_ = 1;
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                   ryzenai::mzdk54x4_A16W8_QDQ_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }

  kernel_x_shape_[0] = input_shape.at(0); // M
  kernel_x_shape_[1] = input_shape.at(1); // K

  kernel_y_shape_[0] = input_shape.at(1); // K
  kernel_y_shape_[1] = input_shape.at(2); // N

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void act_act_matmul<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("atmatmul initialize_const_params(ptr) ...");
  DD_THROW_IF(
      (const_params.size() != 1) || (const_params.at(0).shape.size() != 2),
      OpsFusion::dd_format(
          "Unsupported const spec for actmatmul\n"
          "(Details : #const params == 1 ({}), Const param1 dim == 2 ({})",
          const_params.size(), const_params.at(0).shape.size()));
  const int qdq_idx = 0;

  auto qdq_param = (int32_t *)const_params.at(qdq_idx).data;

  int size_qdqparam;
  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    size_qdqparam = QDQparam_size * sizeof(int32_t);
    qdq_param[(16 * 0) + qdq_Mv_idx] = mha_mzdk5_sq;
    qdq_param[(16 * 0) + qdq_Nv_idx] = mha_mzdk5_val_subv_cols;
  } else {
    size_qdqparam = QDQparam_size * num_qdq_nodes * sizeof(int32_t);
    qdq_param[(16 * 2) + qdq_Mv_idx] = mha_mzdk5_sq;
    // qdq_param[(16 * 2) + qdq_Nv_idx] = mha_mzdk5_st_pad;  // should be 64 for
    // QKt Matmul and 128 for SMV Matmul

    qdq_param[(16 * 3) + qdq_Mv_idx] = mha_mzdk5_sq;
    qdq_param[(16 * 3) + qdq_Nv_idx] = mha_mzdk5_val_subv_cols;
  }
  io.write(0, qdq_param, size_qdqparam);

  RYZENAI_LOG_TRACE("actmatmul initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void act_act_matmul<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {

  // Check the number of inputs
  DD_ASSERT((const_params.size() == 1),
            OpsFusion::dd_format("actmatmul expects one constant. Got {}",
                                 const_params.size()));

  int size_qdqparam;
  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    size_qdqparam = QDQparam_size * sizeof(int32_t);
  } else {
    size_qdqparam = QDQparam_size * num_qdq_nodes * sizeof(int32_t);
  }

  // Create input/output BOs
  const size_t A_BO_SIZE = (((kernel_x_shape_[0] * kernel_x_shape_[1]) +
                             (kernel_y_shape_[0] * kernel_y_shape_[1])) *
                            a_dtype_size_); // TODO:: add batch dimension also
  const size_t B_BO_SIZE = size_qdqparam;
  const size_t C_BO_SIZE = (kernel_x_shape_[0] * kernel_y_shape_[1] *
                            c_dtype_size_); // TODO: add batch dimension also

  RYZENAI_LOG_TRACE("actmatmul: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE size:" + std::to_string(C_BO_SIZE));

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
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

  std::vector<size_t> param_shape = {kernel_x_shape_[0], kernel_x_shape_[1],
                                     kernel_y_shape_[1]};
  if (is_ctrl_pkt_) {
    // Based on the mapped_shape to get the meta json file
    std::vector<uint8_t> json_data;
    try {
      auto json_key =
          get_instr_key(param_fname_prefix_, param_shape) + "_ctrl_meta";
      Transaction &txn = Transaction::getInstance();
      json_data = txn.get_txn_bvec(json_key);
    } catch (...) {
      is_ctrl_pkt_ = 0;
    }

    if (is_ctrl_pkt_) {
      std::cout << "ctrlpkt patching" << std::endl;
      RYZENAI_LOG_TRACE("act_act_matmul patch ctrlpkt ... START");
      // get param_bo address
      auto param_bo_key =
          get_instr_key(param_fname_prefix_, param_shape) + "_param";
      const xrt::bo &param_bo =
          xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;

      // Get ctrl pkt patch info from json
      std::vector<CtrlPktPatchInfo> ctrlpkt_info;
      ctrlpkt_info = json_str_to_ctrlpkt_patch_info(json_data);

      // Get the ctrl pkt
      auto ctrl_bo_key =
          get_instr_key(param_fname_prefix_, param_shape) + "_ctrl";
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
      RYZENAI_LOG_TRACE("act_act_matmul patch ctrlpkt ... DONE");
    }
  }
  RYZENAI_LOG_TRACE("act_act_matmul initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void act_act_matmul<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                             std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 2) {
    throw std::runtime_error("ELWADD IPU Wrapper expect to have two inputs.");
  }
  const int act1_idx = 0;
  const int act2_idx = 1;

  // The first data is a and second data is b
  InT *a = (InT *)input.at(act1_idx).data;
  InT *b = (InT *)input.at(act2_idx).data;

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

  a_shape_[0] = input.at(act1_idx).shape.at(0);
  a_shape_[1] = input.at(act1_idx).shape.at(1);

  size_t M, K, N;
  M = input.at(act1_idx).shape.at(0);
  K = input.at(act1_idx).shape.at(1);
  N = input.at(act2_idx).shape.at(0);

  c_shape_[0] = M;
  c_shape_[1] = N;

  size_t a_size = M * K * sizeof(InT);
  size_t b_size = N * K * sizeof(InT);
  RYZENAI_LOG_TRACE("act1 matmul: a_size:" + std::to_string(a_size));
  RYZENAI_LOG_TRACE("act2 matmul: a_size:" + std::to_string(a_size));

  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  memcpy((void *)a_bo_map, (void *)a, a_size);
  memcpy((void *)(reinterpret_cast<int8_t *>(a_bo_map) + a_size), (void *)b,
         b_size);

  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  // prepare inst_bo and param_bo
  std::vector<size_t> param_shape = {M, K, N};
  auto instr_bo_key = get_instr_key(txn_fname_prefix_, param_shape);
  auto param_bo_key =
      get_instr_key(param_fname_prefix_, param_shape) + "_param";
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  const xrt::bo &param_bo =
      xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;
  size_t instr_bo_words = instr_bo.size() / sizeof(int);

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
         c_shape_[0] * c_shape_[1] * sizeof(OutT));
  auto c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = static_cast<int64_t>(c_copy_stop - c_copy_start);
  auto exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(act_act_matmul_id_) + " " + std::to_string(a_shape_[0]) +
      " " + std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) +
      " " + std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> act_act_matmul<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  auto Act1_shape = extract_shape(input.at(0));
  auto Act2_shape = extract_shape(input.at(1));

  std::vector<size_t> param_shape = {Act1_shape[0], Act1_shape[1],
                                     Act2_shape[0]};
  std::string txn_key = get_instr_key(txn_fname_prefix_, param_shape);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
act_act_matmul<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  auto Act1_shape = extract_shape(input.at(0));
  auto Act2_shape = extract_shape(input.at(1));

  std::vector<size_t> param_shape = {Act1_shape[0], Act1_shape[1],
                                     Act2_shape[0]};
  std::string param_key =
      get_instr_key(param_fname_prefix_, param_shape) + "_param";
  // std::cout << "Super kernel params name : " << fname << std::endl;

  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(param_key);
}

template <typename InT, typename WtT, typename OutT>
std::vector<uint8_t> act_act_matmul<InT, WtT, OutT>::get_ctrl_pkts(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto Act1_shape = extract_shape(input.at(0));
  auto Act2_shape = extract_shape(input.at(1));

  std::vector<size_t> param_shape = {Act1_shape[0], Act1_shape[1],
                                     Act2_shape[0]};
  // TODO: Add check to validate tensor shapes
  std::string ctrl_key =
      get_instr_key(param_fname_prefix_, param_shape) + "_ctrl";
  try {
    Transaction &txn = Transaction::getInstance();
    return txn.get_txn_bvec(ctrl_key);
  } catch (...) {
    return {};
  }
}

template <typename InT, typename WtT, typename OutT>
std::vector<CtrlPktPatchInfo>
act_act_matmul<InT, WtT, OutT>::get_ctrl_pkt_patch_info(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto Act1_shape = extract_shape(input.at(0));
  auto Act2_shape = extract_shape(input.at(1));

  std::vector<size_t> param_shape = {Act1_shape[0], Act1_shape[1],
                                     Act2_shape[0]};
  // TODO: Add check to validate tensor shapes

  try {
    auto ctrl_pkt_meta =
        get_instr_key(param_fname_prefix_, param_shape) + "_ctrl_meta";
    Transaction &txn = Transaction::getInstance();
    return json_str_to_ctrlpkt_patch_info(txn.get_txn_bvec(ctrl_pkt_meta));
  } catch (...) {
    // throw std::runtime_error("act_act_matmul : Can not file the
    // ctrl_meta.json file");
    return {};
  }
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> act_act_matmul<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  // [QKV, qdq_params]
  if (input.size() != 4) {
    throw std::runtime_error("actmatmul: Incorrect number of tensors received");
  }

  size_t size_qdqparam;
  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    size_qdqparam = QDQparam_size * sizeof(int32_t);
  } else {
    size_qdqparam = QDQparam_size * num_qdq_nodes * sizeof(int32_t);
  }
  auto Act1_shape = extract_shape(input.at(0));
  auto Act2_shape = extract_shape(input.at(1));
  auto out_shape = extract_shape(input.at(3));

  size_t Act1_size = (Act1_shape[0] * Act1_shape[1] * sizeof(InT));
  size_t Act2_size = (Act2_shape[0] * Act2_shape[1] * sizeof(InT));

  size_t out_size = (out_shape[0] * out_shape[1] * sizeof(OutT));

  size_t super_kernel_size = get_super_kernel_params(input, output).size();
  size_t ctrl_pkt_size = get_ctrl_pkts(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, Act1_size},
      {OpArgMap::OpArgType::INPUT, 1, 1, Act1_size, Act2_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 2, 0, size_qdqparam},
      {OpArgMap::OpArgType::OUTPUT, 0, 3, 0, out_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size},
      {OpArgMap::OpArgType::CTRL_PKT_BIN, 4, 0, 0, ctrl_pkt_size}};

  return arg_map;
}

template class act_act_matmul<uint8_t, uint8_t, uint16_t>;
template class act_act_matmul<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
