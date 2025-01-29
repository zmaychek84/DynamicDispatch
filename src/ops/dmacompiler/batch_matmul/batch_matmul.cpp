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

#include <utils/instruction_registry.hpp>
#include <xrt_context/xrt_context.hpp>

#include "txn_container.hpp"

#include <ops/dmacompiler/batch_matmul/batch_matmul.hpp>
#include <ops/op_interface.hpp>
#include <ops/ops_common/ctrlpkt.hpp>
#include <txn_container.hpp>
#include <utils/instruction_registry.hpp>

#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>
#include <xrt_context/xrt_context.hpp>

// AIE Driver header
#include "ops/ops_common/matmul_matrix.hpp"
#include "utils/ctrl_pkt_utils.hpp"
#include "xaiengine.h"

using namespace matmul_matrix;

namespace ryzenai {

static std::tuple<size_t, size_t, size_t, size_t>
extract_BMKN(const std::vector<Tensor> &inputs) {
  size_t B, M, K, N;
  if (inputs.at(0).shape.size() == 3) { // has batch_dim
    B = inputs.at(0).shape.at(0);
    M = inputs.at(0).shape.at(1);
    K = inputs.at(0).shape.at(2);
    N = inputs.at(1).shape.at(2);
  } else {
    throw std::runtime_error("Input Shape is not supported");
  }

  return std::make_tuple(B, M, K, N);
}

template <typename InT, typename WtT, typename OutT>
std::tuple<size_t, size_t, size_t, size_t>
batch_matmul<InT, WtT, OutT>::map_padded_shape(size_t B, size_t M, size_t K,
                                               size_t N) const {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<batch_matrix_shapes> &supported_shapes = iter->second;
  size_t Bo = B;
  size_t Mo = M;
  size_t Ko = K;
  size_t No = N;
  size_t fidx = 0;
  bool f_found;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    if (B == mat.B && M == mat.M && K == mat.K && N == mat.N) {
      fidx = i;
      f_found = true;
      break;
    }
  }

  if (f_found) {
    iter = default_shapes_.find(txn_fname_prefix_);
    const std::vector<batch_matrix_shapes> &actual_shapes = iter->second;
    auto mat = actual_shapes.at(fidx);
    Bo = mat.B;
    Mo = mat.M;
    Ko = mat.K;
    No = mat.N;
  } else {
    throw std::runtime_error("Cannot find the shape");
  }
  return std::make_tuple(Bo, Mo, Ko, No);
}

/* Utility function to set the kernel shape based on the weights dimensions
 * Need to fix this to pick shapes independent of the datatype*/
template <typename InT, typename WtT, typename OutT>
void batch_matmul<InT, WtT, OutT>::set_kernel_shapes() {
  kernel_x_shape_[0] = w_shape_[1];
  kernel_x_shape_[1] =
      w_shape_[0] * w_shape_[1]; // may not work for all BMM shapes

  kernel_y_shape_[0] =
      w_shape_[0] * w_shape_[1]; // may not work for all BMM shapes
  kernel_y_shape_[1] = w_shape_[2];

  kernel_z_shape_[0] = w_shape_[0] * w_shape_[1];
  kernel_z_shape_[1] = w_shape_[2];
}

/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename WtT, typename OutT>
void batch_matmul<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  std::vector<batch_matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key = "batch_matmul_" +
               get_instr_key(txn_fname_prefix_, mat.B, mat.M, mat.K, mat.N);
    auto param_key =
        "batch_matmul_" +
        get_instr_key(param_fname_prefix_, mat.B, mat.M, mat.K, mat.N) +
        "_param";
    instructions.push_back(std::make_pair(key, false));
    layer_params.push_back(std::make_pair(param_key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
  xrt_ctx_->get_registry().add_layer_params(layer_params);
}
template <typename InT, typename WtT, typename OutT>
std::string batch_matmul<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                                        size_t b, size_t m,
                                                        size_t k,
                                                        size_t n) const {
  return prefix + "_" + std::to_string(b) + "_" + std::to_string(m) + "_" +
         std::to_string(k) + "_" + std::to_string(n);
}

/*
 * batch_matmul class constructor
 *
 * @param kernel_x_shape tuple containing of M x K dimension base batch_matmul
 * supported on IPU
 * @param kernel_y_shape tuple containing of K x N dimension base batch_matmul
 * supported on IPU
 *
 * NOTE: If the input shape has a smaller M dimension than the kernel
 * shape initialized here, the execute function can transparently
 * call a smaller GeMM to reduce padding overhead. The kernel
 * shape passed here should have largest supported M dimension.
 *
 */
template <typename InT, typename WtT, typename OutT>
batch_matmul<InT, WtT, OutT>::batch_matmul(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"int8", "a8"}, {"uint8", "a8"}, {"uint16", "a16"}};

  txnbin_b_header = {{"int8", "w8"}, {"uint8", "w8"}};

  txnbin_acc_header = {
      {"uint16", "acc16"}, {"int8", "acc8"}, {"uint8", "acc8"}};

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_["batch_matmul_4x4_a16w8acc16"] =
      std::vector<batch_matrix_shapes>{};
  default_shapes_["batch_matmul_4x4_a16w8acc16"].emplace_back(12, 64, 64, 512);

  // raw shape is the actual shape from ONNX
  raw_shapes_["batch_matmul_4x4_a16w8acc16"] =
      std::vector<batch_matrix_shapes>{};
  raw_shapes_["batch_matmul_4x4_a16w8acc16"].emplace_back(12, 64, 64, 512);

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  batch_matmul_id_ = batch_matmul_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dd_base_dir() + ryzenai::mdsqr_A8W8_QDQ_XCLBIN_PATH;

  if (a_dtype_ == "uint16") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mxpzi_A16W8_QDQ_XCLBIN_PATH;
  }

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME));

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
  txn_fname_prefix_ = "batch_matmul_4x2_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
  param_fname_prefix_ = "batch_matmul_4x2_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    txn_fname_prefix_ = "batch_matmul_4x4_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);
    param_fname_prefix_ = "batch_matmul_4x4_" + txnbin_a_header.at(a_dtype_) +
                          txnbin_b_header.at(b_dtype_) +
                          txnbin_acc_header.at(c_dtype_);

    if (attr.count("input_shape") &&
        attr.at("input_shape").type() == typeid(std::vector<int>)) {
      const auto &input_shape_vector =
          std::any_cast<const std::vector<int> &>(attr.at("input_shape"));

      if (input_shape_vector.size() == 3) {
        inputShape_[0] = input_shape_vector[0];
        inputShape_[1] = input_shape_vector[1];
        inputShape_[2] = input_shape_vector[2];
      } else {
        std::cout
            << "Input Shape attribute does not have the expected number of "
               "elements.Number of passed : input_shape_vector.size(), "
               "Expected:4"
            << std::endl;
      }
      RYZENAI_LOG_TRACE(
          "iConv: InputShape: " + std::to_string(input_shape_vector[0]) + ", " +
          std::to_string(input_shape_vector[1]) + ", " +
          std::to_string(input_shape_vector[2]) + ", " +
          std::to_string(input_shape_vector[3]));
    } else {
      std::cout << "Input Shape attribute not found or not of correct type."
                << std::endl;
    }
  }

  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("txn_fname_prefix : {}", txn_fname_prefix_));
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("param_fname_prefix : {}", param_fname_prefix_));

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  KERNEL_M_MAX = 4096;

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
        "batch_matmul_id M K N kernel_m kernel_k kernel_n Execute"
        "time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[GEMM] ID: " + std::to_string(batch_matmul_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype_ + ", " +
                    b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void batch_matmul<InT, WtT, OutT>::set_params(const std::string &model_name,
                                              std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;
  if (model_name == "4x4PSW1.0") {
    is_ctrl_pkt_ = 1;
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::PSW1_0_A16W8_QDQ_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }

  auto [B, M, K, N] = map_padded_shape(input_shape.at(0), input_shape.at(1),
                                       input_shape.at(2), input_shape.at(3));
  KERNEL_M_MAX = M;

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

/*
 * copy weight matrix into XRT BOs with padding and tiling
 *
 * this method copies the weight matrix into XRT BOs. This is re-used for
 * every batch_matmul performed for this object with different activations.
 * weight matrix is padded, tiled and reformatted while copying to XRT BOs.
 * padding is done to align with kernel_y_shape each tile of the weight matrix
 * is of shape kernel_y_shape this method also reformats the matrix b/weight
 * matrix as required by AIE/IPU batch_matmul implementation
 *
 * @param weights pointer to the weight matrix
 * @param w_shape tuple containing the shape of the weight matrix
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void batch_matmul<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("batch_matmul initialize_const_params(ptr) ...");

  DD_THROW_IF(
      (const_params.size() != 3) || (const_params.at(0).shape.size() != 3),
      OpsFusion::dd_format("Unsupported const spec for batch_matmul\n") +
          OpsFusion::dd_format(
              "(Details : #const params == 1 ({}), Const param dim == 2 ({})",
              const_params.size(), const_params.at(0).shape.size()));

  w_shape_[0] = const_params.at(0).shape.at(0);
  w_shape_[1] = const_params.at(0).shape.at(1);
  w_shape_[2] = const_params.at(0).shape.at(2);

  // auto K_raw = w_shape_[0];
  // auto N_raw = w_shape_[1];

  auto weights = (WtT *)const_params.at(0).data;
  auto qdq = (int64_t *)const_params.at(1).data;
  auto qdq_params = (int32_t *)const_params.at(2).data;

  auto Ksubv = matmul_matrix::Ksubv_PSW_BMM;
  auto Msubv = matmul_matrix::Msubv_PSW_BMM;
  auto Nsubv = matmul_matrix::Nsubv_PSW_BMM;

  if (design_param_.find("4x4") != std::string::npos) { // psw1.0 4x4 design
    set_kernel_shapes();
  }

  std::vector<WtT> buf(w_shape_[0] * w_shape_[1] * w_shape_[2], 0);
  if (design_param_.find("4x4") != std::string::npos) { // psw1.0 4x4 design
    Ksubv = matmul_matrix::Ksubv_PSW_BMM;
    Msubv = matmul_matrix::Msubv_PSW_BMM;
    Nsubv = matmul_matrix::Nsubv_PSW_BMM;

    matmul_matrix::WgtMatrix<WtT> W(
        (int)(w_shape_[0] * w_shape_[1]), (int)w_shape_[2],
        matmul_matrix::Ksubv_PSW_BMM, matmul_matrix::Nsubv_PSW_BMM, buf.data());
    for (int r = 0; r < w_shape_[0] * w_shape_[1]; ++r) {
      for (int c = 0; c < w_shape_[2]; ++c) {
        W.at(r, c) = weights[(r * w_shape_[2]) + c];
      }
    }
  }

  std::vector<int64_t> qdq_buf(
      (w_shape_[0] * w_shape_[1]) / Ksubv * w_shape_[2], 0);
  memcpy((void *)&qdq_buf[0], (void *)&qdq[0],
         (w_shape_[0] * w_shape_[1]) / Ksubv * w_shape_[2] * sizeof(int64_t));

  // padding Msubv and Nsubv
  qdq_params[qdq_Mv_idx] = Msubv;
  qdq_params[qdq_Nv_idx] = Nsubv;

  auto total_size = Ksubv * Nsubv;
  auto qdq_size = Nsubv * sizeof(int64_t);
  auto qdq_params_size = matmul_matrix::QDQparam_size * sizeof(int32_t);
  //// WGT + Bias(all zeros)
  { // This section of the code interleaves bias with weights Nsubv of bias
    // with every K x N
    size_t write_offset = 0;
    for (int N_shard = 0; N_shard < (w_shape_[2]) / (Nsubv); N_shard++) {
      for (int K_shard = 0; K_shard < (w_shape_[0] * w_shape_[1]) / (Ksubv);
           K_shard++) {
        io.write(write_offset,
                 (void *)&buf[(N_shard * w_shape_[0] * w_shape_[1] * Nsubv) +
                              (K_shard * total_size)],
                 (total_size));
        write_offset += total_size;
        io.write(write_offset,
                 (void *)&qdq_buf[K_shard * w_shape_[2] + N_shard * Nsubv],
                 qdq_size);
        write_offset += qdq_size;
      }
    }
    io.write(write_offset, (void *)qdq_params, qdq_params_size);
  }

  RYZENAI_LOG_TRACE("batch_matmul initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void batch_matmul<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("batch_matmul initialize_const_params ...");

  DD_THROW_IF(
      (const_params.size() != 3) || (const_params.at(0).shape.size() != 3),
      OpsFusion::dd_format("Unsupported const spec for batch_matmul\n") +
          OpsFusion::dd_format(
              "(Details : #const params == 1 ({}), Const param dim == 2 ({})",
              const_params.size(), const_params.at(0).shape.size()));

  w_shape_[0] = const_params.at(0).shape.at(0);
  w_shape_[1] = const_params.at(0).shape.at(1);
  w_shape_[2] = const_params.at(0).shape.at(2);

  int Ksubv;
  size_t M;
  if (design_param_.find("4x4") != std::string::npos) { // PSW 1.0 4x4 design
    set_kernel_shapes();
    M = 64;
    Ksubv = 64;
  }

  // qdqc
  size_t size_interleaved_qdq =
      w_shape_[0] * w_shape_[1] * w_shape_[2] / Ksubv * sizeof(int64_t);

  size_interleaved_qdq += matmul_matrix::QDQparam_size * sizeof(int32_t);

  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  /* Create input/output BOs */
  const size_t B_BO_SIZE =
      (kernel_y_shape_[0] * kernel_y_shape_[1] * b_dtype_size_ +
       size_interleaved_qdq);
  const size_t A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_);
  const size_t C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_);

  RYZENAI_LOG_TRACE("GEMM: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  auto b_copy_start = GET_ELAPSED_TIME_NS();
  auto b_format_start = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = b_bo_.map<WtT *>();
  auto bo_const = BoConst(b_bo_map);
  initialize_const_params(bo_const, const_params);
  auto b_format_stop = GET_ELAPSED_TIME_NS();
  b_format_time_ += static_cast<int64_t>(b_format_stop - b_format_start);
  auto b_copy_stop = GET_ELAPSED_TIME_NS();
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();
  b_copy_time_ = static_cast<int64_t>(b_copy_stop - b_copy_start);
  b_sync_time_ = static_cast<int64_t>(b_sync_stop - b_sync_start);

  if (is_ctrl_pkt_) {
    // Based on the mapped_shape to get the meta json file
    auto [Bo, Mo, Ko, No] =
        map_padded_shape((size_t)w_shape_[0], (size_t)w_shape_[1],
                         (size_t)w_shape_[1], (size_t)w_shape_[2]);
    std::vector<uint8_t> json_data;
    try {
      auto json_key = "batch_matmul_" +
                      get_instr_key(param_fname_prefix_, Bo, Mo, Ko, No) +
                      "_ctrl_meta";
      Transaction &txn = Transaction::getInstance();
      json_data = txn.get_txn_bvec(json_key);
    } catch (...) {
      is_ctrl_pkt_ = 0;
    }

    if (is_ctrl_pkt_) {
      std::cout << "ctrlpkt patching" << std::endl;
      RYZENAI_LOG_TRACE("batch_matmul patch ctrlpkt ... START");
      // get param_bo address
      auto param_bo_key = "batch_matmul_" +
                          get_instr_key(param_fname_prefix_, Bo, Mo, Ko, No) +
                          "_param";
      const xrt::bo &param_bo =
          xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;

      // Get ctrl pkt patch info from json
      std::vector<CtrlPktPatchInfo> ctrlpkt_info;
      ctrlpkt_info = json_str_to_ctrlpkt_patch_info(json_data);

      // Get the ctrl pkt
      auto ctrl_bo_key = "batch_matmul_" +
                         get_instr_key(param_fname_prefix_, Bo, Mo, Ko, No) +
                         "_ctrl";
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
      RYZENAI_LOG_TRACE("batch_matmul patch ctrlpkt ... DONE");
    }
  }
  RYZENAI_LOG_TRACE("batch_matmul initialize_const_params ... DONE");
}
/*
 * execute matrix multiplication c = a * w
 *
 * perform batch_matmul c = a * w. w is stored in the object with
 * initilize_weights method.
 *
 * @param a pointer to activation matrix
 * @param a_shape tuple containing the shape of the activation matrix
 * @param c pointer to store the result of batch_matmul
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void batch_matmul<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                           std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("batch_matmul execute ...");

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  auto exec_start = GET_ELAPSED_TIME_NS();

  if (input.at(0).shape.size() == 3) {
    a_shape_[0] = input.at(0).shape.at(0);
    a_shape_[1] = input.at(0).shape.at(1);
    a_shape_[2] = input.at(0).shape.at(2);
  } else {
    throw std::runtime_error("batch_matmul : Invalid shape received for input");
  }

  if (output.at(0).shape.size() == 3) {
    c_shape_[0] = output.at(0).shape.at(0);
    c_shape_[1] = output.at(0).shape.at(1);
    c_shape_[2] = output.at(0).shape.at(2);
  } else {
    throw std::runtime_error(
        "batch_matmul : Invalid shape received for output");
  }

  //  RYZENAI_LOG_TRACE("GEMM: a_shape_[0]:" + std::to_string(a_shape_[0]) +
  //                    " a_shape_[1]:" + std::to_string(a_shape_[1]) +
  //                    " c_shape_[1]:" + std::to_string(c_shape_[1]));

  auto aie_out = (OutT *)output.at(0).data;
  auto a = (InT *)input.at(0).data;

  auto [B, M, K, N] =
      map_padded_shape(input.at(0).shape.at(0), input.at(0).shape.at(1),
                       input.at(0).shape.at(2), output.at(0).shape.at(2));

  auto a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  memcpy((void *)a_bo_map, (void *)a,
         (a_shape_[0] * a_shape_[1] * a_shape_[2] * a_dtype_size_));

  auto a_copy_stop = GET_ELAPSED_TIME_NS();
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();
  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  // INIT with zeros
  auto instr_bo_key =
      "batch_matmul_" + txn_fname_prefix_ + "_" + std::to_string(a_shape_[0]) +
      "_" + std::to_string(a_shape_[1]) + "_" + std::to_string(a_shape_[2]) +
      "_" + std::to_string(c_shape_[2]);
  auto param_bo_key = "batch_matmul_" + param_fname_prefix_ + "_" +
                      std::to_string(a_shape_[0]) + "_" +
                      std::to_string(a_shape_[1]) + "_" +
                      std::to_string(a_shape_[2]) + "_" +
                      std::to_string(c_shape_[2]) + "_param";

  // std::cout << instr_bo_key << std::endl;
  // std::cout << param_bo_key << std::endl;

  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  const xrt::bo &param_bo =
      xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();

  // launch the GEMM kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  // kernel call for GEMM that supports transaction binary flow
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  ryzenai::dynamic_dispatch::execute_kernel(
      kernel_, 2, instr_bo, instr_bo_words, c_bo_, a_bo_, b_bo_, param_bo,
      ctrl_bo_, true, is_ctrl_pkt_);
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  // sync output activation to host memory
  auto c_sync_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  OutT *c_bo_map = c_bo_.map<OutT *>();
  auto c_sync_stop = GET_ELAPSED_TIME_NS();
  c_sync_time_ += static_cast<int64_t>(c_sync_stop - c_sync_start);
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);

  memcpy((void *)aie_out, (void *)c_bo_map,
         (c_shape_[0] * c_shape_[1] * c_shape_[2] * c_dtype_size_));

  auto exec_end = GET_ELAPSED_TIME_NS();
  RYZENAI_LOG_INFO(
      std::to_string(batch_matmul_id_) + " " + std::to_string(a_shape_[0]) +
      " " + std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) +
      " " + std::to_string(kernel_x_rows) + " " +
      std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
  RYZENAI_LOG_TRACE("batch_matmul execute ... DONE");
}

/*
 * method to set debug flag
 *
 * When the debug flag is set, execute method will write input, weights and
 * output matricies to a filed. the filename will be
 * ryzenai_qlinear2_<execute_num>_<matrix>.txt
 *
 * @param debug bool value to enable disable debug feature. turned off by
 * default
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void batch_matmul<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> batch_matmul<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [B, M, K, N] = extract_BMKN(input);
  auto [Bo, Mo, Ko, No] = map_padded_shape(B, M, K, N);
  std::string txn_key =
      "batch_matmul_" + get_instr_key(txn_fname_prefix_, Bo, Mo, Ko, No);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
batch_matmul<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [B, M, K, N] = extract_BMKN(input);
  auto [Bo, Mo, Ko, No] = map_padded_shape(B, M, K, N);
  // TODO: Add check to validate tensor shapes
  std::string param_key = "batch_matmul_" +
                          get_instr_key(param_fname_prefix_, Bo, Mo, Ko, No) +
                          "_param";
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(param_key);
}

template <typename InT, typename WtT, typename OutT>
std::vector<uint8_t> batch_matmul<InT, WtT, OutT>::get_ctrl_pkts(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [B, M, K, N] = extract_BMKN(input);
  auto [Bo, Mo, Ko, No] = map_padded_shape(B, M, K, N);
  // TODO: Add check to validate tensor shapes
  std::string ctrl_key = "batch_matmul_" +
                         get_instr_key(param_fname_prefix_, Bo, Mo, Ko, No) +
                         "_ctrl";
  try {
    Transaction &txn = Transaction::getInstance();
    return txn.get_txn_bvec(ctrl_key);
  } catch (...) {
    return {};
  }
}

template <typename InT, typename WtT, typename OutT>
std::vector<CtrlPktPatchInfo>
batch_matmul<InT, WtT, OutT>::get_ctrl_pkt_patch_info(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [B, M, K, N] = extract_BMKN(input);
  auto [Bo, Mo, Ko, No] = map_padded_shape(B, M, K, N);
  // TODO: Add check to validate tensor shapes
  try {
    auto ctrl_pkt_meta = "batch_matmul_" +
                         get_instr_key(param_fname_prefix_, Bo, Mo, Ko, No) +
                         "_ctrl_meta";
    Transaction &txn = Transaction::getInstance();
    return json_str_to_ctrlpkt_patch_info(txn.get_txn_bvec(ctrl_pkt_meta));
  } catch (...) {
    // throw std::runtime_error("batch_matmul : Can not file the ctrl_meta.json
    // file");
    return {};
  }
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> batch_matmul<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  // input --> [input, weights, output]
  // Check if IO buffers have batch.
  auto [B, M, K, N] = extract_BMKN(input);
  auto [Bo, Mo, Ko, No] = map_padded_shape(B, M, K, N);
  int Ksubv;
  if (design_param_.find("4x4") != std::string::npos) { // mzdk5 4x4 design
    Ksubv = matmul_matrix::Ksubv_PSW_BMM;
  }

  // qdqc
  size_t size_interleaved_qdq = Bo * Ko * No / Ksubv * sizeof(int64_t);
  size_interleaved_qdq += matmul_matrix::QDQparam_size * sizeof(int32_t);

  size_t const_params_bo_size =
      (Bo * Ko * No * b_dtype_size_) + size_interleaved_qdq;
  size_t input_bo_size = (Bo * Mo * Ko * a_dtype_size_);
  size_t output_bo_size = (Bo * Mo * No * c_dtype_size_);
  size_t super_kernel_size = get_super_kernel_params(input, output).size();
  size_t ctrl_pkt_size = get_ctrl_pkts(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, 4, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size},
      {OpArgMap::OpArgType::CTRL_PKT_BIN, 4, 0, 0, ctrl_pkt_size}};
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("batch_matmul Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
void batch_matmul<InT, WtT, OutT>::format_output(
    const Tensor &out_tensor, void *hw_out_ptr, size_t sz, size_t tensor_idx,
    const std::map<std::string, std::any> &attr) {
  // format_output(
  //     const Tensor &out_tensor, void *hw_out_ptr, size_t sz, int tensor_idx,
  //     const std::map<std::string, std::any> &attr) {
  size_t B, M, K, N;
  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));
    if (input_shape_vector.size() == 2) {
      M = input_shape_vector[0];
      K = input_shape_vector[1];
    } else if (input_shape_vector.size() == 3) {
      B = input_shape_vector[0];
      M = input_shape_vector[1];
      K = input_shape_vector[2];
    } else if (input_shape_vector.size() == 4) {
      M = input_shape_vector[0] * input_shape_vector[1] * input_shape_vector[2];
      K = input_shape_vector[3];
    } else {
      std::cout << "Input shape attribute does not have the expected number of "
                   "elements.Number of passed : design_param_vector.size(), "
                   "Expected:3"
                << std::endl;
    }
    RYZENAI_LOG_TRACE("batch_matmul: input_shape: " + std::to_string(M) + ", " +
                      std::to_string(K));
  } else {
    std::cout << "Input Shape attribute not found or not of correct type."
              << std::endl;
  }

  if (attr.count("output_shape") &&
      attr.at("output_shape").type() == typeid(std::vector<int>)) {
    const auto &orig_output_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("output_shape"));
    if (orig_output_shape_vector.size() == 2) {
      N = orig_output_shape_vector[1];
    } else if (orig_output_shape_vector.size() == 3) {
      N = orig_output_shape_vector[2];
    } else if (orig_output_shape_vector.size() == 4) {
      N = orig_output_shape_vector[3];
    } else {
      std::cout
          << "output shape attribute does not have the expected number of "
             "elements.Number of passed : design_param_vector.size(), "
             "Expected:3"
          << std::endl;
    }
    RYZENAI_LOG_TRACE("batch_matmul: output_shape: " + std::to_string(M) +
                      ", " + std::to_string(N));
  } else {
    N = out_tensor.shape.at(2);
  }
  // get the mapped shape
  auto [Bo, Mo, Ko, No] = map_padded_shape(B, M, K, N);
  // K, N is the dst.shape
  auto aie_out = (void *)out_tensor.data;

  if (sz != Bo * Mo * No * c_dtype_size_) {
    throw std::runtime_error(
        "batch_matmul : The size of hw_out is not correct.");
  }

  if (N == No) {
    RYZENAI_LOG_TRACE("Triggering batch_matmul Output Memcpy");
    memcpy((void *)aie_out, (void *)hw_out_ptr, (Bo * M * No * c_dtype_size_));
  } else {
    RYZENAI_LOG_TRACE("Triggering batch_matmul Output Strided Memcpy");
    for (int i = 0; i < M; i++) {
      memcpy(
          (void *)(static_cast<uint8_t *>(aie_out) + i * N * c_dtype_size_),
          (void *)(static_cast<uint8_t *>(hw_out_ptr) + i * No * c_dtype_size_),
          (N * c_dtype_size_));
    }
  }
}

template <typename InT, typename WtT, typename OutT>
std::once_flag batch_matmul<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t batch_matmul<InT, WtT, OutT>::batch_matmul_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag batch_matmul<InT, WtT, OutT>::instr_reg_flag_;

template class batch_matmul<uint8_t, uint8_t, uint8_t>;
template class batch_matmul<uint16_t, uint8_t, uint16_t>;

} // namespace ryzenai
