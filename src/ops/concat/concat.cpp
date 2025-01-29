/*
 Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "utils/ctrl_pkt_utils.hpp"
#include <ops/concat/concat.hpp>
#include <ops/ops_common/ctrlpkt.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

namespace ryzenai {
static std::tuple<size_t, size_t> extract_MK(const Tensor &tensor) {
  size_t M = 0, K = 0;
  if (tensor.shape.size() == 2) {
    M = tensor.shape.at(0);
    K = tensor.shape.at(1);
  } else if (tensor.shape.size() == 3) {
    M = tensor.shape.at(1);
    K = tensor.shape.at(2);
  } else if (tensor.shape.size() == 4) {
    if (tensor.shape.at(1) == tensor.shape.at(2)) { // NHWC
      M = tensor.shape.at(1) * tensor.shape.at(2);
      K = tensor.shape.at(3);
    } else { // NCHW
      M = tensor.shape.at(2) * tensor.shape.at(3);
      K = tensor.shape.at(1);
    }
  } else {
    throw std::runtime_error("concat : Invalid shape received for Matrix");
  }

  return std::make_tuple(M, K);
}

template <typename InT, typename OutT>
std::once_flag concat<InT, OutT>::logger_flag_;

template <typename InT, typename OutT>
uint64_t concat<InT, OutT>::concat_count = 0;

template <typename InT, typename OutT>
std::once_flag concat<InT, OutT>::instr_reg_flag_;

template <typename InT, typename OutT>
void concat<InT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename OutT>
std::vector<size_t>
concat<InT, OutT>::map_padded_shape(std::vector<size_t> &in) const {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<std::vector<size_t>> &supported_shapes = iter->second;
  size_t fidx = 0;
  bool f_found = false;
  std::vector<size_t> out_mat;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    int64_t sum_diff = 0;
    int64_t diff = 0;
    for (int j = 0; j < in.size(); j++) {
      diff = mat[j] - in[j];
      diff = diff > 0 ? diff : (-diff);
      sum_diff += diff;
    }
    if (sum_diff == 0) {
      fidx = i;
      f_found = true;
      break;
    }
  }
  if (f_found) {
    iter = default_shapes_.find(txn_fname_prefix_);
    const std::vector<std::vector<size_t>> &actual_shapes = iter->second;
    out_mat = actual_shapes.at(fidx);
  } else {
    throw std::runtime_error("Can not find the shape");
  }
  // std::cout << Mo << ' ' << No << std::endl;
  return out_mat;
}

template <typename InT, typename OutT>
std::string concat<InT, OutT>::get_instr_key(std::string prefix,
                                             std::vector<size_t> &mat) const {
  std::string out_str = "concat_" + prefix;
  for (int i = 0; i < mat.size(); i++) {
    out_str += "_" + std::to_string(mat[i]);
  }
  return out_str;
}

template <typename InT, typename OutT>
void concat<InT, OutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  for (const auto &[mkey, value] : default_shapes_) {
    auto iter = default_shapes_.find(mkey);
    std::vector<std::vector<size_t>> &supported_shapes = iter->second;
    for (int i = 0; i < supported_shapes.size(); i++) {
      auto mat = supported_shapes.at(i);
      auto key = get_instr_key(mkey, mat);
      auto param_key = get_instr_key(mkey, mat) + "_param";
      instructions.push_back(std::make_pair(key, false));
      layer_params.push_back(std::make_pair(param_key, false));
    }
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
  xrt_ctx_->get_registry().add_layer_params(layer_params);
}

template <typename InT, typename OutT>
concat<InT, OutT>::concat(const std::string &a_dtype,
                          const std::string &c_dtype, bool load_xrt,
                          const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"uint16", "a16"}};
  txnbin_c_header = {{"uint16", "acc16"}};

  default_shapes_["concat_4x4_a16"] = std::vector<std::vector<size_t>>();
  default_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{64, 64, 64, 64, 19});
  default_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{256, 64, 256, 64, 19});
  default_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{1024, 64, 1024, 64, 9});
  default_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{4096, 64, 4096, 64, 4});
  default_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{4096, 640, 4096, 320, 1});
  default_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{64, 1280, 64, 1280, 1});
  default_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{256, 1280, 256, 640, 1});
  default_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{256, 1280, 256, 1280, 1});
  default_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{1024, 640, 1024, 320, 1});
  default_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{1024, 640, 1024, 640, 1});
  default_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{1024, 1280, 1024, 640, 1});
  default_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{4096, 320, 4096, 320, 1});

  raw_shapes_["concat_4x4_a16"] = std::vector<std::vector<size_t>>();
  raw_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{64, 64, 64, 64, 19});
  raw_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{256, 64, 256, 64, 19});
  raw_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{1024, 64, 1024, 64, 9});
  raw_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{4096, 64, 4096, 64, 4});
  raw_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{4096, 640, 4096, 320, 1});
  raw_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{64, 1280, 64, 1280, 1});
  raw_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{256, 1280, 256, 640, 1});
  raw_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{256, 1280, 256, 1280, 1});
  raw_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{1024, 640, 1024, 320, 1});
  raw_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{1024, 640, 1024, 640, 1});
  raw_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{1024, 1280, 1024, 640, 1});
  raw_shapes_["concat_4x4_a16"].push_back(
      std::vector<size_t>{4096, 320, 4096, 320, 1});

  a_dtype_ = a_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  c_dtype_size_ = sizeof(OutT);
  concat_id_ = concat_count++;

  std::string XCLBIN_FNAME =
      OpInterface::get_dd_base_dir() + ryzenai::mzdk54x4_A16W8_QDQ_XCLBIN_PATH;

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
    RYZENAI_LOG_TRACE("concat: DesignFormat: " + design_param_);
  }

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    txn_fname_prefix_ = "concat_4x4_" + txnbin_a_header.at(a_dtype_);

    param_fname_prefix_ = "concat_4x4_" + txnbin_a_header.at(a_dtype_);

  } else {
    throw std::runtime_error("CONCAT IPU Wrapper only support 4x4 design.");
  }

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;
  is_ctrl_pkt_ = 1;

  std::call_once(logger_flag_, []() {
    std::string header = "concat_id H W C kernel_h kernel_w kernel_c Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[CONCAT] ID: " + std::to_string(concat_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME + ", (a_dtype, c_dtype): (" +
                    a_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename OutT>
void concat<InT, OutT>::set_params(const std::string &model_name,
                                   std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME =
      OpInterface::get_dd_base_dir() + ryzenai::mzdk54x4_A16W8_QDQ_XCLBIN_PATH;

  kernel_x_shape_[0] = input_shape.at(0); // M
  kernel_x_shape_[1] = input_shape.at(1); // K

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename OutT>
void concat<InT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("concat initialize_const_params(ptr) ...");

  RYZENAI_LOG_TRACE("concat initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename OutT>
void concat<InT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("concat initialize_const_params(ptr) ...");

  RYZENAI_LOG_TRACE("concat initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename OutT>
void concat<InT, OutT>::execute(std::vector<Tensor> &input,
                                std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() <= 1) {
    throw std::runtime_error(
        "concat IPU Wrapper expect to have more than one input.");
  }
  const int a_idx = 0;
  const int b_idx = 1;
  const int c_idx = 0;
  // The first data is a
  InT *a = (InT *)input.at(a_idx).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;

  auto exec_start = GET_ELAPSED_TIME_NS();

  a_shape_[0] = input.at(a_idx).shape.at(0);
  a_shape_[1] = input.at(a_idx).shape.at(1);

  b_shape_[0] = input.at(b_idx).shape.at(0);
  b_shape_[1] = input.at(b_idx).shape.at(1);

  // number of B matrix
  size_t N = input.size() - 1;
  std::vector<size_t> input_shape = {a_shape_[0], a_shape_[1], b_shape_[0],
                                     b_shape_[1], static_cast<size_t>(N)};
  auto mapped_shape = map_padded_shape(input_shape);

  c_shape_[0] = output.at(c_idx).shape.at(0);
  c_shape_[1] = output.at(c_idx).shape.at(1);

  const auto a_bo_size_in_bytes = (mapped_shape[0] * mapped_shape[1] +
                                   N * mapped_shape[2] * mapped_shape[3]) *
                                  a_dtype_size_;
  const auto c_bo_size_in_bytes = a_bo_size_in_bytes;

  /* Create input/output BOs */
  const size_t A_BO_SIZE = a_bo_size_in_bytes;
  const size_t C_BO_SIZE = c_bo_size_in_bytes;
  RYZENAI_LOG_TRACE("concat: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));

  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));

  size_t a_size = a_shape_[0] * a_shape_[1] * sizeof(InT);
  RYZENAI_LOG_TRACE("concat: a_size:" + std::to_string(a_size));

  size_t b_size = b_shape_[0] * b_shape_[1] * sizeof(InT);
  RYZENAI_LOG_TRACE("concat: b_size:" + std::to_string(b_size));

  if (is_ctrl_pkt_) {
    // Based on the mapped_shape to get the meta json file
    std::vector<uint8_t> json_data;
    try {
      auto json_key =
          get_instr_key(param_fname_prefix_, mapped_shape) + "_ctrl_meta";
      Transaction &txn = Transaction::getInstance();
      json_data = txn.get_txn_bvec(json_key);
    } catch (...) {
      is_ctrl_pkt_ = 0;
    }

    if (is_ctrl_pkt_) {
      std::cout << "ctrlpkt patching" << std::endl;
      RYZENAI_LOG_TRACE("concat patch ctrlpkt ... START");
      // get param_bo address
      auto param_bo_key =
          get_instr_key(param_fname_prefix_, mapped_shape) + "_param";
      const xrt::bo &param_bo =
          xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;

      // Get ctrl pkt patch info from json
      std::vector<CtrlPktPatchInfo> ctrlpkt_info;
      ctrlpkt_info = json_str_to_ctrlpkt_patch_info(json_data);

      // Get the ctrl pkt
      auto ctrl_bo_key =
          get_instr_key(param_fname_prefix_, mapped_shape) + "_ctrl";
      std::string ctrl_params =
          Transaction::getInstance().get_txn_str(ctrl_bo_key);
      std::vector<char> ctrl_buffer(ctrl_params.begin(), ctrl_params.end());

      // ctrl pkt patch
      std::vector<char> ctrl_pkt_new;
      std::vector<uint64_t> buffer_addrs = {
          uint64_t(c_bo_.address() + DDR_AIE_ADDR_OFFSET),
          uint64_t(a_bo_.address() + DDR_AIE_ADDR_OFFSET),
          uint64_t(0 + DDR_AIE_ADDR_OFFSET), // No b_bo/const bo for concat
          uint64_t(param_bo.address() + DDR_AIE_ADDR_OFFSET)};
      ctrl_pkt_new = patch_ctrl_bin(ctrl_buffer, ctrlpkt_info, buffer_addrs);

      size_t ctrl_bo_words = ctrl_pkt_new.size();
      ctrl_bo_ =
          xrt::bo(xrt_ctx_->get_device(), ctrl_bo_words, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
      ctrl_bo_.write(ctrl_pkt_new.data());
      ctrl_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      RYZENAI_LOG_TRACE("concat patch ctrlpkt ... DONE");
    }
  }

  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  memcpy((void *)a_bo_map, (void *)a, a_size);
  size_t offset = a_shape_[0] * a_shape_[1];
  for (int i = 0; i < N; i++) {
    InT *b = (InT *)input.at(b_idx + i).data;
    memcpy((void *)(a_bo_map + offset + i * b_shape_[0] * b_shape_[1]),
           (void *)b, b_size);
  }
  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  // prepare inst_bo and param_bo
  auto instr_bo_key = get_instr_key(txn_fname_prefix_, mapped_shape);
  auto param_bo_key =
      get_instr_key(param_fname_prefix_, mapped_shape) + "_param";
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  const xrt::bo &param_bo =
      xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;
  size_t instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  ryzenai::dynamic_dispatch::execute_kernel(
      kernel_, 2, instr_bo, instr_bo_words, c_bo_, a_bo_, 0, param_bo, ctrl_bo_,
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
      std::to_string(concat_id_) + " " + std::to_string(a_shape_[0]) + " " +
      std::to_string(a_shape_[1]) + " " + std::to_string(a_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename OutT>
const std::vector<uint8_t> concat<InT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [MA, KA] = extract_MK(input.at(0));
  auto [MB, KB] = extract_MK(input.at(1));
  size_t N = input.size() - 2;
  std::vector<size_t> input_shape = {MA, KA, MB, KB, N};
  auto mapped_shape = map_padded_shape(input_shape);
  std::string txn_key = get_instr_key(txn_fname_prefix_, mapped_shape);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename OutT>
const std::vector<uint8_t> concat<InT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [MA, KA] = extract_MK(input.at(0));
  auto [MB, KB] = extract_MK(input.at(1));
  size_t N = input.size() - 2;
  std::vector<size_t> input_shape = {MA, KA, MB, KB, N};
  auto mapped_shape = map_padded_shape(input_shape);
  // TODO: Add check to validate tensor shapes
  std::string param_key =
      get_instr_key(param_fname_prefix_, mapped_shape) + "_param";
  // std::cout << "Super kernel params name : " << fname << std::endl;
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(param_key);
}

template <typename InT, typename OutT>
std::vector<uint8_t> concat<InT, OutT>::get_ctrl_pkts(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [MA, KA] = extract_MK(input.at(0));
  auto [MB, KB] = extract_MK(input.at(1));
  size_t N = input.size() - 2;
  std::vector<size_t> input_shape = {MA, KA, MB, KB, N};
  auto mapped_shape = map_padded_shape(input_shape);
  // TODO: Add check to validate tensor shapes
  std::string ctrl_key =
      get_instr_key(param_fname_prefix_, mapped_shape) + "_ctrl";
  // std::cout << "Super kernel params name : " << fname << std::endl;
  try {
    Transaction &txn = Transaction::getInstance();
    return txn.get_txn_bvec(ctrl_key);
  } catch (...) {
    return {};
  }
}

template <typename InT, typename OutT>
std::vector<CtrlPktPatchInfo> concat<InT, OutT>::get_ctrl_pkt_patch_info(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [MA, KA] = extract_MK(input.at(0));
  auto [MB, KB] = extract_MK(input.at(1));
  size_t N = input.size() - 2;
  std::vector<size_t> input_shape = {MA, KA, MB, KB, N};
  auto mapped_shape = map_padded_shape(input_shape);
  // TODO: Add check to validate tensor shapes
  try {
    auto ctrl_pkt_meta =
        get_instr_key(param_fname_prefix_, mapped_shape) + "_ctrl_meta";
    Transaction &txn = Transaction::getInstance();
    return json_str_to_ctrlpkt_patch_info(txn.get_txn_bvec(ctrl_pkt_meta));
  } catch (...) {
    // throw std::runtime_error("concat : Can not file the ctrl_meta.json
    // file");
    return {};
  }
}

template <typename InT, typename OutT>
std::vector<OpArgMap> concat<InT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [MA, KA] = extract_MK(input.at(0));
  auto [MB, KB] = extract_MK(input.at(1));
  size_t N = input.size() - 2;
  std::vector<size_t> input_shape = {MA, KA, MB, KB, N};
  auto mapped_shape = map_padded_shape(input_shape);
  MA = mapped_shape[0];
  KA = mapped_shape[1];
  MB = mapped_shape[2];
  KB = mapped_shape[3];
  auto out_shape = MA * KA + MB * KB * N;

  size_t input_a_bo_size = (MA * KA * sizeof(InT));
  size_t input_b_bo_size = (MB * KB * sizeof(InT));
  size_t const_params_bo_size = 16; // Dummy Buffer
  size_t output_bo_size = (out_shape * sizeof(OutT));
  size_t super_kernel_size = get_super_kernel_params(input, output).size();
  size_t ctrl_pkt_size = get_ctrl_pkts(input, output).size();

  std::vector<OpArgMap> arg_map;
  struct OpArgMap inmap = {OpArgMap::OpArgType::INPUT, 1, 0, 0,
                           input_a_bo_size};
  arg_map.push_back(inmap);
  for (size_t n = 0; n < N; n++) {
    inmap = {OpArgMap::OpArgType::INPUT, 1, 1 + n,
             input_a_bo_size + n * input_b_bo_size, input_b_bo_size};
    arg_map.push_back(inmap);
  }
  inmap = {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0,
           const_params_bo_size}, // Dummy allocation
      arg_map.push_back(inmap);
  inmap = {OpArgMap::OpArgType::OUTPUT, 0, input.size() - 1, 0, output_bo_size};
  arg_map.push_back(inmap);
  inmap = {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
           super_kernel_size};
  arg_map.push_back(inmap);
  inmap = {OpArgMap::OpArgType::CTRL_PKT_BIN, 4, 0, 0, ctrl_pkt_size};
  arg_map.push_back(inmap);

  return arg_map;
}

template class concat<uint16_t, uint16_t>;

} // namespace ryzenai
