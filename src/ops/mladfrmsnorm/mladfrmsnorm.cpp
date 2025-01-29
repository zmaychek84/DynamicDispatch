/*
 Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 Licensed under the MIT License.
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
#include <xclbin_container.hpp>
#include <xrt_context/xrt_context.hpp>

#include <ops/mladfrmsnorm/mladfrmsnorm.hpp>
#include <ops/op_interface.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

#include <txn_helper/txn_tiling_util.hpp>

namespace ryzenai {

namespace {
std::string getXCLBinName(std::string op_version) {
  if (op_version == "v1") {
    return LLAMA2_MLADF_2x4x4_V1_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_NAME;
  } else if (op_version == "flat") {
    return LLAMA2_MLADF_2x4x4_BFP16_GEMM_SILU_MUL_FLAT_RMS_XCLBIN_NAME;
  } else {
    return LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_NAME;
  }
}

std::tuple<size_t, size_t> extract_MK(const Tensor &input) {
  size_t M = 0;
  size_t K = 0;
  if (input.shape.size() == 2) {
    M = input.shape.at(0);
    K = input.shape.at(1);
  } else if (input.shape.size() == 3) {
    M = input.shape.at(1);
    K = input.shape.at(2);
  } else {
    throw std::runtime_error(
        "rmsnorm expects a rank 2 [Rows,Cols] or 3 [1,Rows,Cols] tensor");
  }
  return std::make_tuple(M, K);
}
} // namespace

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
  if (attr.find("shapes") != attr.end()) {
    supported_shapes_.clear();
    auto shapes = std::any_cast<std::vector<std::vector<int>>>(
        attr.find("shapes")->second);
    for (auto sh : shapes) {
      supported_shapes_.push_back(std::tuple<int, int>{sh[0], sh[1]});
    }
  }
}

template <typename LhsT, typename WtsT, typename OutT>
void rms_norm<LhsT, WtsT, OutT>::setup_supported_shapes() {
  Transaction &txn = Transaction::getInstance();
  constexpr int shape_M_idx = 4;
  constexpr int shape_K_idx = shape_M_idx + 1;
  // auto start = std::chrono::steady_clock::now();
  const std::vector<std::string> &txn_file_names =
      txn.match_prefix("mladfrmsnorm_" + txn_fname_prefix_);

  for (const std::string &filename : txn_file_names) {
    std::stringstream filename_ss(filename);
    std::string token;
    uint8_t i = 0;
    int M;
    int K;
    mladf_matrix_shapes mat_shape;
    while (std::getline(filename_ss, token, '_')) {
      if (i >= shape_M_idx) {
        std::stringstream token_stream(token);
        if (i == shape_M_idx) {
          token_stream >> M;
        } else if (i == shape_K_idx) {
          token_stream >> K;
        }
      }
      i++;
    }
    // RYZENAI_LOG_TRACE("Supported shape: M: " + std::to_string(M) +
    //                   ", K: " + std::to_string(K));
    supported_shapes_.push_back(std::make_tuple(M, K));
  }
}

template <typename LhsT, typename WtsT, typename OutT>
std::tuple<std::tuple<int, int>, std::vector<int64_t>, double>
rms_norm<LhsT, WtsT, OutT>::map_padded_shape(int64_t M, int64_t K) const {
  RYZENAI_LOG_TRACE("Map padded shape");
  std::set<int64_t> tile_m;
  for (const auto &supported : supported_shapes_) {
    int64_t mat_M = std::get<0>(supported);
    int64_t mat_K = std::get<1>(supported);
    if (mat_M == M && mat_K == K) {
      std::vector<int64_t> tiling_m = {mat_M};
      if (m_tiling_cost_.find(mat_M) == m_tiling_cost_.end()) {
        return std::make_tuple(std::make_tuple((int)M, (int)K), tiling_m, 100);
      }
      return std::make_tuple(std::make_tuple((int)M, (int)K), tiling_m,
                             m_tiling_cost_.at(mat_M));
    }
    if (mat_K == K) {
      tile_m.insert(mat_M);
    }
  }
  // for efficiency purposes, remove M=1 for tiling for now
  if (tile_m.size() > 1 && tile_m.find(1) != tile_m.end()) {
    tile_m.erase(1);
  }
  std::pair<double, std::vector<int64_t>> tiling_info_m =
      minimum_tiles(tile_m, m_tiling_cost_, M);
  int m = (int)std::reduce(tiling_info_m.second.begin(),
                           tiling_info_m.second.end());

  RYZENAI_LOG_TRACE("Tiling M shape: " + std::to_string(m));
  return std::make_tuple(std::make_tuple(m, (int)K), tiling_info_m.second,
                         tiling_info_m.first);
}

template <typename LhsT, typename WtsT, typename OutT>
void rms_norm<LhsT, WtsT, OutT>::set_kernel_shape(
    const std::vector<size_t> &shape) {

  // kernel_x_shape_[1] = 4096;
  kernel_x_shape_[1] = shape.at(1);

  auto [tiling_info, tiling_info_m, cost] =
      map_padded_shape(shape.at(0), kernel_x_shape_[1]);

  kernel_x_shape_[0] = std::get<0>(tiling_info);
  if (load_xrt_) {
    instr_bo_key_ = get_instr_key(txn_fname_prefix_, kernel_x_shape_[0],
                                  kernel_x_shape_[1]);
  }
  if (tiling_info_m.size() > 1) {
    {
      std::lock_guard lock(instr_reg_mutex_);
      if (!xrt_ctx_->get_registry().instr_in_registry(instr_bo_key_)) {
        RYZENAI_LOG_TRACE(
            "Tiling required, generating tiled transaction binary ...");
        auto txn_bin_vec = generate_fused_txnbin(tiling_info, tiling_info_m,
                                                 kernel_x_shape_[1]);
        auto instr = std::make_pair(instr_bo_key_, true);
        xrt_ctx_->get_registry().insert_fused_instr_to_instruction_map(
            instr, txn_bin_vec);
      }
    }
  }
  if ((size_t)kernel_x_shape_[0] * kernel_x_shape_[1] > kernel_max_size_) {
    kernel_max_size_ = kernel_x_shape_[0] * kernel_x_shape_[1];
    if (!skip_create_input_a_) {
      RYZENAI_LOG_TRACE("BO size too small, alloacting dynamically");
      a_bo_ =
          xrt::bo(xrt_ctx_->get_device(), kernel_max_size_ * sizeof(LhsT),
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    }
    if (!skip_create_output_) {
      RYZENAI_LOG_TRACE("BO size too small, alloacting dynamically");
      c_bo_ =
          xrt::bo(xrt_ctx_->get_device(), kernel_max_size_ * sizeof(OutT),
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    }
    RYZENAI_LOG_TRACE("Allocated BOs of size: " +
                      std::to_string(kernel_max_size_));
  }
}
template <typename LhsT, typename WtsT, typename OutT>
void rms_norm<LhsT, WtsT, OutT>::create_bo(void *data_b, size_t size,
                                           int index) {
  if (index == 1) {
    c_bo_ =
        xrt::bo(xrt_ctx_->get_context(), data_b, size,
                xrt::bo::flags::host_only, xrt_ctx_->get_kernel().group_id(0));
  } else {
    a_bo_ =
        xrt::bo(xrt_ctx_->get_context(), data_b, size,
                xrt::bo::flags::host_only, xrt_ctx_->get_kernel().group_id(0));
  }
  // b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
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
    if (op_version_ != "v1" && op_version_ != "flat") {
      throw std::runtime_error("The selected op version does not exist");
    }
  }

  txn_fname_prefix_ =
      "rmsnorm_" + op_version_ + "_" + txnbin_operand_header.at(operand_dtype_);
  setup_supported_shapes();
  m_tiling_cost_ = {{1, 0.24},   {128, 1.0},  {256, 1.98},
                    {512, 3.94}, {1024, 7.8}, {2048, 15.5}};
  rms_norm_id_ = rms_norm_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = getXCLBinName(op_version_);

  // preempting bo creation with largest shape
  std::vector<std::vector<int>> shape_vector;
  for (const auto &entry : supported_shapes_) {
    shape_vector.push_back(utils::tuple_to_vector(entry));
  }
  kernel_max_size_ = utils::max_element_count_with_skips(shape_vector);
  const auto operand_size_in_bytes = kernel_max_size_ * operand_dtype_size_;
  const auto wts_size_in_bytes =
      utils::max_element_count_with_skips(shape_vector, {0}) *
      operand_dtype_size_;

  skip_create_output_ = true;
  if (load_xrt) {

    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(
        XCLBIN_FNAME, 0, {},
        XclbinContainer::getInstance().get_xclbin_content(XCLBIN_FNAME));

    if (op_version_ == "v1") {
      std::call_once(instr_reg_v1_flag_,
                     [this, &attr]() { setup_instr_init(); });
    } else {
      std::call_once(instr_reg_flag_, [this, &attr]() { setup_instr_init(); });
    }
    setup_instr_registry(attr);

    skip_create_input_a_ = attr.find("skip_create_input_a") != attr.end() ||
                           attr.find("skip_create_input") != attr.end();

    skip_create_input_b_ = attr.find("skip_create_input_b") != attr.end() ||
                           attr.find("skip_create_input") != attr.end();

    if (!skip_create_input_a_) {
      a_bo_ =
          xrt::bo(xrt_ctx_->get_device(), operand_size_in_bytes,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    }

    if (!skip_create_input_b_) {
      b_bo_ =
          xrt::bo(xrt_ctx_->get_device(), wts_size_in_bytes,
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    }
    if (attr.find("skip_create_output") == attr.end()) {
      skip_create_output_ = false;
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

  // prepare inst_bo and param_bo
  set_kernel_shape(input.at(0).shape);

  // a_bo copy
  operand_size_in_bytes_ =
      utils::running_product_with_skips(input.at(0).shape) *
      operand_dtype_size_;
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  LhsT *a_bo_map = a_bo_.map<LhsT *>();
  memcpy((void *)a_bo_map, (void *)a, operand_size_in_bytes_);
  auto a_copy_stop = GET_ELAPSED_TIME_NS();
  RYZENAI_LOG_TRACE("Copied a to BO");

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
  RYZENAI_LOG_TRACE("Copied wts to BO");

  // b_bo sync
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();

  b_copy_time_ = static_cast<int64_t>(b_copy_stop - b_copy_start);
  b_sync_time_ = static_cast<int64_t>(b_sync_stop - b_sync_start);

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

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, input[0], input[1],
                                            output[0], 0, 0, wait, false);
  return;
}

template <typename LhsT, typename WtsT, typename OutT>
void rms_norm<LhsT, WtsT, OutT>::execute(std::vector<xrt::bo> &input,
                                         std::vector<xrt::bo> &output,
                                         bool wait) {
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key_);
  auto instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, input[0], input[1],
                                            output[0], 0, 0, wait, false);
  return;
}

template <typename LhsT, typename WtsT, typename OutT>
const std::vector<uint8_t> rms_norm<LhsT, WtsT, OutT>::generate_fused_txnbin(
    const std::tuple<int, int> &tiling_info,
    const std::vector<int64_t> &tiling_info_m, const int64_t &K) const {
  RYZENAI_LOG_TRACE("Generating tiling for tiling, num m tiles: " +
                    std::to_string(tiling_info_m.size()));
  std::vector<uint8_t> data;
  Transaction &txn = Transaction::getInstance();
  // get base transactions
  std::vector<std::vector<uint8_t>> base_transactions;
  base_transactions.reserve(tiling_info_m.size());
  std::vector<OpArgMap> source_arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, 0},
      {OpArgMap::OpArgType::INPUT, 1, 1, 0, 0},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, 0},
  };
  std::vector<std::vector<OpArgMap>> dest_arg_maps;
  dest_arg_maps.reserve(tiling_info_m.size());

  size_t accumulated_input_size = 0;
  size_t accumulated_output_size = 0;
  for (const auto &m : tiling_info_m) {
    std::string txn_key = get_instr_key(txn_fname_prefix_, m, K);
    base_transactions.emplace_back(txn.get_txn_bvec(txn_key));
    std::vector<OpArgMap> dest_arg_map{
        {OpArgMap::OpArgType::INPUT, 0, 0, accumulated_input_size, 0},
        {OpArgMap::OpArgType::INPUT, 1, 1, 0, 0},
        {OpArgMap::OpArgType::OUTPUT, 2, 2, accumulated_output_size, 0},
    };
    size_t num_elem_operand = m * K;
    dest_arg_maps.emplace_back(dest_arg_map);
    accumulated_output_size += num_elem_operand * sizeof(OutT);
    accumulated_input_size += num_elem_operand * sizeof(LhsT);
  }
  RYZENAI_LOG_TRACE("base_transactions size: " +
                    std::to_string(base_transactions.size()));
  RYZENAI_LOG_TRACE("dest_arg_maps size: " +
                    std::to_string(dest_arg_maps.size()));
  data = rmsnorm_nonuniform_tile_transaction_bin(base_transactions,
                                                 source_arg_map, dest_arg_maps);
  return data;
}

template <typename LhsT, typename WtsT, typename OutT>
const std::vector<uint8_t> rms_norm<LhsT, WtsT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  const auto [M, K] = extract_MK(input.at(0));
  std::vector<uint8_t> data;
  auto [tiling_info, tiling_info_m, cost] = map_padded_shape(M, K);
  if (tiling_info_m.size() > 1) {
    data = generate_fused_txnbin(tiling_info, tiling_info_m, K);
  } else {
    Transaction &txn = Transaction::getInstance();
    std::string txn_key = get_instr_key(txn_fname_prefix_, M, K);
    data = txn.get_txn_bvec(txn_key);
  }
  return data;
}

template <typename InT, typename WtsT, typename OutT>
const std::vector<uint8_t> rms_norm<InT, WtsT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return {};
}

template <typename LhsT, typename WtsT, typename OutT>
void rms_norm<LhsT, WtsT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  int16_t *weights = (int16_t *)const_params.at(0).data; // bf16
  size_t w_shape = const_params.at(0).shape.at(0);
  io.write(0, weights, w_shape * sizeof(int16_t));
}

template <typename LhsT, typename WtsT, typename OutT>
std::vector<OpArgMap> rms_norm<LhsT, WtsT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  const auto [M, K] = extract_MK(input.at(0));
  std::tuple<size_t, size_t> shape_operand = std::make_tuple(M, K);
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
  auto [tiling_info, tiling_info_m, cost] = map_padded_shape(M, K);
  const auto num_elem_operand =
      utils::running_product_with_skips(utils::tuple_to_vector(tiling_info));
  const auto num_elem_wts = shape_wts;
  size_t input_1_bo_size = (num_elem_operand * sizeof(LhsT));
  size_t input_2_bo_size = (num_elem_wts * sizeof(WtsT));
  size_t output_bo_size = (num_elem_operand * sizeof(OutT));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, input_2_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, output_bo_size},
  };
  return arg_map;
}

template <typename LhsT, typename WtsT, typename OutT>
std::mutex rms_norm<LhsT, WtsT, OutT>::instr_reg_mutex_;

template class rms_norm<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
