/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */
#include <iostream>
#include <map>
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
#include <vector>
#include <xrt_context/xrt_context.hpp>

#include <ops/op_interface.hpp>
#include <ops/silu/silu.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include "xaiengine.h"

#include "../ops_common/matmul_matrix.hpp"
#include <txn_helper/txn_tiling_util.hpp>

using namespace matmul_matrix;

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

static std::tuple<size_t, size_t>
extract_MK(const std::vector<Tensor> &inputs) {
  size_t M = 0;
  size_t K = 0;
  if (inputs.at(0).shape.size() == 2) {
    M = inputs.at(0).shape.at(0);
    K = inputs.at(0).shape.at(1);
  } else if (inputs.at(0).shape.size() == 3) {
    if (inputs.at(0).shape.at(0) != 1) {
      throw std::runtime_error("Only batch size of 1 supported for silu");
    }
    M = inputs.at(0).shape.at(1);
    K = inputs.at(0).shape.at(2);
  }
  return std::make_tuple(M, K);
}

template <typename InT, typename OutT>
std::once_flag silu<InT, OutT>::logger_flag_;

template <typename InT, typename OutT> uint64_t silu<InT, OutT>::silu_count = 0;

template <typename InT, typename OutT>
std::once_flag silu<InT, OutT>::instr_reg_flag_;

template <typename InT, typename OutT>
std::once_flag silu<InT, OutT>::instr_reg_v1_flag_;

template <typename InT, typename OutT>
void silu<InT, OutT>::debug(bool enable) {
  debug_ = enable;
}
template <typename InT, typename OutT>
void silu<InT, OutT>::setup_instr_init() {}

template <typename InT, typename OutT>
std::string silu<InT, OutT>::get_instr_key(std::string prefix, size_t m,
                                           size_t k, int64_t shape) {
  // NOTE the need of that first "silu_" is weird....
  //  it seems that the first "silu_" indicates a higher level folder?
  if (shape > 0) {
    return "silu_" + prefix + "_" + std::to_string(shape);
  }
  return "silu_" + prefix + "_" + std::to_string(m) + "_" + std::to_string(k);
}

template <typename InT, typename OutT>
void silu<InT, OutT>::setup_instr_registry(
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

template <typename InT, typename OutT>
void silu<InT, OutT>::setup_supported_shapes() {
  Transaction &txn = Transaction::getInstance();
  constexpr int shape_M_idx = 4;
  constexpr int shape_K_idx = shape_M_idx + 1;
  const std::vector<std::string> &txn_file_names =
      txn.match_prefix("silu_" + txn_fname_prefix_);
  for (const std::string &filename : txn_file_names) {
    std::stringstream filename_ss(filename);
    std::string token;
    uint8_t i = 0;
    int M;
    int K;
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
    supported_shapes_.push_back(std::make_tuple(M, K));
  }
}

template <typename InT, typename OutT>
silu<InT, OutT>::silu(const std::string &operand_dtype, bool load_xrt,
                      const std::map<std::string, std::any> &attr) {
  if (operand_dtype != "bfloat16") {
    throw std::runtime_error("Silu only supports bfloat16 data type "
                             "for operand and result");
  }
  operand_dtype_ = operand_dtype;
  operand_dtype_size_ = sizeof(InT);

  txnbin_operand_header = {{"bfloat16", "a16"}};

  op_version_ = "v1";
  if (attr.find("op_version") != attr.end()) {
    op_version_ = std::any_cast<std::string>(attr.find("op_version")->second);
    if (op_version_ != "v1") {
      throw std::runtime_error("The selected op version does not exist");
    }
  }

  txn_fname_prefix_ =
      "silu_" + op_version_ + "_" + txnbin_operand_header.at(operand_dtype_);

  setup_supported_shapes();

  silu_id_ = silu_count++;

  tiled_shape_.clear();
  /* construct cost function */
  std::map<int64_t, double> m_cost = {{1, 0.24},   {128, 1.0},  {256, 1.98},
                                      {512, 3.92}, {1024, 7.8}, {2048, 15.5},
                                      {3072, 21.5}};

  for (auto shape : supported_shapes_) {
    if (m_cost.count(std::get<0>(shape))) {
      tiling_cost_.insert(
          {{std::get<0>(shape) * std::get<1>(shape),
            m_cost.at(std::get<0>(shape)) * (std::get<1>(shape) / 3072)}});
    }
  }
  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = getXCLBinName(op_version_);

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    if (op_version_ == "v1") {
      std::call_once(instr_reg_v1_flag_,
                     [this, &attr]() { setup_instr_init(); });
    } else {
      std::call_once(instr_reg_flag_, [this, &attr]() { setup_instr_init(); });
    }
    setup_instr_registry(attr);

    // preempting bo creation with largest shape
    std::vector<std::vector<int>> shape_vector;
    for (const auto &entry : supported_shapes_) {
      shape_vector.push_back(utils::tuple_to_vector(entry));
    }
    const auto operand_num_elements =
        utils::max_element_count_with_skips(shape_vector);
    kernel_max_size_ = ryzenai::utils::to_next_multiple(
        operand_num_elements, (int)bo_element_granularity);
    if (attr.find("skip_create_input") == attr.end()) {
      a_bo_ =
          xrt::bo(xrt_ctx_->get_device(), kernel_max_size_ * sizeof(InT),
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    }
    if (attr.find("skip_create_output") == attr.end()) {
      c_bo_ =
          xrt::bo(xrt_ctx_->get_device(), kernel_max_size_ * sizeof(OutT),
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    }
  }

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  num_run_aie_ = 0;

  std::call_once(logger_flag_, []() {
    std::string header = "silu_id M K kernel_m kernel_k num_aie_runs Execute"
                         "time(us) run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[SILU] ID: " + std::to_string(silu_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME + ", (a_dtype, c_dtype): (" +
                    operand_dtype + ", " + operand_dtype + ")");
}
template <typename InT, typename OutT>
void silu<InT, OutT>::execute(std::vector<xrt::bo> &inputs,
                              std::vector<xrt::bo> &outputs, bool wait) {
  std::string instr_bo_key;
  if (tiled_shape_.size() > 1) {
    instr_bo_key = get_instr_key(txn_fname_prefix_, tiled_shape_.at(0),
                                 tiled_shape_.at(1));
  } else {
    instr_bo_key = get_instr_key(txn_fname_prefix_, 0, 0, tiled_shape_.at(0));
  }
  // prepare inst_bo and param_bo
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);

  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  xrt::run run;
  // do we really need to sync before? c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run = kernel_(2, instr_bo, instr_bo_words,
                inputs[0].address() + DDR_AIE_ADDR_OFFSET,
                outputs[0].address() + DDR_AIE_ADDR_OFFSET, 0, 0, 0);
  if (wait) {
    run.wait2();
  }
}
template <typename InT, typename OutT>
void silu<InT, OutT>::set_kernel_shape(std::vector<size_t> shape) {
  op_tiling_spec tiling_spec = map_padded_shape(
      shape.at(0), shape.at(1), supported_shapes_, tiling_cost_);
  if (tiling_spec.info_.size() > 1) {
    std::string instr_bo_key =
        get_instr_key(txn_fname_prefix_, 0, 0, tiling_spec.size_);
    {
      std::lock_guard lock(instr_reg_mutex_);
      if (!xrt_ctx_->get_registry().instr_in_registry(instr_bo_key)) {
        RYZENAI_LOG_TRACE(
            "Tiling required, generating tiled transaction binary ...");
        std::vector<OpArgMap> arg_map{
            {OpArgMap::OpArgType::INPUT, 0, 0, 0, sizeof(InT)},
            {OpArgMap::OpArgType::OUTPUT, 1, 1, 0, sizeof(OutT)}};
        auto txn_bin_vec = get_tiled_fused_txnbin(tiling_spec, &get_instr_key,
                                                  txn_fname_prefix_, arg_map);
        auto instr = std::make_pair(instr_bo_key, true);
        xrt_ctx_->get_registry().insert_fused_instr_to_instruction_map(
            instr, txn_bin_vec);
        RYZENAI_LOG_TRACE("Tiling: " + std::to_string(tiling_spec.size_) +
                          ", cost: " + std::to_string(tiling_spec.cost_));
      }
    }
    tiled_shape_ = {tiling_spec.size_};
  } else {
    tiled_shape_ = tiling_spec.info_.at(0).second;
  }
  int64_t padded_shape =
      std::accumulate(tiled_shape_.begin(), tiled_shape_.end(), (int64_t)1,
                      std::multiplies<int64_t>());
  RYZENAI_LOG_TRACE("Padded shape: " + std::to_string(padded_shape));
  if (padded_shape > kernel_max_size_) {
    RYZENAI_LOG_TRACE("BO size too small, alloacting dynamically");
    kernel_max_size_ = padded_shape;
    a_bo_ = xrt::bo(xrt_ctx_->get_device(), kernel_max_size_ * sizeof(InT),
                    XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    c_bo_ = xrt::bo(xrt_ctx_->get_device(), kernel_max_size_ * sizeof(OutT),
                    XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
  }
  return;
}

template <typename InT, typename OutT>
void silu<InT, OutT>::execute(std::vector<Tensor> &input,
                              std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 1) {
    throw std::runtime_error("silu IPU Wrapper expect to have one input.");
  }
  const int a_idx = 0;
  // The first data is a and second data is b
  InT *a = (InT *)input.at(a_idx).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  num_run_aie_ = 0;

  if (tiled_shape_.size() == 0) {
    set_kernel_shape(input.at(0).shape);
  }

  auto exec_start = GET_ELAPSED_TIME_NS();

  const auto operand_size_in_bytes =
      input.at(a_idx).shape.at(0) * input.at(a_idx).shape.at(1) * sizeof(InT);
  RYZENAI_LOG_TRACE("elwmul: operand_size_in_bytes:" +
                    std::to_string(operand_size_in_bytes));

  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  memcpy((void *)a_bo_map, (void *)a, operand_size_in_bytes);
  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  auto inputs = get_inputs();
  auto outputs = get_outputs();
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
  memcpy((void *)aie_out, (void *)c_bo_map, operand_size_in_bytes);
  auto c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = static_cast<int64_t>(c_copy_stop - c_copy_start);
  auto exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(silu_id_) + " " + std::to_string(kernel_x_shape_[0]) +
      " " + std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_x_shape_[0]) + " " +
      std::to_string(kernel_x_shape_[1]) + " " + std::to_string(num_run_aie_) +
      " " + std::to_string(exec_end - exec_start) + " " +
      std::to_string(run_aie_time_) + " " + std::to_string(a_copy_time_) + " " +
      std::to_string(a_sync_time_) + " " + std::to_string(c_copy_time_) + " " +
      std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename OutT>
const std::vector<uint8_t> silu<InT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K] = extract_MK(input);
  op_tiling_spec tiling_spec =
      map_padded_shape(M, K, supported_shapes_, tiling_cost_);

  std::vector<uint8_t> data;
  if (tiling_spec.info_.size() > 1) {
    RYZENAI_LOG_TRACE("Tiling needed for shape: M: " + std::to_string(M) +
                      ", K: " + std::to_string(K));
    std::vector<OpArgMap> arg_map{
        {OpArgMap::OpArgType::INPUT, 0, 0, 0, sizeof(InT)},
        {OpArgMap::OpArgType::OUTPUT, 1, 1, 0, sizeof(OutT)}};
    data = get_tiled_fused_txnbin(tiling_spec, &get_instr_key,
                                  txn_fname_prefix_, arg_map);
  } else {
    Transaction &txn = Transaction::getInstance();
    std::string txn_key = get_instr_key(txn_fname_prefix_, M, K);
    data = txn.get_txn_bvec(txn_key);
  }
  return data;
}

template <typename InT, typename OutT>
const std::vector<uint8_t> silu<InT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return {};
}

template <typename InT, typename OutT>
std::vector<OpArgMap> silu<InT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto M1 = input.at(0).shape.at(1); // [1xMxN : 1x512x768]
  auto N1 = input.at(0).shape.at(2);
  auto M2 = output.at(1).shape.at(1); // [1xMxN : 1x512x768]
  auto N2 = output.at(1).shape.at(2);

  if ((M1 != M2) || (N1 != N2)) {
    throw std::runtime_error(
        "Dimensions of all tensors should be equal for silu op\n");
  }
  op_tiling_spec tiling_spec =
      map_padded_shape(M1, N1, supported_shapes_, tiling_cost_);
  auto operand_num_elements = tiling_spec.size_;
  operand_num_elements = ryzenai::utils::to_next_multiple(
      (int)operand_num_elements, (int)bo_element_granularity);

  size_t input_1_bo_size = (operand_num_elements * sizeof(InT));
  size_t output_bo_size = (operand_num_elements * sizeof(OutT));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 1, 1, 0, output_bo_size},
  };
  return arg_map;
}

template <typename InT, typename OutT>
std::mutex silu<InT, OutT>::instr_reg_mutex_;

template class silu<uint16_t, uint16_t>;

} // namespace ryzenai
