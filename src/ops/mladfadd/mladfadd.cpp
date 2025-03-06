// Copyright (c) 2025 Advanced Micro Devices, Inc
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
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
#include <xclbin_container.hpp>
#include <xrt_context/xrt_context.hpp>

#include <ops/mladfadd/mladfadd.hpp>
#include <ops/op_interface.hpp>
#include <txn_helper/txn_tiling_util.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include "xaiengine.h"

#include "../ops_common/matmul_matrix.hpp"

using namespace matmul_matrix;

namespace ryzenai {

namespace {
std::string getXCLBinName(std::string op_version) {
  if (op_version == "v1") {
    return LLAMA2_MLADF_2x4x4_V1_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_NAME;
  } else if (op_version == "v2") {
    return LLAMA2_MLADF_2x4x4_V2_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_NAME;
  } else if (op_version == "flat") {
    return LLAMA2_MLADF_2x4x4_BFP16_GEMM_SILU_MUL_FLAT_RMS_XCLBIN_NAME;
  } else {
    return LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_NAME;
  }
}

// Function to calculate the maximum pairwise product
auto max_pairwise_product(const std::tuple<int, int> &t) {
  return std::get<0>(t) * std::get<1>(t);
}

// Function to calculate the maximum pairwise product of a vector of tuples
int max_pairwise_product(
    const std::vector<std::tuple<int, int>> &supportedMatrixShapes) {
  auto max_product_iter =
      max_element(supportedMatrixShapes.begin(), supportedMatrixShapes.end(),
                  [](const auto &t1, const auto &t2) {
                    return max_pairwise_product(t1) < max_pairwise_product(t2);
                  });
  return max_pairwise_product(*max_product_iter);
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
      std::runtime_error("Only batch size of 1 supported for mladf add");
    }
    M = inputs.at(0).shape.at(1);
    K = inputs.at(0).shape.at(2);
  }
  return std::make_tuple(M, K);
}

template <typename LhsT, typename RhsT, typename OutT>
void mladf_add<LhsT, RhsT, OutT>::setup_supported_shapes() {
  Transaction &txn = Transaction::getInstance();
  constexpr int shape_M_idx = 4;
  constexpr int shape_K_idx = shape_M_idx + 1;
  const std::vector<std::string> &txn_file_names =
      txn.match_prefix("mladfadd_" + txn_fname_prefix_);
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

template <typename LhsT, typename RhsT, typename OutT>
std::once_flag mladf_add<LhsT, RhsT, OutT>::logger_flag_;

template <typename LhsT, typename RhsT, typename OutT>
uint64_t mladf_add<LhsT, RhsT, OutT>::mladf_add_count = 0;

template <typename LhsT, typename RhsT, typename OutT>
std::once_flag mladf_add<LhsT, RhsT, OutT>::instr_reg_flag_;

template <typename LhsT, typename RhsT, typename OutT>
std::once_flag mladf_add<LhsT, RhsT, OutT>::instr_reg_v1_flag_;

template <typename LhsT, typename RhsT, typename OutT>
void mladf_add<LhsT, RhsT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename LhsT, typename RhsT, typename OutT>
std::string mladf_add<LhsT, RhsT, OutT>::get_instr_key(std::string prefix,
                                                       size_t m, size_t k,
                                                       int64_t shape) {
  if (shape > 0) {
    return "mladfadd_" + prefix + "_" + std::to_string(shape);
  }
  return "mladfadd_" + prefix + "_" + std::to_string(m) + "_" +
         std::to_string(k);
}

template <typename LhsT, typename RhsT, typename OutT>
void mladf_add<LhsT, RhsT, OutT>::setup_instr_init() {}

template <typename LhsT, typename RhsT, typename OutT>
void mladf_add<LhsT, RhsT, OutT>::setup_instr_registry(
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

template <typename LhsT, typename RhsT, typename OutT>
mladf_add<LhsT, RhsT, OutT>::mladf_add(
    const std::string &operand_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {
  if (operand_dtype != "bfloat16") {
    std::runtime_error("Add only supportes homogeneous bfloat16 data type "
                       "for all operands and result");
  }
  operand_dtype_ = operand_dtype;
  operand_dtype_size_ = sizeof(LhsT);

  txnbin_operand_header = {{"bfloat16", "a16"}};

  op_version_ = "v1";
  if (attr.find("op_version") != attr.end()) {
    op_version_ = std::any_cast<std::string>(attr.find("op_version")->second);
    if (op_version_ != "v1" && op_version_ != "v2" && op_version_ != "flat") {
      throw std::runtime_error("The selected op version does not exist");
    }
  }

  txn_fname_prefix_ = "mladfadd_" + op_version_ + "_" +
                      txnbin_operand_header.at(operand_dtype_) +
                      txnbin_operand_header.at(operand_dtype_);

  setup_supported_shapes();
  std::sort(supported_shapes_.begin(), supported_shapes_.end(),
            [](const std::tuple<int, int> &a, const std::tuple<int, int> &b) {
              return std::get<0>(a) * std::get<1>(a) <
                     std::get<0>(b) * std::get<1>(b);
            });
  mladf_add_id_ = mladf_add_count++;

  tiled_shape_.clear();
  /* construct cost function */
  for (auto shape : supported_shapes_) {
    tiling_cost_.insert({{std::get<0>(shape) * std::get<1>(shape),
                          std::get<0>(shape) * std::get<1>(shape) * 1.0f}});
  }
  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = getXCLBinName(op_version_);

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

    // preempting bo creation with largest shape
    std::vector<std::vector<int>> shape_vector;
    for (const auto &entry : supported_shapes_) {
      shape_vector.push_back(utils::tuple_to_vector(entry));
    }
    auto operand_num_elements =
        utils::max_element_count_with_skips(shape_vector);

    if (attr.find("max_m") != attr.end() && attr.find("max_k") != attr.end()) {
      operand_num_elements = std::any_cast<int>(attr.at("max_m")) *
                             std::any_cast<int>(attr.at("max_k"));
    }

    kernel_max_size_ = ryzenai::utils::to_next_multiple(
        operand_num_elements, (int)bo_element_granularity);
    skip_create_output_ = true;
    auto itr = attr.find("skip_create_input");
    skipinput_v_ = (itr == attr.end() ? 0 : std::any_cast<int>(itr->second));
    if (skipinput_v_ != 1) { // 0 or 2: need create a_bo
      a_bo_ =
          xrt::bo(xrt_ctx_->get_device(), kernel_max_size_ * sizeof(LhsT),
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
      if (skipinput_v_ == 0) { // need create b_bo
        b_bo_ =
            xrt::bo(xrt_ctx_->get_device(), kernel_max_size_ * sizeof(RhsT),
                    XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
      }
    }
    if (attr.find("skip_create_output") == attr.end()) {
      skip_create_output_ = false;
      c_bo_ =
          xrt::bo(xrt_ctx_->get_device(), kernel_max_size_ * sizeof(OutT),
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
    std::string header = "mladf_add_id M K Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "B_copy_time(ns) B_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[MUL] ID: " + std::to_string(mladf_add_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (operand_dtype, b_dtype, c_dtype): (" + operand_dtype_ +
                    ", " + operand_dtype_ + ", " + operand_dtype_ + ")");
}
template <typename LhsT, typename RhsT, typename OutT>
bool mladf_add<LhsT, RhsT, OutT>::create_bo(void *usr_ptr, size_t size,
                                            int operand_index) {
  std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(usr_ptr);
  constexpr std::uint32_t MASK = ((1 << 12) - 1);
  if ((addr & MASK) != 0) {
    return false;
  }
  auto bo =
      xrt::bo(xrt_ctx_->get_context(), usr_ptr, size, xrt::bo::flags::host_only,
              xrt_ctx_->get_kernel().group_id(0));
  if (operand_index == 0) {
    a_bo_ = bo;
  } else if (operand_index == 1) {
    b_bo_ = bo;
  } else if (operand_index == 2) {
    c_bo_ = bo;
  } else {
    return false;
  }
  return true;
}
template <typename LhsT, typename RhsT, typename OutT>
void mladf_add<LhsT, RhsT, OutT>::execute(std::vector<Tensor> &input,
                                          std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 2) {
    throw std::runtime_error("mladfadd IPU Wrapper expect to have two inputs.");
  }
  set_kernel_shape(input.at(0).shape);
  const int a_idx = 0;
  // The first data is a and second data is b
  LhsT *a = (LhsT *)input.at(a_idx).data;
  RhsT *b = (RhsT *)input.at(a_idx + 1).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;

  auto exec_start = GET_ELAPSED_TIME_NS();

  const auto operand_size_in_bytes =
      input.at(0).shape.at(0) * input.at(0).shape.at(1) * sizeof(LhsT);
  RYZENAI_LOG_TRACE("mladfadd: operand_size_in_bytes:" +
                    std::to_string(operand_size_in_bytes));

  // TODO this trace looks repetitive w.r.t. the previous
  /* Create input/output BOs */
  RYZENAI_LOG_TRACE(
      "mladfadd: A_BO_SIZE:" + std::to_string(operand_size_in_bytes) +
      " B_BO_SIZE:" + std::to_string(operand_size_in_bytes) +
      " C_BO_SIZE:" + std::to_string(operand_size_in_bytes));

  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  LhsT *a_bo_map = a_bo_.map<LhsT *>();
  memcpy((void *)a_bo_map, (void *)a, operand_size_in_bytes);
  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  // b_bo copy
  auto b_copy_start = GET_ELAPSED_TIME_NS();
  RhsT *b_bo_map = b_bo_.map<RhsT *>();
  memcpy((void *)b_bo_map, (void *)b, operand_size_in_bytes);
  auto b_copy_stop = GET_ELAPSED_TIME_NS();

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
  memcpy((void *)aie_out, (void *)c_bo_map, operand_size_in_bytes);
  auto c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = static_cast<int64_t>(c_copy_stop - c_copy_start);
  auto exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(mladf_add_id_) + " " + std::to_string(kernel_x_shape_[0]) +
      " " + std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(b_copy_time_) + " " + std::to_string(b_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename LhsT, typename RhsT, typename OutT>
std::vector<xrt::bo> mladf_add<LhsT, RhsT, OutT>::get_inputs() {
  return {a_bo_, b_bo_};
}

template <typename LhsT, typename RhsT, typename OutT>
std::vector<xrt::bo> mladf_add<LhsT, RhsT, OutT>::get_outputs() {
  return {c_bo_};
}

template <typename LhsT, typename RhsT, typename OutT>
void mladf_add<LhsT, RhsT, OutT>::set_kernel_shape(
    const std::vector<size_t> &shape) {
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
            {OpArgMap::OpArgType::INPUT, 0, 0, 0, sizeof(LhsT)},
            {OpArgMap::OpArgType::INPUT, 1, 1, 0, sizeof(RhsT)},
            {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, sizeof(OutT)},
        };
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
  if (padded_shape > kernel_max_size_) {
    RYZENAI_LOG_TRACE("BO size too small, alloacting dynamically");
    kernel_max_size_ = padded_shape;
    if (skipinput_v_ != 1) {
      a_bo_ =
          xrt::bo(xrt_ctx_->get_device(), kernel_max_size_ * sizeof(LhsT),
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
      if (skipinput_v_ == 0) {
        b_bo_ =
            xrt::bo(xrt_ctx_->get_device(), kernel_max_size_ * sizeof(RhsT),
                    XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
      }
    }
    if (!skip_create_output_) {
      c_bo_ =
          xrt::bo(xrt_ctx_->get_device(), kernel_max_size_ * sizeof(OutT),
                  XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(0));
    }
  }
  if (tiled_shape_.size() > 1) {
    instr_bo_key_ = get_instr_key(txn_fname_prefix_, tiled_shape_.at(0),
                                  tiled_shape_.at(1));
  } else {
    instr_bo_key_ = get_instr_key(txn_fname_prefix_, 0, 0, tiled_shape_.at(0));
  }
  return;
}

template <typename LhsT, typename RhsT, typename OutT>
void mladf_add<LhsT, RhsT, OutT>::execute(std::vector<xrt::bo> &input,
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

template <typename LhsT, typename RhsT, typename OutT>
void mladf_add<LhsT, RhsT, OutT>::execute(std::vector<uint64_t> &input,
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

template <typename LhsT, typename RhsT, typename OutT>
const std::vector<uint8_t> mladf_add<LhsT, RhsT, OutT>::get_transaction_bin(
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
        {OpArgMap::OpArgType::INPUT, 0, 0, 0, sizeof(LhsT)},
        {OpArgMap::OpArgType::INPUT, 1, 1, 0, sizeof(RhsT)},
        {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, sizeof(OutT)},
    };
    data = get_tiled_fused_txnbin(tiling_spec, &get_instr_key,
                                  txn_fname_prefix_, arg_map);
  } else {
    Transaction &txn = Transaction::getInstance();
    std::vector<int64_t> tiled_shape = tiling_spec.info_.front().second;
    std::string txn_key =
        get_instr_key(txn_fname_prefix_, tiled_shape.at(0), tiled_shape.at(1));
    data = txn.get_txn_bvec(txn_key);
  }
  return data;
}

template <typename InT, typename RhsT, typename OutT>
const std::vector<uint8_t> mladf_add<InT, RhsT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return {};
}

template <typename LhsT, typename RhsT, typename OutT>
std::vector<OpArgMap> mladf_add<LhsT, RhsT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto M1 = input.at(0).shape.at(1); // [1xMxN : 1x512x768]
  auto N1 = input.at(0).shape.at(2);
  auto M2 = input.at(1).shape.at(1); // [1xMxN : 1x512x768]
  auto N2 = input.at(1).shape.at(2);
  auto M3 = output.at(0).shape.at(1); // [1xMxN : 1x512x768]
  auto N3 = output.at(0).shape.at(2);

  if ((M1 != M2) || (N1 != N2) || (N3 != N2) || (M3 != M1)) {
    throw std::runtime_error(
        "Dimensions of all tensors should be equal for mladf add op\n");
  }
  op_tiling_spec tiling_spec =
      map_padded_shape(M1, N1, supported_shapes_, tiling_cost_);
  auto operand_num_elements = tiling_spec.size_;
  operand_num_elements = ryzenai::utils::to_next_multiple(
      (int)operand_num_elements, (int)bo_element_granularity);

  size_t input_1_bo_size = (operand_num_elements * sizeof(LhsT));
  size_t input_2_bo_size = (operand_num_elements * sizeof(RhsT));
  size_t output_bo_size = (operand_num_elements * sizeof(OutT));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::INPUT, 1, 1, 0, input_2_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, output_bo_size},
  };
  return arg_map;
}

template <typename LhsT, typename RhsT, typename OutT>
std::mutex mladf_add<LhsT, RhsT, OutT>::instr_reg_mutex_;

template class mladf_add<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
