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
#include <ops/flat/mha_v2.hpp>
#include <ops/op_interface.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>
#include <xclbin_container.hpp>

namespace ryzenai {
namespace flat {

template <typename InT, typename OutT>
void mha_v2<InT, OutT>::setup_instr_registry(
    const std::map<std::string, std::any> &attr) {
  std::vector<std::vector<size_t>> supported_shapes;
  std::vector<std::pair<std::string, bool>> instructions;
  if (attr.find("shapes") != attr.end()) {
    auto shapes = std::any_cast<std::vector<std::vector<int>>>(
        attr.find("shapes")->second);
    for (auto sh : shapes) {
      supported_shapes.emplace_back(
          // num_head, seq_len_q, seq_len_total_k, head_size
          std::vector<size_t>{(size_t)sh[0], (size_t)sh[1], (size_t)sh[2],
                              (size_t)sh[3]});
    }
  } else {
    supported_shapes = default_shapes_.find(txn_fname_prefix_)->second;
  }
  for (auto &sh : supported_shapes) {
    auto key = get_instr_key(txn_fname_prefix_,
                             std::vector<size_t>{(size_t)sh[0], (size_t)sh[1],
                                                 (size_t)sh[2], (size_t)sh[3]});
    instructions.push_back(std::make_pair(key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename OutT>
std::string
mha_v2<InT, OutT>::get_instr_key(std::string prefix,
                                 const std::vector<size_t> &shape) const {
  std::string out_str = prefix;
  for (size_t i = 0; i < shape.size(); i++) {
    out_str += "_" + std::to_string(shape[i]);
  }
  return out_str;
}

template <typename InT, typename OutT>
mha_v2<InT, OutT>::mha_v2(const std::string &a_dtype,
                          const std::string &c_dtype, bool load_xrt,
                          const std::map<std::string, std::any> &attr) {
  a_dtype_ = a_dtype;
  c_dtype_ = c_dtype;
  txnbin_a_header = {{"bfloat16", "a16bf"}, {"bfp16ebs8", "a16bfp"}};
  txnbin_acc_header = {{"bfloat16", "acc16bf"}};
  txn_fname_prefix_ = "flat_mha_v2_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
  // input_shape {num_heads, seq_len_q, seq_len_total_k, head_size}
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{32, 1, 1024, 96});

  a_dtype_size_ = sizeof(InT);
  c_dtype_size_ = sizeof(OutT);

  mha_v2_id_++;

  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));
    if (input_shape_vector.size() == 4) {
      num_heads_ = input_shape_vector[0];
      seq_len_q_ = input_shape_vector[1];
      seq_len_total_k_ = input_shape_vector[2];
      head_size_ = input_shape_vector[3];
    } else {
      throw std::runtime_error(
          "FLAT_mha_v2 input_shape attr should be a vector of size 4");
    }
    RYZENAI_LOG_TRACE(
        "FLAT_mha_v2: InputShape: " + std::to_string(B_) + ", " +
        std::to_string(seq_len_q_) + ", " + std::to_string(num_heads_) + ", " +
        std::to_string(head_size_) + ", " + std::to_string(seq_len_total_k_));
  } else {
    RYZENAI_LOG_TRACE("FLAT_mha_v2: InputShape not inserted.");
  }

  q_size_ = B_ * num_heads_ * seq_len_q_ * head_size_ * sizeof(InT);
  kv_size_ = B_ * num_heads_ * seq_len_total_k_ * head_size_ * sizeof(InT);
  rope_wts_size_ = B_ * seq_len_q_ * head_size_ * sizeof(InT);
  mask_size_ = seq_len_q_ * seq_len_total_k_ * sizeof(InT);

  ifm_bo_size_ = q_size_ * 2 + rope_wts_size_ + mask_size_;
  total_k_bo_size_ = kv_size_;
  total_v_bo_size_ = kv_size_;
  ofm1_size_ = B_ * num_heads_ * seq_len_total_k_ * head_size_ * sizeof(OutT);
  ofm2_size_ = B_ * num_heads_ * seq_len_q_ * head_size_ * sizeof(OutT);

  XCLBIN_FNAME_ = LLAMA2_MLADF_2x4x4_BFP16_GEMM_SILU_MUL_FLAT_RMS_XCLBIN_NAME;

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME_));
  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(
        XCLBIN_FNAME_, 0, {},
        XclbinContainer::getInstance().get_xclbin_content(XCLBIN_FNAME_));
    std::call_once(instr_reg_flag_,
                   [this, attr]() { setup_instr_registry(attr); });
  }

  std::call_once(logger_flag_, []() {
    std::string header = "ipu_wrapper_id num_heads, seq_len_q, "
                         "seq_len_total_k, head_size Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[FLAT mha_v2] ID: " + std::to_string(mha_v2_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME_ + ", (a_dtype, c_dtype): (" +
                    a_dtype + ", " + c_dtype + ")");
}

template <typename InT, typename OutT>
void mha_v2<InT, OutT>::set_params(
    const std::string &model_name,
    const std::vector<size_t>
        &input_shape, // input_shape {num_heads, seq_len_q, seq_len_total_k,
                      // head_size}
    const std::map<std::string, std::any> &attr) {
  set_kernel_shapes(input_shape);
  std::string UT_XCLBIN_FNAME =
      LLAMA2_MLADF_2x4x4_BFP16_GEMM_SILU_MUL_FLAT_RMS_XCLBIN_NAME;
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(
      UT_XCLBIN_FNAME, 0, {},
      XclbinContainer::getInstance().get_xclbin_content(UT_XCLBIN_FNAME));
  std::call_once(instr_reg_flag_,
                 [this, attr]() { setup_instr_registry(attr); });
  const int gid = 0;
  ifm_bo_ =
      xrt::bo(xrt_ctx_->get_device(), ifm_bo_size_, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(gid));
  ofm1_bo_ = xrt::bo(xrt_ctx_->get_device(), ofm1_size_, XRT_BO_FLAGS_HOST_ONLY,
                     xrt_ctx_->get_kernel().group_id(gid));
  ofm2_bo_ = xrt::bo(xrt_ctx_->get_device(), ofm2_size_, XRT_BO_FLAGS_HOST_ONLY,
                     xrt_ctx_->get_kernel().group_id(gid));

  auto kernel_ = xrt_ctx_->get_kernel();

  skip_create_total_k_ = (attr.find("skip_create_total_k") != attr.end());
  skip_create_total_v_ = (attr.find("skip_create_total_v") != attr.end());
  if (!skip_create_total_k_) {
    total_k_bo_ = xrt::bo(xrt_ctx_->get_device(), total_k_bo_size_,
                          XRT_BO_FLAGS_HOST_ONLY, kernel_.group_id(gid));
  }
  if (!skip_create_total_v_) {
    total_v_bo_ = xrt::bo(xrt_ctx_->get_device(), total_v_bo_size_,
                          XRT_BO_FLAGS_HOST_ONLY, kernel_.group_id(gid));
  }
}

template <typename InT, typename OutT>
void mha_v2<InT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {}

template <typename InT, typename OutT>
void mha_v2<InT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("flat mha_v2 initialize_const_params ...");
  // confirm group_ids
  const int gid = 0;
  auto kernel_ = xrt_ctx_->get_kernel();
  total_k_bo_ = xrt::bo(xrt_ctx_->get_device(), total_k_bo_size_,
                        XRT_BO_FLAGS_HOST_ONLY, kernel_.group_id(gid));
  total_v_bo_ = xrt::bo(xrt_ctx_->get_device(), total_v_bo_size_,
                        XRT_BO_FLAGS_HOST_ONLY, kernel_.group_id(gid));
}

template <typename InT, typename OutT>
xrt::bo mha_v2<InT, OutT>::create_bo(void *usr_ptr, size_t size,
                                     int operand_index) {
  std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(usr_ptr);
  constexpr std::uint32_t MASK = ((1 << 12) - 1);
  // DD_ASSERT((addr & MASK) == 0, "addr must be multiple of 4096 address.");
  auto bo =
      xrt::bo(xrt_ctx_->get_context(), usr_ptr, size, xrt::bo::flags::host_only,
              xrt_ctx_->get_kernel().group_id(0));
  if (operand_index == 0) {
    ifm_bo_ = bo;
  } else if (operand_index == 1) {
    total_k_bo_ = bo;
  } else if (operand_index == 2) {
    total_v_bo_ = bo;
  } else if (operand_index == 3) {
    ofm1_bo_ = bo;
  } else if (operand_index == 4) {
    ofm2_bo_ = bo;
  } else {
    throw std::runtime_error("create_bo with invalid operand_index " +
                             std::to_string(operand_index));
  }
  return bo;
}

template <typename InT, typename OutT>
void mha_v2<InT, OutT>::execute(std::vector<Tensor> &input,
                                std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("mha_v2 execute ...");
  run_aie_time_ = 0;
  DD_ASSERT(input.size() == 3,
            OpsFusion::dd_format("flat mha_v2 input tensor expects 4. Got {}",
                                 input.size()));
  DD_ASSERT(output.size() == 2,
            OpsFusion::dd_format("flat mha_v2 output tensor expects 1. Got {}",
                                 output.size()));
  // inputs
  ifm_bo_.write(input.at(0).data);
  ifm_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  total_k_bo_.write(input.at(1).data);
  total_k_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  total_v_bo_.write(input.at(2).data);
  total_v_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  // prepare inst_bo
  std::vector<size_t> param_shape = {num_heads_, seq_len_q_, seq_len_total_k_,
                                     head_size_};
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key_);
  size_t instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  xrt::run run;
  auto run_aie_start = GET_ELAPSED_TIME_NS();

  // param order to be confirmed
  run = kernel_(2, instr_bo, instr_bo_words,
                ifm_bo_.address() + DDR_AIE_ADDR_OFFSET,
                total_k_bo_.address() + DDR_AIE_ADDR_OFFSET,
                total_k_bo_.address() + DDR_AIE_ADDR_OFFSET,
                total_v_bo_.address() + DDR_AIE_ADDR_OFFSET,
                ofm2_bo_.address() + DDR_AIE_ADDR_OFFSET);
  run.wait2();
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);
  num_run_aie_++;

  ofm1_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  ofm1_bo_.read(output.at(0).data);
  ofm2_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  ofm2_bo_.read(output.at(1).data);
}

template <typename InT, typename OutT>
void mha_v2<InT, OutT>::execute(std::vector<xrt::bo> &input,
                                std::vector<xrt::bo> &output, size_t offset,
                                bool wait) {
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key_);
  auto instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();
  auto run = kernel_(2, instr_bo, instr_bo_words,
                     ifm_bo_.address() + DDR_AIE_ADDR_OFFSET,
                     total_k_bo_.address() + DDR_AIE_ADDR_OFFSET + offset,
                     total_k_bo_.address() + DDR_AIE_ADDR_OFFSET,
                     total_v_bo_.address() + DDR_AIE_ADDR_OFFSET,
                     ofm2_bo_.address() + DDR_AIE_ADDR_OFFSET);
  if (wait) {
    run.wait2();
  }
}

template <typename InT, typename OutT>
std::vector<xrt::bo> mha_v2<InT, OutT>::get_inputs() {
  return {ifm_bo_, total_k_bo_, total_v_bo_};
}

template <typename InT, typename OutT>
std::vector<xrt::bo> mha_v2<InT, OutT>::get_outputs() {
  return {ofm1_bo_, ofm2_bo_};
}

template <typename InT, typename OutT>
void mha_v2<InT, OutT>::set_execute_kernel_shape(
    const std::vector<size_t> &mha_shape) {
  auto num_heads = mha_shape[0];
  auto seq_len_q = mha_shape[1];
  auto seq_len_total_k = mha_shape[2];
  auto head_size = mha_shape[3];
  if (seq_len_q_ != seq_len_q || num_heads_ != num_heads ||
      head_size_ != head_size || seq_len_total_k_ != seq_len_total_k) {
    set_kernel_shapes(mha_shape);
  }
}

template <typename InT, typename OutT>
void mha_v2<InT, OutT>::set_kernel_shapes(
    const std::vector<size_t> &mha_shape) {
  num_heads_ = mha_shape[0];
  seq_len_q_ = mha_shape[1];
  seq_len_total_k_ = mha_shape[2];
  head_size_ = mha_shape[3];
  q_size_ = B_ * num_heads_ * seq_len_q_ * head_size_ * sizeof(InT);
  kv_size_ = B_ * num_heads_ * seq_len_total_k_ * head_size_ * sizeof(InT);
  rope_wts_size_ = B_ * seq_len_q_ * head_size_ * sizeof(InT);
  mask_size_ = seq_len_q_ * seq_len_total_k_ * sizeof(InT);

  ifm_bo_size_ = q_size_ * 2 + rope_wts_size_ + mask_size_;
  total_k_bo_size_ = kv_size_;
  total_v_bo_size_ = kv_size_;
  ofm1_size_ = B_ * num_heads_ * seq_len_total_k_ * head_size_ * sizeof(OutT);
  ofm2_size_ = B_ * num_heads_ * seq_len_q_ * head_size_ * sizeof(OutT);
  instr_bo_key_ = get_instr_key(txn_fname_prefix_, mha_shape);
}

template <typename InT, typename OutT>
void mha_v2<InT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename OutT>
const std::vector<uint8_t> mha_v2<InT, OutT>::get_transaction_bin() const {
  std::string txn_key = txn_fname_prefix_ + "_" + std::to_string(num_heads_) +
                        "_" + std::to_string(seq_len_q_) + "_" +
                        std::to_string(seq_len_total_k_) + "_" +
                        std::to_string(head_size_);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename OutT>
const std::vector<uint8_t> mha_v2<InT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return this->get_transaction_bin();
}

template <typename InT, typename OutT>
std::vector<OpArgMap> mha_v2<InT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  // TODO: where did the ofm present_v goes in kernel?
  // mask has the same size as scratch
  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, ifm_bo_size_},
      {OpArgMap::OpArgType::OUTPUT, 1, 10, 0, ofm1_size_}, // present_k
      {OpArgMap::OpArgType::INPUT, 2, 3, 0, kv_size_},     // passed_k
      {OpArgMap::OpArgType::INPUT, 3, 4, 0, kv_size_},     // passed_v
      {OpArgMap::OpArgType::OUTPUT, 4, 9, 0, ofm2_size_},  // ofm
      {OpArgMap::OpArgType::OUTPUT, 1, 11, 0,
       ofm1_size_} // present_v but not used by kerenl
  };

  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("flat mha_v2 argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename OutT>
std::once_flag mha_v2<InT, OutT>::logger_flag_;

template <typename InT, typename OutT>
uint64_t mha_v2<InT, OutT>::mha_v2_count = 0;

template <typename InT, typename OutT>
std::once_flag mha_v2<InT, OutT>::instr_reg_flag_;

template class mha_v2<std::uint16_t, std::uint16_t>;
} // namespace flat
} // namespace ryzenai
