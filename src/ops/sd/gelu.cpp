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
#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include <txn_container.hpp>
#include <utils/dpu_mdata.hpp>
#include <utils/instruction_registry.hpp>
#include <xrt_context/xrt_context.hpp>

#include "ops/ops_common/gelu_lut_bf16_512.h"
#include "ops/ops_common/matmul_matrix.hpp"
// #include "ops/ops_common/silu_lut_bf16_512.h"
#include <ops/op_interface.hpp>
#include <ops/sd/gelu.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include <xaiengine.h>
using namespace matmul_matrix;
namespace ryzenai {
// stable diffusion 1.5
namespace sd {

template <typename InT, typename WtT, typename OutT>
void gelu<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  DD_ASSERT(default_shapes_.find(txn_fname_prefix_) != default_shapes_.end(),
            OpsFusion::dd_format("txn_fname_prefix_ {} not found",
                                 txn_fname_prefix_));
  auto supported_shapes = default_shapes_[txn_fname_prefix_];
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto [b, m, n] = supported_shapes.at(i);
    auto key = get_instr_key(txn_fname_prefix_, b, m, n);
    instructions.push_back(std::make_pair(key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
std::string gelu<InT, WtT, OutT>::get_instr_key(std::string prefix, size_t b,
                                                size_t m, size_t n) const {
  return prefix + "_" + std::to_string(b) + "_" + std::to_string(m) + "_" +
         std::to_string(n);
}

template <typename InT, typename WtT, typename OutT>
gelu<InT, WtT, OutT>::gelu(const std::string &a_dtype,
                           const std::string &b_dtype,
                           const std::string &c_dtype, bool load_xrt,
                           const std::map<std::string, std::any> &attr) {
  txnbin_a_header = {{"bfloat16", "a16bf"}, {"uint16", "a16"}};
  txnbin_acc_header = {{"bfloat16", "acc16bf"}, {"uint16", "acc16"}};
  XCLBIN_FNAME_ =
      OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\SDGelu.xclbin";
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME_));
  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  const auto &input_shape_vector =
      std::any_cast<const std::vector<int> &>(attr.at("input_shape"));
  B_ = input_shape_vector[0];
  M_ = input_shape_vector[1];
  N_ = input_shape_vector[2];
  txn_fname_prefix_ = sd_gelu_key_ + txnbin_a_header.at(a_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
  default_shapes_[txn_fname_prefix_] = std::vector<std::tuple<int, int, int>>();
  // sd1.5
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 4096, 1280));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 1024, 2560));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 256, 5120));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 64, 5120));
  // sd3.0
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 154, 6144));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 160, 6144));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 1024, 6144));
  default_shapes_[txn_fname_prefix_].push_back(std::make_tuple(2, 4096, 6144));

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);
  gelu_id_ = gelu_count++;
  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
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

  std::call_once(logger_flag_, []() {
    std::string header = "gelu_id M K N kernel_m kernel_k kernel_n Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[Gelu] ID: " + std::to_string(gelu_id_) + ", XCLBIN: " +
                    XCLBIN_FNAME_ + ", (a_dtype, b_dtype, c_dtype): (" +
                    a_dtype + ", " + b_dtype + ", " + c_dtype + ")");
}

template <typename InT, typename WtT, typename OutT>
void gelu<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Gelu initialize_const_params(ptr) ...");
  io.write(0, const_params.at(0).data, b_bo_size_);
  RYZENAI_LOG_TRACE("Gelu initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void gelu<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  const size_t A_BO_SIZE = (B_ * M_ * N_ * a_dtype_size_);
  const size_t C_BO_SIZE = (B_ * M_ * N_ * c_dtype_size_);
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), b_bo_size_, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel(pdi_name_).group_id(0));
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel(pdi_name_).group_id(0));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel(pdi_name_).group_id(0));
  WtT *b_bo_map = b_bo_.map<WtT *>();
  auto bo_const = BoConst(b_bo_map);
  initialize_const_params(bo_const, const_params);
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

template <typename InT, typename WtT, typename OutT>
void gelu<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                   std::vector<Tensor> &output) {
  a_bo_.write(input.at(0).data);
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto instr_bo_key = get_instr_key(txn_fname_prefix_, B_, M_, N_);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  size_t instr_bo_words = instr_bo.size() / sizeof(int);

  // launch the kernel
  auto kernel_ = xrt_ctx_->get_kernel(pdi_name_);

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, a_bo_, b_bo_, c_bo_,
                                            0, 0, true, false);
  // sync output activation to host memory
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  c_bo_.read(output.at(0).data);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> gelu<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  std::string txn_key = get_instr_key(txn_fname_prefix_, B_, M_, N_);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
void gelu<InT, WtT, OutT>::set_params(const std::string &xclbin,
                                      const std::string &pdi_name) {
  if (!xclbin.empty()) {
    XCLBIN_FNAME_ = OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\" + xclbin;
  }
  pdi_name_ = pdi_name;
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> gelu<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  size_t input_bo_size = (B_ * M_ * N_ * sizeof(InT));
  size_t output_bo_size = (B_ * M_ * N_ * sizeof(OutT));
  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, b_bo_size_},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, output_bo_size}};
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
std::once_flag gelu<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t gelu<InT, WtT, OutT>::gelu_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag gelu<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
void gelu<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template class gelu<uint16_t, uint16_t, uint16_t>;
} // namespace sd
} // namespace ryzenai
