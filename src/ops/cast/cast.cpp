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

#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>

#ifndef _WIN32
#include <cmath>
#endif

#include <iomanip>
#include <iterator>
#include <string>

#include <ops/cast/cast.hpp>
#include <ops/op_interface.hpp>
#include <ops/ops_common/help_file.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>

#include "txn_helper/txn_helper.hpp"
#include "utils/dpu_mdata.hpp"

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

using std::vector;

namespace ryzenai {

inline int get_shape_ele_num(const std::vector<int> &shape) {
  int total_num = 1;
  for (int dim : shape) {
    total_num *= dim;
  }
  return total_num;
}

template <typename InT, typename OutT>
void cast<InT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  auto supported_shapes = default_shapes_.find(txn_fname_prefix_)->second;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto shape = supported_shapes.at(i);
    auto key = get_instr_key(txn_fname_prefix_, shape[0], shape[1]);
    instructions.push_back(std::make_pair(key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename OutT>
std::string cast<InT, OutT>::get_instr_key(std::string prefix, size_t H,
                                           size_t W) const {
  return "cast_" + prefix + "_" + std::to_string(H) + "_" + std::to_string(W);
}

// cast class constructor
template <typename InT, typename OutT>
cast<InT, OutT>::cast(const std::string &a_dtype, const std::string &c_dtype,
                      bool load_xrt,
                      const std::map<std::string, std::any> &attr)
    : attr_(attr) {

  txnbin_a_header = {{"bfloat16", "a16bf"}};
  txnbin_c_header = {{"bfp16ebs8", "acc16bfp"}};

  a_dtype_ = a_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  c_dtype_size_ = sizeof(OutT);
  cast_id_ = cast_count++;
  XCLBIN_FNAME_ = OpInterface::get_dd_base_dir() +
                  "\\xclbin\\stx\\Cast_Bf16Bfp16_2x4x4.xclbin";
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME_));
  txn_fname_prefix_ =
      "cast_" + txnbin_a_header.at(a_dtype) + txnbin_c_header.at(c_dtype);
  default_shapes_["cast_a16bfacc16bfp"].emplace_back(
      std::vector<int>{2048, 4096});
  default_shapes_["cast_a16bfacc16bfp"].emplace_back(
      std::vector<int>{2048, 11008});
  default_shapes_["cast_a16bfacc16bfp"].emplace_back(
      std::vector<int>{2048, 13696});

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  ebs_ = 8;
  if (attr.count("ebs") && attr.at("ebs").type() == typeid(int)) {
    ebs_ = std::any_cast<const int &>(attr.at("ebs"));
  }

  std::call_once(logger_flag_, []() {
    std::string header =
        "cast_id | Execute time | total time | Avg_time_per_aie_run\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[cast] ID: " + std::to_string(cast_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME_ + ", (a_dtype, c_dtype): (" +
                    a_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename OutT>
void cast<InT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("cast initialize_const_params ...");
  RYZENAI_LOG_TRACE("cast initialize_const_params ... DONE");
}

template <typename InT, typename OutT>
void cast<InT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {

  size_t A_BO_SIZE, B_BO_SIZE, C_BO_SIZE;
  A_BO_SIZE = B_BO_SIZE = C_BO_SIZE = 0;
  A_BO_SIZE = H_ * W_ * a_dtype_size_;
  B_BO_SIZE = b_bo_size_;
  /// output shape is H / ebs * W * ebs * ebs * (ebs + 1)
  /// equals to H / ebs * W * (ebs + 1)
  C_BO_SIZE = (H_ / ebs_ * W_ * (ebs_ + 1) * c_dtype_size_);

  RYZENAI_LOG_TRACE("cast: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  dummy_b_bo_ =
      xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  RYZENAI_LOG_TRACE("cast initialize_const_params ... DONE");
}

template <typename InT, typename OutT>
void cast<InT, OutT>::execute(std::vector<Tensor> &input,
                              std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("cast execute ...");
  a_bo_.write(input.at(0).data);
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  size_t H = input.at(0).shape.at(0);
  size_t W = input.at(0).shape.at(1);

  auto instr_bo_key = get_instr_key(txn_fname_prefix_, H, W);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  size_t instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();

  auto run_aie_start = GET_ELAPSED_TIME_NS();

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, a_bo_, c_bo_,
                                            dummy_b_bo_, 0, 0, true, false);
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);
  /* sync output activation to host memory */
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  c_bo_.read(output.at(0).data);

  RYZENAI_LOG_INFO(std::to_string(cast_id_) + " " +
                   std::to_string(num_run_aie_) + " " +
                   std::to_string(run_aie_time_) + " " +
                   std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename OutT>
void cast<InT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename OutT>
const std::vector<uint8_t> cast<InT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  size_t H = input.at(0).shape.at(0);
  size_t W = input.at(0).shape.at(1);
  std::string txn_key = get_instr_key(txn_fname_prefix_, H, W);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename OutT>
void cast<InT, OutT>::set_params(const std::vector<size_t> &shape,
                                 const int ebs) {
  H_ = shape.at(0);
  W_ = shape.at(1);
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename OutT>
std::vector<OpArgMap> cast<InT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  size_t H = input.at(0).shape.at(0);
  size_t W = input.at(0).shape.at(1);
  size_t a_bo_size = (H * W * a_dtype_size_);
  /// output shape is H / ebs * W * ebs * ebs * (ebs + 1)
  /// equals to H / ebs * W * (ebs + 1)
  size_t c_bo_size = (H / ebs_ * W * (ebs_ + 1) * c_dtype_size_);

  RYZENAI_LOG_TRACE("cast: A_BO_SIZE:" + std::to_string(a_bo_size) +
                    " C_BO_SIZE:" + std::to_string(c_bo_size));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, a_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 1, 1, 0, c_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 2, 0,
       b_bo_size_} // Dummy allocation
  };

  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("cast Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename OutT>
const std::vector<uint8_t> cast<InT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return {};
}

template <typename InT, typename OutT>
std::once_flag cast<InT, OutT>::logger_flag_;

template <typename InT, typename OutT> uint64_t cast<InT, OutT>::cast_count = 0;

template <typename InT, typename OutT>
std::once_flag cast<InT, OutT>::instr_reg_flag_;

template class cast<std::uint16_t, std::uint8_t>;

} // namespace ryzenai
