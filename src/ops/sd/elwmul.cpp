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

#include <ops/op_interface.hpp>
#include <ops/ops_common/help_file.hpp>
#include <ops/sd/elwmul.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>

#include "txn_helper/txn_helper.hpp"
#include "utils/dpu_mdata.hpp"

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

using std::vector;

namespace ryzenai {

namespace sd {

inline int get_shape_ele_num(const std::vector<int> &shape) {
  int total_num = 1;
  for (int dim : shape) {
    total_num *= dim;
  }
  return total_num;
}

template <typename InT, typename WtT, typename OutT>
const bool elwmul<InT, WtT, OutT>::is_bias_cal() const {
  return is_bias_cal_;
}

template <typename InT, typename WtT, typename OutT>
void elwmul<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  auto supported_shapes = default_shapes_.find(txn_fname_prefix_)->second;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto elwmul_shapes = supported_shapes.at(i);
    auto key =
        get_key(txn_fname_prefix_, elwmul_shapes.first, elwmul_shapes.second);
    instructions.push_back(std::make_pair(key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
std::string
elwmul<InT, WtT, OutT>::get_key(std::string prefix,
                                const std::vector<int> &a_shape,
                                const std::vector<int> &b_shape) const {
  auto key = prefix;
  for (int i = 0; i < a_shape.size(); i++) {
    key += "_" + std::to_string(a_shape[i]);
  }
  key += "__";
  for (int i = 0; i < b_shape.size(); i++) {
    key += "_" + std::to_string(b_shape[i]);
  }
  return key;
}

// elwmul class constructor
template <typename InT, typename WtT, typename OutT>
elwmul<InT, WtT, OutT>::elwmul(const std::string &a_dtype,
                               const std::string &b_dtype,
                               const std::string &c_dtype, bool load_xrt,
                               const std::map<std::string, std::any> &attr)
    : attr_(attr) {
  if (Utils::get_env_var("DEBUG_SD", "0") != "0") {
    this->debug_ = true;
  }
  txnbin_a_header = {{"bfloat16", "a16bf"}, {"bfp16ebs8", "a16bfp"}};
  txnbin_b_header = {{"bfloat16", "w16bf"}, {"bfp16ebs8", "w16bfp"}};
  txnbin_c_header = {{"bfloat16", "acc16bf"}};

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);
  elwmul_id_ = elwmul_count++;
  XCLBIN_FNAME_ =
      OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\SDMul.xclbin";
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME_));
  txn_fname_prefix_ = "sd_elwmul_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_c_header.at(c_dtype_);
  // default shape is pair of a_shape + b_shape, ifm shape always first
  // uet
  // layer 1
  default_shapes_[txn_fname_prefix_].emplace_back(std::make_pair(
      std::vector<int>{2, 4096, 1280}, std::vector<int>{2, 4096, 1280}));
  // layer 2
  default_shapes_[txn_fname_prefix_].emplace_back(std::make_pair(
      std::vector<int>{2, 1024, 2560}, std::vector<int>{2, 1024, 2560}));
  // layer 3
  default_shapes_[txn_fname_prefix_].emplace_back(std::make_pair(
      std::vector<int>{2, 256, 5120}, std::vector<int>{2, 256, 5120}));
  // layer 4
  default_shapes_[txn_fname_prefix_].emplace_back(std::make_pair(
      std::vector<int>{2, 64, 5120}, std::vector<int>{2, 64, 5120}));
  // SD3
  // layer 5
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::make_pair(std::vector<int>{2, 1536}, std::vector<int>{2, 1536}));
  // layer 6
  default_shapes_[txn_fname_prefix_].emplace_back(std::make_pair(
      std::vector<int>{2, 1024, 1536}, std::vector<int>{2, 1, 1536}));
  // layer 7
  default_shapes_[txn_fname_prefix_].emplace_back(std::make_pair(
      std::vector<int>{2, 154, 1536}, std::vector<int>{2, 1, 1536}));
  // layer 8
  default_shapes_[txn_fname_prefix_].emplace_back(std::make_pair(
      std::vector<int>{2, 4096, 1536}, std::vector<int>{2, 1, 1536}));
  default_shapes_[txn_fname_prefix_].emplace_back(std::make_pair(
      std::vector<int>{2, 333, 1536}, std::vector<int>{2, 1, 1536}));

  // bf16-160 shapes
  default_shapes_[txn_fname_prefix_].emplace_back(std::make_pair(
      std::vector<int>{2, 160, 1536}, std::vector<int>{2, 1, 1536}));

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_format_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  if (attr.count("a_shape") &&
      attr.at("a_shape").type() == typeid(std::vector<int>)) {
    a_shape_ = std::any_cast<const std::vector<int> &>(attr.at("a_shape"));
  } else {
    RYZENAI_LOG_INFO(
        "Input A Shape attribute not found or not of correct type.");
  }

  if (attr.count("b_shape") &&
      attr.at("b_shape").type() == typeid(std::vector<int>)) {
    b_shape_ = std::any_cast<const std::vector<int> &>(attr.at("b_shape"));
  } else {
    RYZENAI_LOG_INFO(
        "Input B Shape attribute not found or not of correct type.");
  }

  if (attr.count("c_shape") &&
      attr.at("c_shape").type() == typeid(std::vector<int>)) {
    c_shape_ = std::any_cast<const std::vector<int> &>(attr.at("c_shape"));
  } else {
    RYZENAI_LOG_INFO(
        "Output Shape attribute not found or not of correct type.");
  }

  if (a_shape_.size() == b_shape_.size()) {
    int a_size = get_shape_ele_num(a_shape_);
    int b_size = get_shape_ele_num(b_shape_);
    c_shape_ = a_size > b_size ? a_shape_ : b_shape_;
    is_Nx1x1xC_cal_ = a_size == b_size ? false : true;
  } else {
    if (a_shape_.size() > b_shape_.size()) {
      c_shape_ = a_shape_;
    } else {
      c_shape_ = b_shape_;
    }
    is_bias_cal_ = true;
  }

  std::call_once(logger_flag_, []() {
    std::string header =
        "SD_ELWMUL id | Execute time | total time | Avg_time_per_aie_run\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[SD_ELWMUL] ID: " + std::to_string(elwmul_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME_ +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype_ + ", " +
                    b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void elwmul<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("elwmul initialize_const_params ...");
  if (is_bias_cal_) {
    size_t b_bo_size = (get_shape_ele_num(b_shape_) * b_dtype_size_);
    io.write(0, const_params.at(0).data, b_bo_size);
  }
  RYZENAI_LOG_TRACE("elwmul initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void elwmul<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  std::vector<Tensor> input;
  std::vector<Tensor> output;
  size_t CONST_BO_SIZE, IFM_BO_SIZE, OFM_BO_SIZE;
  CONST_BO_SIZE, IFM_BO_SIZE, OFM_BO_SIZE = 0;
  auto args_map_list = this->get_buffer_reqs(input, output, attr);
  for (const auto &args_map : args_map_list) {
    if (args_map.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
      CONST_BO_SIZE = args_map.size;
    }
    if (args_map.arg_type == OpArgMap::OpArgType::INPUT) {
      if (args_map.xrt_arg_idx == 0) {
        IFM_BO_SIZE = args_map.size;
      } else if (args_map.xrt_arg_idx == 1) {
        CONST_BO_SIZE = args_map.size;
      } else {
        DD_ASSERT(false, "SDMul illegal OpArgMap arg_type");
      }
    }
    if (args_map.arg_type == OpArgMap::OpArgType::OUTPUT) {
      OFM_BO_SIZE = args_map.size;
    }
  }

  RYZENAI_LOG_TRACE("SD_ELWMUL: IFM_BO_SIZE:" + std::to_string(IFM_BO_SIZE) +
                    " CONST_BO_SIZE:" + std::to_string(CONST_BO_SIZE) +
                    " OFM_BO_SIZE:" + std::to_string(OFM_BO_SIZE));

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), IFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel(pdi_name_).group_id(0));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), CONST_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel(pdi_name_).group_id(0));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), OFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel(pdi_name_).group_id(0));

  if (is_bias_cal_) {
    b_copy_time_ = 0;
    b_format_time_ = 0;
    b_sync_time_ = 0;
    auto b_copy_start = GET_ELAPSED_TIME_NS();
    auto b_fomat_start = GET_ELAPSED_TIME_NS();
    WtT *b_bo_map = b_bo_.map<WtT *>();
    auto bo_const = BoConst(b_bo_map);
    initialize_const_params(bo_const, const_params, attr);
    auto b_format_stop = GET_ELAPSED_TIME_NS();
    b_format_time_ += static_cast<int64_t>(b_format_stop - b_fomat_start);
    auto b_copy_stop = GET_ELAPSED_TIME_NS();
    auto b_sync_start = GET_ELAPSED_TIME_NS();
    b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto b_sync_stop = GET_ELAPSED_TIME_NS();
    b_copy_time_ = static_cast<int64_t>(b_copy_stop - b_copy_start);
    b_sync_time_ = static_cast<int64_t>(b_sync_stop - b_sync_start);
  }
  RYZENAI_LOG_TRACE("elwmul initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void elwmul<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                     std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("elwmul execute ...");
  if (input.size() == 2) {
    is_bias_cal_ = false;
    auto a_shape = input[0].shape;
    auto b_shape = input[1].shape;
    int a_size = get_shape_ele_num(a_shape_);
    int b_size = get_shape_ele_num(b_shape_);
    is_Nx1x1xC_cal_ = a_size == b_size ? false : true;
  } else {
    is_bias_cal_ = true;
    is_Nx1x1xC_cal_ = false;
  }
  a_bo_.write(input.at(0).data);
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (!is_bias_cal_) {
    b_bo_.write(input.at(1).data);
    b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  auto instr_bo_key = get_key(txn_fname_prefix_, a_shape_, b_shape_);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  size_t instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel(pdi_name_);

  auto run_aie_start = GET_ELAPSED_TIME_NS();
  // TODO: figure out the Bo order

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, a_bo_, b_bo_, c_bo_,
                                            0, 0, true, false);
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);
  /* sync output activation to host memory */
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  c_bo_.read(output.at(0).data);

  RYZENAI_LOG_INFO(std::to_string(elwmul_id_) + " " +
                   std::to_string(num_run_aie_) + " " +
                   std::to_string(run_aie_time_) + " " +
                   std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename OutT>
void elwmul<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> elwmul<InT, WtT, OutT>::get_transaction_bin() const {
  std::string txn_key = get_key(txn_fname_prefix_, a_shape_, b_shape_);
  Transaction &txn = Transaction::getInstance();
  std::vector<uint8_t> txnData = txn.get_txn_bvec(txn_key);
  return txnData;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> elwmul<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return this->get_transaction_bin();
}

template <typename InT, typename WtT, typename OutT>
void elwmul<InT, WtT, OutT>::set_params(const std::string &xclbin,
                                        const std::string &pdi_name,
                                        const std::vector<int> &a_shape,
                                        const std::vector<int> &b_shape) {
  a_shape_ = a_shape;
  b_shape_ = b_shape;
  if (a_shape_.size() == b_shape_.size()) {
    int a_size = get_shape_ele_num(a_shape_);
    int b_size = get_shape_ele_num(b_shape_);
    c_shape_ = a_size > b_size ? a_shape_ : b_shape_;
    is_Nx1x1xC_cal_ = a_size == b_size ? false : true;
  } else {
    if (a_shape_.size() > b_shape_.size()) {
      c_shape_ = a_shape_;
    } else {
      c_shape_ = b_shape_;
    }
    is_bias_cal_ = true;
  }
  DD_ASSERT(c_shape_.size() > 0, "SD_ELWMUL illegal shape");
  if (!xclbin.empty()) {
    XCLBIN_FNAME_ = OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\" + xclbin;
  }
  pdi_name_ = pdi_name;
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> elwmul<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  size_t a_bo_size = (get_shape_ele_num(a_shape_) * a_dtype_size_);
  size_t b_bo_size = (get_shape_ele_num(b_shape_) * b_dtype_size_);
  size_t c_bo_size = (get_shape_ele_num(c_shape_) * c_dtype_size_);
  RYZENAI_LOG_TRACE("SD_ELWMUL: A_BO_SIZE:" + std::to_string(a_bo_size) +
                    " B_BO_SIZE:" + std::to_string(b_bo_size) +
                    " C_BO_SIZE:" + std::to_string(c_bo_size));
  std::vector<OpArgMap> arg_map;
  if (is_bias_cal_) {
    arg_map = {{OpArgMap::OpArgType::INPUT, 0, 0, 0, a_bo_size},
               {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, b_bo_size},
               {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, c_bo_size}};
  } else {
    arg_map = {{OpArgMap::OpArgType::INPUT, 0, 0, 0, a_bo_size},
               {OpArgMap::OpArgType::INPUT, 1, 1, 0, b_bo_size},
               {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, c_bo_size}};
  }

  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("SD_ELWMUL Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
}

template <typename InT, typename WtT, typename OutT>
std::once_flag elwmul<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t elwmul<InT, WtT, OutT>::elwmul_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag elwmul<InT, WtT, OutT>::instr_reg_flag_;

template class elwmul<std::uint16_t, std::uint16_t, std::uint16_t>;

} // namespace sd

} // namespace ryzenai
