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
#include <numeric>
#include <sstream>
#include <tuple>
#include <utility>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include <utils/dpu_mdata.hpp>

#include <ops/mladfelwmul/mladfelwmul.hpp>
#include <ops/op_interface.hpp>
#include <txn_container.hpp>
#include <utils/instruction_registry.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>
#include <xrt_context/xrt_context.hpp>
// #include <utils/utils.h>

// AIE Driver header
#include "xaiengine.h"

#include "ops/ops_common/matmul_matrix.hpp"

using namespace matmul_matrix;

namespace ryzenai {

static constexpr int32_t layer_params_size = 64;
static constexpr int32_t wgt_size_1x4096x4096 = 1024;

static void process_shape(std::vector<size_t> &input_shape) {
  if (input_shape.size() != 1 && !input_shape.empty() && input_shape[0] == 1) {
    input_shape.erase(input_shape.begin());
  }
}

template <typename InT, typename WtT, typename OutT>
std::string ml_adf_elw_mul<InT, WtT, OutT>::get_instr_key(
    std::string prefix, const std::vector<size_t> &dimensions) const {
  std::string key = "mladfelwmul_" + prefix;
  for (const auto &dim : dimensions) {
    key += "_" + std::to_string(dim);
  }
  return key;
}

/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename WtT, typename OutT>
void ml_adf_elw_mul<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  for (const auto &[mkey, value] : default_shapes_) {
    auto iter = default_shapes_.find(mkey);
    std::vector<std::vector<size_t>> &supported_shapes = iter->second;
    for (int i = 0; i < supported_shapes.size(); i++) {
      auto shape = supported_shapes[i];
      auto key = get_instr_key(mkey, shape);
      instructions.push_back(std::make_pair(key, false));
    }
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
ml_adf_elw_mul<InT, WtT, OutT>::ml_adf_elw_mul(const std::string &a_dtype,
                                               const std::string &b_dtype,
                                               const std::string &c_dtype,
                                               bool load_xrt) {
  txnbin_a_header = {{"uint16", "a16"}};
  txnbin_c_header = {{"uint16", "acc16"}};

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_["mladfelwmul_a16acc16"] = std::vector<std::vector<size_t>>();
  default_shapes_["mladfelwmul_a16acc16"].push_back({512, 4096});
  default_shapes_["mladfelwmul_a16acc16"].push_back({512, 64, 64});
  default_shapes_["mladfelwmul_a16acc16"].push_back({512, 128, 128});
  default_shapes_["mladfelwmul_a16acc16"].push_back({512, 256, 256});
  default_shapes_["mladfelwmul_a16acc16"].push_back({256, 512, 512});
  default_shapes_["mladfelwmul_a16acc16"].push_back({128, 512, 512});
  default_shapes_["mladfelwmul_a16acc16"].push_back({128, 256, 256});
  default_shapes_["mladfelwmul_a16acc16"].push_back({256, 256, 256});
  default_shapes_["mladfelwmul_a16acc16"].push_back({256, 128, 128});
  default_shapes_["mladfelwmul_a16acc16"].push_back({4096, 4096});
  // raw shape is the actual shape from ONNX
  raw_shapes_["mladfelwmul_a16acc16"] = std::vector<std::vector<size_t>>();
  raw_shapes_["mladfelwmul_a16acc16"].push_back({512, 4096});
  raw_shapes_["mladfelwmul_a16acc16"].push_back({512, 64, 64});
  raw_shapes_["mladfelwmul_a16acc16"].push_back({512, 128, 128});
  raw_shapes_["mladfelwmul_a16acc16"].push_back({512, 256, 256});
  raw_shapes_["mladfelwmul_a16acc16"].push_back({256, 512, 512});
  raw_shapes_["mladfelwmul_a16acc16"].push_back({128, 512, 512});
  raw_shapes_["mladfelwmul_a16acc16"].push_back({128, 256, 256});
  raw_shapes_["mladfelwmul_a16acc16"].push_back({256, 256, 256});
  raw_shapes_["mladfelwmul_a16acc16"].push_back({256, 128, 128});
  raw_shapes_["mladfelwmul_a16acc16"].push_back({4096, 4096});
  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  ml_adf_elw_mul_id_ = ml_adf_elw_mul_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                             ryzenai::MLADF_ELWMUL_A16_QDQ_XCLBIN_PATH;

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME));

  txn_fname_prefix_ = "mladfelwmul_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_c_header.at(c_dtype_);
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("txn_fname_prefix : {}", txn_fname_prefix_));

  KERNEL_M_MAX = 512;

  if (load_xrt) {
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

  std::call_once(logger_flag_, []() {
    std::string header =
        "ml_adf_elw_mul_id M K N kernel_m kernel_k kernel_n Execute"
        "time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[ADF_MUL] ID: " + std::to_string(ml_adf_elw_mul_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype_ + ", " +
                    b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void ml_adf_elw_mul<InT, WtT, OutT>::set_params(
    const std::string &model_name, std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;
  if (model_name == "PST" || model_name == "PSS") {
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                   ryzenai::MLADF_ELWMUL_A16_QDQ_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }

  kernel_x_shape_ = input_shape;
  kernel_z_shape_ = input_shape;
  if (input_shape.size() == 3 && input_shape[0] == 1 &&
      input_shape[1] == 4096 && input_shape[2] == 4096) {
    is_1x4096x4096 = true;
  }

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

// used in fusion_rt.cpp
template <typename InT, typename WtT, typename OutT>
void ml_adf_elw_mul<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("MLADFElWMUL initialize_const_params(ptr) ...");
  // configure const param here, for example QDQ
  if (const_params.size() == 1) {
    return;
  }
  DD_THROW_IF((const_params.size() != 2),
              OpsFusion::dd_format("Unsupported const spec for MLADFElwmul\n") +
                  OpsFusion::dd_format("(Details : #const params == 2 or 1({})",
                                       const_params.size()));
  kernel_y_shape_ = const_params.at(0).shape;
  // NOTE(haozhu12): if ifm shape is 1x4096x4096 and wgt shape is 1,
  // then wgt size is 512 according to kernel team
  if (kernel_y_shape_.size() == 1 && kernel_y_shape_[0] == 1) {
    is_1x4096x4096 = true;
  }
  process_shape(kernel_y_shape_);

  size_t wgt_sz =
      std::accumulate(kernel_y_shape_.begin(), kernel_y_shape_.end(), size_t{1},
                      std::multiplies{}) *
      sizeof(WtT);

  auto expand_vector = [](const WtT *original, size_t M) {
    std::vector<WtT> expanded(2 * M);
    for (size_t i = 0; i < M; ++i) {
      expanded[2 * i] = original[i];
      expanded[2 * i + 1] = original[i];
    }
    return expanded;
  };

  auto expand_vector_special = [](const WtT *original, size_t M) {
    std::vector<WtT> expanded(2 * M);
    for (size_t i = 0; i < M; ++i) {
      expanded[2 * i] = original[0];
      expanded[2 * i + 1] = original[0];
    }
    return expanded;
  };

  std::vector<WtT> w_expand =
      expand_vector(static_cast<const WtT *>(const_params.at(0).data), wgt_sz);
  // NOTE(haozhu12): if ifm shape is 1x4096x4096 and wgt shape is 1,
  // then wgt size is 1024 according to kernel team
  if (is_1x4096x4096) {
    RYZENAI_LOG_TRACE("special shape");
    wgt_sz = wgt_size_1x4096x4096;
    w_expand = expand_vector_special(
        static_cast<const WtT *>(const_params.at(0).data), wgt_sz);
  }
  io.write(0, (char *)w_expand.data(), wgt_sz * 2);
  RYZENAI_LOG_TRACE("MLADFElwmul initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void ml_adf_elw_mul<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("MLADFElWMUL initialize_const_params ...");
  DD_THROW_IF((const_params.size() != 2),
              OpsFusion::dd_format("Unsupported const spec for MLADFElwadd\n") +
                  OpsFusion::dd_format("(Details : #const params == 2 ({})",
                                       const_params.size()));
  kernel_y_shape_ = const_params.at(0).shape;
  uint8_t *layer_params = (uint8_t *)const_params.at(1).data;
  const size_t PARAMS_BO_SIZE = layer_params_size * sizeof(uint8_t);
  /* Create input/output BOs */
  size_t B_BO_SIZE =
      2 *
      std::accumulate(kernel_y_shape_.begin(), kernel_y_shape_.end(), size_t{1},
                      std::multiplies{}) *
      b_dtype_size_;
  // if ifm shape is 1x4096x4096, wgt shape is 1,
  // then wgt bo size is 2 * 512 *dtype according to kernel team
  if (is_1x4096x4096) {
    B_BO_SIZE = 2 * wgt_size_1x4096x4096 * b_dtype_size_;
  }
  const size_t A_BO_SIZE =
      std::accumulate(kernel_x_shape_.begin(), kernel_x_shape_.end(), size_t{1},
                      std::multiplies{}) *
      a_dtype_size_;
  const size_t C_BO_SIZE =
      std::accumulate(kernel_z_shape_.begin(), kernel_z_shape_.end(), size_t{1},
                      std::multiplies{}) *
      c_dtype_size_;

  RYZENAI_LOG_TRACE(
      "MLADFELWMUL: PARAMS_BO_SIZE: " + std::to_string(PARAMS_BO_SIZE) +
      " A_BO_SIZE:" + std::to_string(A_BO_SIZE) + " B_BO_SIZE:" +
      std::to_string(B_BO_SIZE) + " C_BO_SIZE:" + std::to_string(C_BO_SIZE));
  params_bo_ =
      xrt::bo(xrt_ctx_->get_device(), PARAMS_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));

  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));

  // params_bo copy
  params_copy_time_ = 0;
  params_sync_time_ = 0;
  auto params_copy_start = GET_ELAPSED_TIME_NS();
  uint8_t *params_bo_map = params_bo_.map<uint8_t *>();
  memcpy((void *)params_bo_map, (void *)layer_params, PARAMS_BO_SIZE);
  auto params_copy_stop = GET_ELAPSED_TIME_NS();
  params_copy_time_ =
      static_cast<int64_t>(params_copy_stop - params_copy_start);
  // params_bo sync
  auto params_sync_start = GET_ELAPSED_TIME_NS();
  params_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto params_sync_stop = GET_ELAPSED_TIME_NS();
  params_sync_time_ =
      static_cast<int64_t>(params_sync_stop - params_sync_start);

  // b_bo copy
  b_copy_time_ = 0;
  b_sync_time_ = 0;
  auto b_copy_start = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = b_bo_.map<WtT *>();
  auto bo_const = BoConst(b_bo_map);
  initialize_const_params(bo_const, const_params);
  auto b_copy_stop = GET_ELAPSED_TIME_NS();

  // b_bo sync
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();

  b_copy_time_ = static_cast<int64_t>(b_copy_stop - b_copy_start);
  b_sync_time_ = static_cast<int64_t>(b_sync_stop - b_sync_start);
  RYZENAI_LOG_TRACE("MLADFElWMUL initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void ml_adf_elw_mul<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                             std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("MLADFElWMUL execute ...");
  const int a_idx = 0;
  InT *a = (InT *)input.at(a_idx).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;

  auto exec_start = GET_ELAPSED_TIME_NS();

  a_shape_ = input.at(a_idx).shape;
  c_shape_ = a_shape_;

  size_t a_size = std::accumulate(a_shape_.begin(), a_shape_.end(), size_t{1},
                                  std::multiplies{}) *
                  sizeof(InT);
  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  // a_size: copy original shape of data from input tensor into bo
  memcpy((void *)a_bo_map, (void *)a, a_size);
  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  // prepare inst_bo and param_bo
  process_shape(kernel_x_shape_);
  auto instr_bo_key = get_instr_key(txn_fname_prefix_, kernel_x_shape_);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  ;
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, params_bo_, b_bo_,
                                            a_bo_, c_bo_, 0, true, false);
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
  auto c_size = std::accumulate(c_shape_.begin(), c_shape_.end(), size_t{1},
                                std::multiplies{}) *
                c_dtype_size_;
  memcpy((void *)aie_out, (void *)c_bo_map, c_size);
  auto c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = static_cast<int64_t>(c_copy_stop - c_copy_start);
  auto exec_end = GET_ELAPSED_TIME_NS();
  RYZENAI_LOG_INFO(
      std::to_string(ml_adf_elw_mul_id_) + " " + std::to_string(a_shape_[0]) +
      " " + std::to_string(a_shape_[1]) +
      " " + // std::to_string(w_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

// need to change to multiple dimension
// used in fuse_ops.hpp
template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> ml_adf_elw_mul<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  // maybe need to preprocess shape to remove batch dimension
  auto a_shape = input.at(0).shape;
  auto wgt_shape = input.at(1).shape;
  // assgin kernel shape for usage in initialize_const_params
  kernel_x_shape_ = a_shape;
  kernel_y_shape_ = wgt_shape;
  process_shape(a_shape);
  process_shape(wgt_shape);

  std::string txn_key = get_instr_key(txn_fname_prefix_, a_shape);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

// used in fuse_ops.hpp
// important
template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
ml_adf_elw_mul<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  auto layer_params = input.at(2);

  DD_THROW_IF(
      (layer_params.dtype != "uint8"),
      OpsFusion::dd_format("Invalid dtype: expected int8 for MLADFElwmul\n") +
          OpsFusion::dd_format("(Details : type is ({})", layer_params.dtype));

  DD_THROW_IF(
      (layer_params.shape.size() != 1 ||
       layer_params.shape[0] != layer_params_size),
      OpsFusion::dd_format(
          "Invalid shape: expected shape[0] to be 64 for MLADFElwmul\n"));

  size_t total_bytes = layer_params.shape[0] * sizeof(uint8_t);
  std::vector<uint8_t> data(total_bytes);

  std::memcpy(data.data(), layer_params.data, total_bytes);

  return data;
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> ml_adf_elw_mul<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  kernel_x_shape_ = input.at(0).shape;
  kernel_y_shape_ = input.at(1).shape;
  // check if ifm shape is 1x4096x4096
  if (kernel_x_shape_.size() == 3 && kernel_x_shape_[0] == 1 &&
      kernel_x_shape_[1] == 4096 && kernel_x_shape_[2] == 4096 &&
      kernel_y_shape_[0] == 1) {
    is_1x4096x4096 = true;
  }

  process_shape(kernel_x_shape_);
  process_shape(kernel_y_shape_);
  size_t input_1_bo_size =
      std::accumulate(kernel_x_shape_.begin(), kernel_x_shape_.end(), size_t{1},
                      std::multiplies{}) *
      sizeof(InT);
  size_t const_bo_size =
      std::accumulate(kernel_y_shape_.begin(), kernel_y_shape_.end(), size_t{1},
                      std::multiplies{}) *
      sizeof(WtT) * 2;
  // if ifm shape is 1x4096x4096, wgt shape is 1,
  // then wgt bo size is 2 * 512 * dtype according to kernel team
  if (is_1x4096x4096) {
    const_bo_size = 2 * wgt_size_1x4096x4096 * sizeof(WtT);
  }

  size_t output_bo_size =
      std::accumulate(kernel_x_shape_.begin(), kernel_x_shape_.end(), size_t{1},
                      std::multiplies{}) *
      sizeof(OutT);
  // QDQ param size
  size_t param_size = input.at(2).shape[0];
  RYZENAI_LOG_INFO("set arg map.");
  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 2, 0, 0, input_1_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, const_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 0, 2, 0, param_size},
      {OpArgMap::OpArgType::OUTPUT, 3, 3, 0, output_bo_size}};
  return arg_map;
}

template <typename InT, typename WtT, typename OutT>
std::once_flag ml_adf_elw_mul<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t ml_adf_elw_mul<InT, WtT, OutT>::ml_adf_elw_mul_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag ml_adf_elw_mul<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
void ml_adf_elw_mul<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
std::vector<size_t> ml_adf_elw_mul<InT, WtT, OutT>::map_padded_shape(
    const std::vector<size_t> &dims) const {
  return {};
}

template class ml_adf_elw_mul<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
