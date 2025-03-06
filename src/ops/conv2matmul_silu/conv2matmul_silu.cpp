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

#include "utils/ctrl_pkt_utils.hpp"

#include "ops/ops_common/psu_silu_lut_bf16_512.h"
#include <ops/conv2matmul_silu/conv2matmul_silu.hpp>
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
#include "xaiengine.h"

#include "ops/ops_common/matmul_matrix.hpp"

using namespace matmul_matrix;

namespace ryzenai {

static std::tuple<size_t, size_t, size_t>
extract_MKN(const std::vector<Tensor> &inputs, std::string input_format) {
  size_t M;
  if (inputs.at(0).shape.size() == 2) {
    M = inputs.at(0).shape.at(0);
  } else if (inputs.at(0).shape.size() == 3) { // has batch_dim
    M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(1);
  } else if (inputs.at(0).shape.size() == 4) { // has batch_dim
    if (input_format == "NHWC") {              // NHWC
      M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(1) *
          inputs.at(0).shape.at(2);
    } else { // NCHW
      M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(2) *
          inputs.at(0).shape.at(3);
    }
  } else {
    throw std::runtime_error("Input Shape is not supported");
  }

  size_t K = inputs.at(1).shape.at(1);
  size_t N = inputs.at(1).shape.at(0);

  return std::make_tuple(M, K, N);
}

template <typename InT, typename WtT, typename OutT>
std::tuple<size_t, size_t, size_t>
conv2matmul_silu<InT, WtT, OutT>::map_padded_shape(size_t M, size_t K,
                                                   size_t N) const {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<matrix_shapes> &supported_shapes = iter->second;
  size_t Mo = M;
  size_t Ko = K;
  size_t No = N;
  size_t fidx = 0;
  bool f_found = false;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    if (M == mat.M && K == mat.K && N == mat.N) {
      fidx = i;
      f_found = true;
      break;
    }
  }

  if (f_found) {
    iter = default_shapes_.find(txn_fname_prefix_);
    const std::vector<matrix_shapes> &actual_shapes = iter->second;
    auto mat = actual_shapes.at(fidx);
    Mo = mat.M;
    Ko = mat.K;
    No = mat.N;
  } else {
    throw std::runtime_error("Cannot find the shape");
  }
  return std::make_tuple(Mo, Ko, No);
}

/*
 * conv2matmul_silu is an experimental class to offload int8_t * int8_t matrix
 * multiplications to AIE. this class uses lite runtime stack to interface with
 * XRT and submit jobs to IPU. Even though the instructions in this template
 * supports transaction format, it can be extended to support DPU sequence
 * format.
 */

/* Utility function to set the kernel shape based on the weights dimensions
 * Pick kernel shape using weight matrix size
 * Select OPT shapes when a_type is int8
 * Select Llamav2 shapes when a_type is int16
 * Need to fix this to pick shapes independent of the datatype*/
template <typename InT, typename WtT, typename OutT>
void conv2matmul_silu<InT, WtT, OutT>::set_kernel_shapes() {
  // Use largest M dimension as the default
  //    NOTE: smaller M's can be selected in run_aie if needed
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;

  kernel_x_shape_[1] = w_shape_[0];
  kernel_y_shape_[0] = w_shape_[0];
  kernel_y_shape_[1] = w_shape_[1];
  kernel_z_shape_[1] = w_shape_[1];
}

/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename WtT, typename OutT>
void conv2matmul_silu<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  for (const auto &[mkey, value] : default_shapes_) {
    auto iter = default_shapes_.find(mkey);
    std::vector<matrix_shapes> &supported_shapes = iter->second;
    for (size_t i = 0; i < supported_shapes.size(); i++) {
      auto mat = supported_shapes.at(i);
      auto key = "conv2gemm_silu_" + get_instr_key(mkey, mat.M, mat.K, mat.N);
      auto param_key = "conv2gemm_silu_" +
                       get_instr_key(mkey, mat.M, mat.K, mat.N) + "_param";
      instructions.push_back(std::make_pair(key, false));
      layer_params.push_back(std::make_pair(param_key, false));
    }
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
  xrt_ctx_->get_registry().add_layer_params(layer_params);
}
template <typename InT, typename WtT, typename OutT>
std::string conv2matmul_silu<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                                            size_t m, size_t k,
                                                            size_t n) const {
  return prefix + "_" + std::to_string(m) + "_" + std::to_string(k) + "_" +
         std::to_string(n);
}

template <typename InT, typename WtT, typename OutT>
conv2matmul_silu<InT, WtT, OutT>::conv2matmul_silu(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"uint16", "a16"}, {"int16", "a16"}};

  txnbin_b_header = {
      {"uint8", "w8"}, {"int8", "w8"}, {"uint4", "w4"}, {"int4", "w4"}};

  txnbin_acc_header = {{"uint16", "acc16"}, {"bfloat16", "accbf16"}};

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_["conv2gemm_silu_4x4_a16w4accbf16"] =
      std::vector<matrix_shapes>{};

  default_shapes_["conv2gemm_silu_4x4_a16w4accbf16"].emplace_back(1, 3072,
                                                                  8192);
  default_shapes_["conv2gemm_silu_4x4_a16w4accbf16"].emplace_back(64, 3072,
                                                                  8192);

  default_shapes_["conv2gemm_silu_8x4_a16w4accbf16"] =
      std::vector<matrix_shapes>{};

  default_shapes_["conv2gemm_silu_8x4_a16w4accbf16"].emplace_back(1, 3072,
                                                                  8192);
  default_shapes_["conv2gemm_silu_8x4_a16w4accbf16"].emplace_back(64, 3072,
                                                                  8192);

  default_shapes_["conv2gemm_silu_8x4_a16w4accbf16"].emplace_back(1, 1536,
                                                                  8960);
  default_shapes_["conv2gemm_silu_8x4_a16w4accbf16"].emplace_back(64, 1536,
                                                                  8960);

  // raw shape is the actual shape from ONNX
  raw_shapes_["conv2gemm_silu_4x4_a16w4accbf16"] = std::vector<matrix_shapes>{};

  raw_shapes_["conv2gemm_silu_4x4_a16w4accbf16"].emplace_back(1, 3072, 8192);
  raw_shapes_["conv2gemm_silu_4x4_a16w4accbf16"].emplace_back(64, 3072, 8192);

  raw_shapes_["conv2gemm_silu_8x4_a16w4accbf16"] = std::vector<matrix_shapes>{};

  raw_shapes_["conv2gemm_silu_8x4_a16w4accbf16"].emplace_back(1, 3072, 8192);
  raw_shapes_["conv2gemm_silu_8x4_a16w4accbf16"].emplace_back(64, 3072, 8192);

  raw_shapes_["conv2gemm_silu_8x4_a16w4accbf16"].emplace_back(1, 1536, 8960);
  raw_shapes_["conv2gemm_silu_8x4_a16w4accbf16"].emplace_back(64, 1536, 8960);

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  b_shift_value_ = 0;
  if (b_dtype == "int4") {
    b_shift_value_ = 1;
  }
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  conv2matmul_silu_id_ = conv2matmul_silu_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dd_base_dir() + ryzenai::PSU_4x4_A16W8_QDQ_XCLBIN_PATH;

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
    RYZENAI_LOG_TRACE("conv2gemm_silu: DesignFormat: " + design_param_);
  }

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    txn_fname_prefix_ = "conv2gemm_silu_4x4_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);
    param_fname_prefix_ = "conv2gemm_silu_4x4_" + txnbin_a_header.at(a_dtype_) +
                          txnbin_b_header.at(b_dtype_) +
                          txnbin_acc_header.at(c_dtype_);
  } else if (design_param_.find("8x4") != std::string::npos) { // 8x4 design
    txn_fname_prefix_ = "conv2gemm_silu_8x4_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);
    param_fname_prefix_ = "conv2gemm_silu_8x4_" + txnbin_a_header.at(a_dtype_) +
                          txnbin_b_header.at(b_dtype_) +
                          txnbin_acc_header.at(c_dtype_);
  }

  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));

    if (input_shape_vector.size() == 4) {
      inputShape_[0] = input_shape_vector[0];
      inputShape_[1] = input_shape_vector[1];
      inputShape_[2] = input_shape_vector[2];
      inputShape_[3] = input_shape_vector[3];
    } else {
      std::cout << "Input Shape attribute does not have the expected number of "
                   "elements.Number of passed : input_shape_vector.size(), "
                   "Expected:4"
                << std::endl;
    }
    RYZENAI_LOG_TRACE(
        "conv2gemm_silu: InputShape: " + std::to_string(input_shape_vector[0]) +
        ", " + std::to_string(input_shape_vector[1]) + ", " +
        std::to_string(input_shape_vector[2]) + ", " +
        std::to_string(input_shape_vector[3]));
  } else {
    std::cout << "Input Shape attribute not found or not of correct type."
              << std::endl;
  }

  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("txn_fname_prefix : {}", txn_fname_prefix_));
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("param_fname_prefix : {}", param_fname_prefix_));

  if (attr.count("input_format") &&
      attr.at("input_format").type() == typeid(std::vector<std::string>)) {
    const auto &input_format_vector =
        std::any_cast<const std::vector<std::string> &>(
            attr.at("input_format"));

    if (input_format_vector.size() == 1) {
      input_format_ = input_format_vector[0];
    } else {
      std::cout
          << "Input Format attribute does not have the expected number of "
             "elements.Number of passed : input_format_vector.size(), "
             "Expected:4"
          << std::endl;
    }
    RYZENAI_LOG_TRACE("conv2gemm_silu: InputFormat: " + input_format_);
  } else {
    std::cout << "Input Format attribute not found or not of correct type."
              << std::endl;
  }

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
        "conv2matmul_silu_id M K N kernel_m kernel_k kernel_n Execute"
        "time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE(
      "[CONV2GEMM_SILU] ID: " + std::to_string(conv2matmul_silu_id_) +
      ", XCLBIN: " + XCLBIN_FNAME + ", (a_dtype, b_dtype, c_dtype): (" +
      a_dtype_ + ", " + b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void conv2matmul_silu<InT, WtT, OutT>::set_params(
    const std::string &model_name, std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;
  if (model_name == "4x4PSU") {
    is_ctrl_pkt_ = 1;
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::PSU_4x4_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "8x4PSU") {
    is_ctrl_pkt_ = 1;
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::PSU_8x4_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "8x4HFDS") {
    is_ctrl_pkt_ = 1;
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                   ryzenai::HFDS_8x4_A16W8_QDQ_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }

  auto [M, K, N] =
      map_padded_shape(input_shape.at(0), input_shape.at(1), input_shape.at(2));
  KERNEL_M_MAX = M;

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void conv2matmul_silu<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("conv2matmul_silu initialize_const_params(ptr) ...");

  DD_THROW_IF(
      (const_params.size() < 4),
      OpsFusion::dd_format("Unsupported const spec for conv2matmul_silu\n") +
          OpsFusion::dd_format(
              "(Details : #const params == 1 ({}), Const param dim == 2 ({})",
              const_params.size(), const_params.at(0).shape.size()));

  auto K_orig = const_params.at(0).shape.at(1);
  w_shape_[0] = const_params.at(0).shape.at(1); // K
  w_shape_[1] = const_params.at(0).shape.at(0); //

  auto weights = (WtT *)const_params.at(0).data;

  auto qdq = (int64_t *)const_params.at(1).data;
  auto qdq_params = (int32_t *)const_params.at(2).data;
  auto silu_qdq_params = (int32_t *)const_params.at(3).data;

  int32_t *c1_vec, *c2_vec;
  if (const_params.size() == 6) {
    c1_vec = (int32_t *)const_params.at(4).data;
    c2_vec = (int32_t *)const_params.at(5).data;
  }

  set_kernel_shapes();

  std::vector<WtT> buf(w_shape_[0] * w_shape_[1]);

  size_t M;
  if ((design_param_ == "4x4PSU" && input_format_ == "NHWC") ||
      (design_param_ == "8x4PSU" && input_format_ == "NHWC") ||
      (design_param_ == "8x4HFDS" && input_format_ == "NHWC")) {
    M = inputShape_[1] * inputShape_[2];
  } else { // NCHW
    M = inputShape_[2] * inputShape_[3];
  }

  SUBV_T key = {(int)M, (int)w_shape_[0], (int)w_shape_[1]};
  auto subv_mode = search_subv_mode(key, b_shift_value_, design_param_);
  if (subv_mode < 0) {
    throw std::runtime_error("conv2matmul_silu : Invalid subv mode");
  }

  format_wgt_trans<WtT>(weights, buf.data(), subv_mode, (int)w_shape_[0],
                        (int)w_shape_[1], (int)K_orig, b_shift_value_);
  SUBV_T subv = get_subv(subv_mode);
  auto Msubv = subv[0];
  auto Ksubv = subv[1];
  auto Nsubv = subv[2];

  // padding Msubv and Nsubv
  qdq_params[qdq_Mv_idx] = 0;
  qdq_params[qdq_Nv_idx] = 0;

  auto total_size = (Ksubv * Nsubv) >> b_shift_value_;
  auto qdq_size = Nsubv * sizeof(int64_t);
  auto qdq_params_size = matmul_matrix::QDQparam_size * sizeof(int32_t);
  auto silu_qdq_params_size = matmul_matrix::QDQparam_size * sizeof(int32_t);
  const int Ngran = 8;
  auto qdq_ngran_size = Ngran * sizeof(int64_t);
  auto c1_vec_ngran_size = Ngran * sizeof(int32_t);
  auto c2_vec_ngran_size = Ngran * sizeof(int32_t);
  int const lut_ab_size = sizeof(silu_lutab);
  int const lut_cd_size = sizeof(silu_lutcd);
  //// WGT + Bias(all zeros)
  { // This section of the code interleaves bias with weights Nsubv of bias
    // with every K x N
    size_t write_offset = 0;
    for (int N_shard = 0; N_shard < (w_shape_[1]) / (Nsubv); N_shard++) {
      for (int K_shard = 0; K_shard < (w_shape_[0]) / (Ksubv); K_shard++) {
        if (b_shift_value_) {
          io.write(write_offset,
                   (void *)&buf[(N_shard * w_shape_[0] * Nsubv) / 2 +
                                (K_shard * total_size)],
                   (total_size));
        } else {
          io.write(write_offset,
                   (void *)&buf[(N_shard * w_shape_[0] * Nsubv) +
                                (K_shard * total_size)],
                   (total_size));
        }
        write_offset += total_size;
        if (const_params.size() == 6) {
          for (int Nchunk = 0; Nchunk < Nsubv; Nchunk += Ngran) {
            io.write(write_offset, (void *)&qdq[N_shard * Nsubv + Nchunk],
                     qdq_ngran_size);
            write_offset += qdq_ngran_size;
            io.write(write_offset, (void *)&c1_vec[N_shard * Nsubv + Nchunk],
                     c1_vec_ngran_size);
            write_offset += c1_vec_ngran_size;
            io.write(write_offset, (void *)&c2_vec[N_shard * Nsubv + Nchunk],
                     c2_vec_ngran_size);
            write_offset += c2_vec_ngran_size;
          }
        } else {
          io.write(write_offset, (void *)&qdq[N_shard * Nsubv], qdq_size);
          write_offset += qdq_size;
        }
      }
    }
    io.write(write_offset, (void *)qdq_params, qdq_params_size);
    write_offset += qdq_params_size;

    // silu lut
    io.write(write_offset, (void *)silu_lutab, lut_ab_size);
    write_offset += lut_ab_size;
    io.write(write_offset, (void *)silu_lutcd, lut_cd_size);
    write_offset += lut_cd_size;
    io.write(write_offset, (void *)silu_qdq_params, silu_qdq_params_size);
    write_offset += silu_qdq_params_size;
  }

  RYZENAI_LOG_TRACE("conv2matmul_silu initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void conv2matmul_silu<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("conv2matmul_silu initialize_const_params ...");

  DD_THROW_IF(
      (const_params.size() < 3) || (const_params.at(0).shape.size() != 2),
      OpsFusion::dd_format("Unsupported const spec for conv2matmul_silu\n") +
          OpsFusion::dd_format(
              "(Details : #const params == 1 ({}), Const param dim == 2 ({})",
              const_params.size(), const_params.at(0).shape.size()));

  w_shape_[0] = const_params.at(0).shape.at(1); // K
  w_shape_[1] = const_params.at(0).shape.at(0); // N
  int Ksubv;
  size_t M;

  set_kernel_shapes();
  M = inputShape_[2] * inputShape_[3];

  SUBV_T key = {(int)M, (int)w_shape_[0], (int)w_shape_[1]};
  auto subv_mode = search_subv_mode(key, b_shift_value_, design_param_);
  if (subv_mode < 0) {
    throw std::runtime_error("conv2matmul_silu : Invalid subv mode");
  }
  SUBV_T subv = get_subv(subv_mode);
  Ksubv = subv[1];

  // qdqc
  auto qdq_params = (int32_t *)const_params.at(2).data;
  size_t size_interleaved_qdq;
  if (const_params.size() == 4) {
    size_interleaved_qdq = w_shape_[0] * w_shape_[1] / Ksubv * sizeof(int64_t);
  } else { // channelwise qdq
    // C0, C1 and C2 are all vectors
    size_interleaved_qdq =
        w_shape_[0] * w_shape_[1] / Ksubv *
        (sizeof(int64_t) + sizeof(int32_t) + sizeof(int32_t));
  }
  int const lut_ab_size = sizeof(silu_lutab);
  int const lut_cd_size = sizeof(silu_lutcd);
  size_interleaved_qdq += matmul_matrix::QDQparam_size * sizeof(int32_t) * 2 +
                          lut_ab_size + lut_cd_size;

  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  /* Create input/output BOs */
  size_t B_BO_SIZE;
  if (b_shift_value_) {
    B_BO_SIZE = (kernel_y_shape_[0] * kernel_y_shape_[1] * b_dtype_size_) / 2 +
                size_interleaved_qdq;
  } else {
    B_BO_SIZE = (kernel_y_shape_[0] * kernel_y_shape_[1] * b_dtype_size_) +
                size_interleaved_qdq;
  }

  const size_t A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_);
  const size_t C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_);

  RYZENAI_LOG_TRACE("CONV2GEMM_SILU: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
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
    auto [Mo, Ko, No] = map_padded_shape(M, w_shape_[0], w_shape_[1]);
    // Based on the mapped_shape to get the meta json file
    std::vector<uint8_t> json_data;
    try {
      auto json_key = "conv2gemm_silu_" +
                      get_instr_key(param_fname_prefix_, Mo, Ko, No) +
                      "_ctrl_meta";
      Transaction &txn = Transaction::getInstance();
      json_data = txn.get_txn_bvec(json_key);
    } catch (...) {
      is_ctrl_pkt_ = 0;
    }

    if (is_ctrl_pkt_) {
      std::cout << "ctrlpkt patching" << std::endl;
      RYZENAI_LOG_TRACE("conv2matmul_silu patch ctrlpkt ... START");
      // get param_bo address
      auto param_bo_key = "conv2gemm_silu_" +
                          get_instr_key(param_fname_prefix_, Mo, Ko, No) +
                          "_param";
      const xrt::bo &param_bo =
          xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;

      // Get ctrl pkt patch info from json
      std::vector<CtrlPktPatchInfo> ctrlpkt_info;
      ctrlpkt_info = json_str_to_ctrlpkt_patch_info(json_data);

      // Get the ctrl pkt
      auto ctrl_bo_key = "conv2gemm_silu_" +
                         get_instr_key(param_fname_prefix_, Mo, Ko, No) +
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
      RYZENAI_LOG_TRACE("conv2matmul_silu patch ctrlpkt ... DONE");
    }
  }
  RYZENAI_LOG_TRACE("conv2matmul_silu initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void conv2matmul_silu<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                               std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("conv2matmul_silu execute ...");

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  auto exec_start = GET_ELAPSED_TIME_NS();
  if (input.at(0).shape.size() == 4) {
    a_shape_[0] = input.at(0).shape.at(0) * input.at(0).shape.at(1) *
                  input.at(0).shape.at(2);
    a_shape_[1] = input.at(0).shape.at(3);
  } else if (input.at(0).shape.size() == 3) {
    a_shape_[0] = input.at(0).shape.at(0) * input.at(0).shape.at(1);
    a_shape_[1] = input.at(0).shape.at(2);
  } else if (input.at(0).shape.size() == 2) {
    a_shape_[0] = input.at(0).shape.at(0);
    a_shape_[1] = input.at(0).shape.at(1);
  } else {
    throw std::runtime_error(
        "conv2matmul_silu : Invalid shape received for input");
  }

  RYZENAI_LOG_TRACE(
      "CONV2GEMM_SILU: a_shape_[0]:" + std::to_string(a_shape_[0]) +
      " a_shape_[1]:" + std::to_string(a_shape_[1]) +
      " w_shape_[1]:" + std::to_string(w_shape_[1]));

  c_shape_[0] = a_shape_[0];
  c_shape_[1] = w_shape_[1];
  auto aie_out = (OutT *)output.at(0).data;
  auto a = (InT *)input.at(0).data;

  auto [M, K, N] = map_padded_shape(a_shape_[0], a_shape_[1], w_shape_[1]);
  kernel_x_rows = M;

  auto a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  if (K == a_shape_[1]) {
    memcpy((void *)a_bo_map, (void *)a,
           (a_shape_[0] * a_shape_[1] * a_dtype_size_));
  } else {
    for (int i = 0; i < a_shape_[0]; i++) {
      memcpy((void *)&a_bo_map[i * K], (void *)&a[i * a_shape_[1]],
             (a_shape_[1] * a_dtype_size_));
    }
  }
  auto a_copy_stop = GET_ELAPSED_TIME_NS();
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();
  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  // INIT with zeros
  auto instr_bo_key = "conv2gemm_silu_" + txn_fname_prefix_ + "_" +
                      std::to_string(kernel_x_rows) + "_" +
                      std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_y_shape_[1]);
  auto param_bo_key = "conv2gemm_silu_" + param_fname_prefix_ + "_" +
                      std::to_string(kernel_x_rows) + "_" +
                      std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_y_shape_[1]) + "_param";
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
         (c_shape_[0] * c_shape_[1] * c_dtype_size_));
  auto exec_end = GET_ELAPSED_TIME_NS();
  RYZENAI_LOG_INFO(
      std::to_string(conv2matmul_silu_id_) + " " + std::to_string(a_shape_[0]) +
      " " + std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) +
      " " + std::to_string(kernel_x_rows) + " " +
      std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
  RYZENAI_LOG_TRACE("conv2matmul_silu execute ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void conv2matmul_silu<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
conv2matmul_silu<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, N] = extract_MKN(input, input_format_);
  auto [Mo, Ko, No] = map_padded_shape(M, K, N);
  std::string txn_key =
      "conv2gemm_silu_" + get_instr_key(txn_fname_prefix_, Mo, Ko, No);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
conv2matmul_silu<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, N] = extract_MKN(input, input_format_);
  auto [Mo, Ko, No] = map_padded_shape(M, K, N);
  // TODO: Add check to validate tensor shapes
  std::string param_key = "conv2gemm_silu_" +
                          get_instr_key(param_fname_prefix_, Mo, Ko, No) +
                          "_param";
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(param_key);
}

template <typename InT, typename WtT, typename OutT>
std::vector<uint8_t> conv2matmul_silu<InT, WtT, OutT>::get_ctrl_pkts(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, N] = extract_MKN(input, input_format_);
  auto [Mo, Ko, No] = map_padded_shape(M, K, N);
  // TODO: Add check to validate tensor shapes
  std::string ctrl_key = "conv2gemm_silu_" +
                         get_instr_key(param_fname_prefix_, Mo, Ko, No) +
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
conv2matmul_silu<InT, WtT, OutT>::get_ctrl_pkt_patch_info(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, N] = extract_MKN(input, input_format_);
  auto [Mo, Ko, No] = map_padded_shape(M, K, N);
  // TODO: Add check to validate tensor shapes
  try {
    auto ctrl_pkt_meta = "conv2gemm_silu_" +
                         get_instr_key(param_fname_prefix_, Mo, Ko, No) +
                         "_ctrl_meta";
    Transaction &txn = Transaction::getInstance();
    return json_str_to_ctrlpkt_patch_info(txn.get_txn_bvec(ctrl_pkt_meta));
  } catch (...) {
    /*throw std::runtime_error(
        "conv2matmul_silu : Can not file the ctrl_meta.json file");*/
    return {};
  }
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> conv2matmul_silu<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  // input --> [input, weights, output]
  // Check if IO buffers have batch.
  auto [M, K, N] = extract_MKN(input, input_format_);

  auto [Mo, Ko, No] = map_padded_shape(M, K, N);

  int Ksubv;
  SUBV_T key = {(int)M, (int)K, (int)N};
  auto subv_mode = search_subv_mode(key, b_shift_value_, design_param_);
  if (subv_mode < 0) {
    throw std::runtime_error("conv2matmul_silu : Invalid subv mode");
  }
  SUBV_T subv = get_subv(subv_mode);
  Ksubv = subv[1];

  // qdqc
  auto qdq_params = (int32_t *)input.at(3).data;
  size_t size_interleaved_qdq;
  size_t out_idx = 5;
  if (input.size() == 6) {
    size_interleaved_qdq = Ko * No / Ksubv * sizeof(int64_t);
    out_idx = 5;
  } else { // channelwise qdq
    // C0, C1 and C2 are all vectors
    size_interleaved_qdq =
        Ko * No / Ksubv * (sizeof(int64_t) + sizeof(int32_t) + sizeof(int32_t));
    out_idx = 7;
  }
  int const lut_ab_size = sizeof(silu_lutab);
  int const lut_cd_size = sizeof(silu_lutcd);
  size_interleaved_qdq += matmul_matrix::QDQparam_size * sizeof(int32_t) * 2 +
                          lut_ab_size + lut_cd_size;

  size_t const_params_bo_size;
  if (b_shift_value_) {
    const_params_bo_size = (Ko * No * b_dtype_size_) / 2 + size_interleaved_qdq;
  } else {
    const_params_bo_size = (Ko * No * b_dtype_size_) + size_interleaved_qdq;
  }
  size_t input_bo_size = (Mo * Ko * a_dtype_size_);
  size_t output_bo_size = (Mo * No * c_dtype_size_);
  size_t super_kernel_size = get_super_kernel_params(input, output).size();
  size_t ctrl_pkt_size = get_ctrl_pkts(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, out_idx, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size},
      {OpArgMap::OpArgType::CTRL_PKT_BIN, 4, 0, 0, ctrl_pkt_size}};
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("conv2matmul_silu Argmap : {}",
                                         cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
std::once_flag conv2matmul_silu<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t conv2matmul_silu<InT, WtT, OutT>::conv2matmul_silu_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag conv2matmul_silu<InT, WtT, OutT>::instr_reg_flag_;

template class conv2matmul_silu<uint16_t, int8_t, uint16_t>;

} // namespace ryzenai
