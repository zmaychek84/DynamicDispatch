/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <utils/instruction_registry.hpp>
#include <xrt_context/xrt_context.hpp>

#include "txn_container.hpp"
#include <ops/op_interface.hpp>
#include <ops/sd/matmul.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include <xaiengine.h>

#include "ops/ops_common/dtype_utils.h"
#include "ops/ops_common/matmul_matrix.hpp"

using namespace matmul_matrix;

namespace ryzenai {
namespace sd {

template <typename InT, typename WtT, typename OutT>
void matmul<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;

  std::vector<std::vector<size_t>> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key = get_instr_key(txn_fname_prefix_, mat);
    // auto param_key = get_instr_key(param_fname_prefix_, mat) + "_param";
    instructions.push_back(std::make_pair(key, false));
    // layer_params.push_back(std::make_pair(param_key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
  // xrt_ctx_->get_registry().add_layer_params(layer_params);
}

template <typename InT, typename WtT, typename OutT>
std::string
matmul<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                      const std::vector<size_t> &mat) const {
  std::string out_str = prefix;
  for (size_t i = 0; i < mat.size(); i++) {
    out_str += "_" + std::to_string(mat[i]);
  }
  return out_str;
}

// matmul constructor
template <typename InT, typename WtT, typename OutT>
matmul<InT, WtT, OutT>::matmul(const std::string &a_dtype,
                               const std::string &b_dtype,
                               const std::string &c_dtype, bool load_xrt,
                               const std::map<std::string, std::any> &attr) {
  txnbin_a_header = {{"bfloat16", "a16bf"}, {"bfp16ebs8", "a16bfp"}};
  txnbin_b_header = {{"bfloat16", "w16bf"}, {"bfp16ebs8", "w16bfp"}};
  txnbin_acc_header = {{"bfloat16", "acc16bf"}};

  // unet layer1
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 1024, 2560, 640});
  // unet layer2
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 1024, 640, 5120});
  // unet layer3
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 1024, 640, 640});
  // unet layer4
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 256, 1280, 10240});
  // unet layer5
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 256, 1280, 1280});
  // unet layer6
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 256, 5120, 1280});
  // unet layer7
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 4096, 1280, 320});
  // unet layer8
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 4096, 320, 2560});
  // unet layer9
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 4096, 320, 320});
  // unet layer10
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 64, 1280, 10240});
  // unet layer11
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 64, 1280, 1280});
  // unet layer12
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 64, 5120, 1280});
  // unet layer13
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 77, 768, 1280});
  // unet layer14
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 77, 768, 320});
  // unet layer15
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 77, 768, 640});
  // from unet gemm1
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 1, 320, 1280});
  // from unet gemm2
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 1, 1280, 320});
  // from unet gemm3
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 1, 1280, 640});
  // from unet gemm4
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 1, 1280, 1280});
  // from unet gemm10
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{1, 2, 2048, 1536});
  // from unet gemm11
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{1, 2, 256, 1536});

  // from sd3 mmdit 512 layer1
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 1024, 1536, 1536});
  // from sd3 mmdit 512 layer2
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 1024, 1536, 6144});
  // from sd3 mmdit 512 layer3
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 1024, 1536, 64});
  // from sd3 mmdit 512 layer4
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 1024, 6144, 1536});

  // from sd3 mmdit 512 and 1024
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 1, 1536, 1536});

  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 154, 1536, 1536});

  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 154, 1536, 6144});

  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 154, 4096, 1536});

  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 154, 6144, 1536});

  // from sd3 mmdit 1024 layer1
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 4096, 1536, 1536});
  // from sd3 mmdit 1024 layer2
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 4096, 1536, 6144});
  // from sd3 mmdit 1024 layer3
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 4096, 1536, 64});
  // from sd3 mmdit 1024 layer4
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 4096, 6144, 1536});
  // from sd3 vae decoder 1024 layer1
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{1, 16384, 512, 512});

  // from matmul_add_to_matmul pass
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 64, 1280, 5120});

  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 256, 1280, 5120});

  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 1024, 640, 2560});

  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{2, 4096, 320, 1280});

  // vae
  default_shapes_["sd_matmul_a16bfw16bfacc16bf"].emplace_back(
      std::vector<size_t>{1, 4096, 512, 512});

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  matmul_id_++;

  XCLBIN_FNAME_ =
      OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\SDMatmul.xclbin";
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME_));
  txn_fname_prefix_ = "sd_matmul_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
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

  if (attr.count("weight_shape") &&
      attr.at("weight_shape").type() == typeid(std::vector<int>)) {
    const auto &weight_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("weight_shape"));
    if (weight_shape_vector.size() == 2) {
      weightShape_[0] = weight_shape_vector[0]; // K
      weightShape_[1] = weight_shape_vector[1]; // N
    } else {
      RYZENAI_LOG_INFO(
          "Weight Shape attribute does not have the expected number of "
          "elements.Number of passed : " +
          std::to_string(weight_shape_vector.size()) + ", Expected:2");
    }
    RYZENAI_LOG_TRACE(
        "matmul: WeightShape: " + std::to_string(weight_shape_vector[0]) +
        ", " + std::to_string(weight_shape_vector[1]));
  } else {
    RYZENAI_LOG_INFO(
        "Weight Shape attribute not found or not of correct type.");
  }

  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));

    if (input_shape_vector.size() == 3) {
      inputShape_[0] = input_shape_vector[0]; // B
      inputShape_[1] = input_shape_vector[1]; // M
      inputShape_[2] = input_shape_vector[2]; // K
    } else {
      RYZENAI_LOG_INFO(
          "Input Shape attribute does not have the expected number of "
          "elements.Number of passed : " +
          std::to_string(input_shape_vector.size()) + ", Expected:3");
    }
    RYZENAI_LOG_TRACE(
        "Conv: InputShape: " + std::to_string(input_shape_vector[0]) + ", " +
        std::to_string(input_shape_vector[1]) + ", " +
        std::to_string(input_shape_vector[2]));
  } else {
    RYZENAI_LOG_INFO("Input Shape attribute not found or not of correct type.");
  }

  if (attr.count("output_shape") &&
      attr.at("output_shape").type() == typeid(std::vector<int>)) {
    const auto &output_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("output_shape"));

    if (output_shape_vector.size() == 3) {
      outputShape_[0] = output_shape_vector[0]; // B
      outputShape_[1] = output_shape_vector[1]; // M
      outputShape_[2] = output_shape_vector[2]; // N
    } else {
      RYZENAI_LOG_INFO(
          "Input Shape attribute does not have the expected number of "
          "elements.Number of passed : " +
          std::to_string(output_shape_vector.size()) + ", Expected:3");
    }
    RYZENAI_LOG_TRACE(
        "Conv: OutputShape: " + std::to_string(output_shape_vector[0]) + ", " +
        std::to_string(output_shape_vector[1]) + ", " +
        std::to_string(output_shape_vector[2]));
  } else {
    RYZENAI_LOG_INFO(
        "Output Shape attribute not found or not of correct type.");
  }

  B_ = inputShape_[0];
  M_ = inputShape_[1];
  K_ = inputShape_[2];
  N_ = outputShape_[2];
  std::call_once(logger_flag_, []() {
    std::string header = "ipu_wrapper_id M K N Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[SD Matmul] ID: " + std::to_string(matmul_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME_ +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype + ", " +
                    b_dtype + ", " + c_dtype + ")");
}

template <typename InT, typename WtT, typename OutT>
void matmul<InT, WtT, OutT>::set_params(const std::string &model_name,
                                        std::vector<size_t> input_shape) {
  DD_ASSERT(
      input_shape.size() == 4,
      OpsFusion::dd_format("sd matmul input_shape set_params expects 4. Got {}",
                           input_shape.size()));
  B_ = input_shape[0];
  M_ = input_shape[1];
  K_ = input_shape[2];
  N_ = input_shape[3];
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void matmul<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("SD matmul initialize_const_params(ptr) ...");
  DD_THROW_IF(
      (const_params.size() != 1) || (const_params.at(0).shape.size() != 1),
      OpsFusion::dd_format(
          "Unsupported const spec for SD matmul\n"
          "(Details : #const params == 1 ({}), Const param1 dim == 1 ({})",
          const_params.size(), const_params.at(0).shape.size()));
  auto wts_size = (const_params.at(0).shape[0] * sizeof(WtT));
  auto expected_wts_size = K_ * N_ * sizeof(WtT);
  DD_ASSERT(wts_size == expected_wts_size,
            OpsFusion::dd_format("SDMatMul expect weight size {}, Got {}",
                                 expected_wts_size, wts_size));
  io.write(0, (int32_t *)const_params.at(0).data, wts_size);
}

template <typename InT, typename WtT, typename OutT>
void matmul<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("SD matmul initialize_const_params ...");
  // Check the number of inputs
  DD_ASSERT((const_params.size() == 1),
            OpsFusion::dd_format("SD matmul expects one constant. Got {}",
                                 const_params.size()));
  std::vector<Tensor> input;
  std::vector<Tensor> output;
  size_t A_BO_SIZE, B_BO_SIZE, C_BO_SIZE;
  A_BO_SIZE = B_BO_SIZE = C_BO_SIZE = 0;
  auto args_map_list = this->get_buffer_reqs(input, output, attr);
  for (const auto &args_map : args_map_list) {
    if (args_map.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
      B_BO_SIZE = args_map.size;
    }
    if (args_map.arg_type == OpArgMap::OpArgType::INPUT) {
      A_BO_SIZE = args_map.size;
    }
    if (args_map.arg_type == OpArgMap::OpArgType::OUTPUT) {
      C_BO_SIZE = args_map.size;
    }
  }
  RYZENAI_LOG_TRACE("SD matmul: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE size:" + std::to_string(C_BO_SIZE));

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(0));

  // copy b_bo
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  auto b_copy_start = GET_ELAPSED_TIME_NS();
  auto b_format_start = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = b_bo_.map<WtT *>();
  auto bo_const = BoConst(b_bo_map);
  initialize_const_params(bo_const, const_params, attr);
  auto b_format_stop = GET_ELAPSED_TIME_NS();
  auto b_copy_stop = GET_ELAPSED_TIME_NS();
  b_format_time_ += static_cast<int64_t>(b_format_stop - b_format_start);
  b_copy_time_ = static_cast<int64_t>(b_copy_stop - b_copy_start);

  // sync b_bo
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();
  b_sync_time_ = static_cast<int64_t>(b_sync_stop - b_sync_start);
}

template <typename InT, typename WtT, typename OutT>
void matmul<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                     std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("Matmul execute ...");
  DD_ASSERT(input.size() == 1,
            OpsFusion::dd_format("sd matmul input tensor expects 1. Got {}",
                                 input.size()));
  DD_ASSERT(output.size() == 1,
            OpsFusion::dd_format("sd matmul output tensor expects 1. Got {}",
                                 output.size()));

  InT *a = (InT *)input.at(0).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;

  auto exec_start = GET_ELAPSED_TIME_NS();

  size_t a_size = B_ * M_ * K_ * sizeof(InT);
  RYZENAI_LOG_TRACE("act1 matmul: a_size:" + std::to_string(a_size));

  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  memcpy((void *)a_bo_map, (void *)a, a_size);
  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  std::vector<size_t> param_shape = {B_, M_, K_, N_};
  auto instr_bo_key = get_instr_key(txn_fname_prefix_, param_shape);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  size_t instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, a_bo_, b_bo_, c_bo_,
                                            0, 0, true, false);
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
  memcpy((void *)aie_out, (void *)c_bo_map, B_ * M_ * N_ * sizeof(OutT));
  auto c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = static_cast<int64_t>(c_copy_stop - c_copy_start);
  auto exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(matmul_id_) + " " + std::to_string(B_) + " " +
      std::to_string(M_) + " " + std::to_string(K_) + " " + std::to_string(N_) +
      " " + std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename OutT>
void matmul<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> matmul<InT, WtT, OutT>::get_transaction_bin() const {
  std::string txn_key = txn_fname_prefix_ + "_" + std::to_string(B_) + "_" +
                        std::to_string(M_) + "_" + std::to_string(K_) + "_" +
                        std::to_string(N_);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> matmul<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return this->get_transaction_bin();
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> matmul<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return {};
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> matmul<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  size_t ifm_bo_size = (B_ * M_ * K_ * sizeof(InT));
  size_t const_params_bo_size = (K_ * N_ * sizeof(WtT));
  size_t ofm_bo_size = (B_ * M_ * N_ * sizeof(OutT));
  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, ifm_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, ofm_bo_size}};
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("SD matmul argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
const std::vector<uint32_t> matmul<InT, WtT, OutT>::get_layer_params() const {
  std::vector<size_t> param_shape = {B_, M_, K_, N_};
  auto param_key = get_instr_key(txn_fname_prefix_, param_shape) + "_param";
  Transaction &txn = Transaction::getInstance();
  auto uint8_data = txn.get_txn_bvec(param_key);
  DD_ASSERT(uint8_data.size() == 24,
            OpsFusion::dd_format("wrong lp size {}", uint8_data.size()));
  auto *uint32_data = reinterpret_cast<const uint32_t *>(uint8_data.data());
  std::vector<uint32_t> layer_params(
      uint32_data, uint32_data + uint8_data.size() / sizeof(uint32_t));
  // std::cerr << "layer_params: ";
  // for (int i = 0; i < layer_params.size(); i++) {
  //   std::cerr << layer_params[i] << " ";
  // }
  // std::cerr << std::endl;
  return layer_params;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint16_t>
matmul<InT, WtT, OutT>::shuffle_wts_bf16(const std::vector<float> &wts) const {
  std::vector<uint16_t> wts_bf16(wts.size());
  auto lp = get_layer_params();
  DD_ASSERT(wts.size() == K_ * N_, "wts size and shape mismatch");
  // K_ = k_iter * l1k, N_ = n_iter * l1n
  // paddings?
  auto l1k = lp[4];
  auto k_iter = K_ / l1k;
  auto l1n = lp[5];
  auto n_iter = N_ / l1n;
  // std::cerr << "l1k: " << l1k << " k_iter: " << k_iter << " l1n: " << l1n
  //           << " n_iter: " << n_iter << std::endl;
  // wts.reshape(k_iter, l1k, n_iter, l1n).transpose(2,0,1,3)
  // to n_iter, k_iter, l1k, l1n
  for (uint32_t i = 0; i < k_iter; i++) {
    for (uint32_t j = 0; j < l1k; j++) {
      for (uint32_t k = 0; k < n_iter; k++) {
        for (uint32_t l = 0; l < l1n; l++) {
          wts_bf16[k * k_iter * l1k * l1n + i * l1k * l1n + j * l1n + l] =
              float_to_bfloat16(
                  wts[i * l1k * n_iter * l1n + j * n_iter * l1n + k * l1n + l]);
        }
      }
    }
  }
  return wts_bf16;
}

template <typename InT, typename WtT, typename OutT>
std::once_flag matmul<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t matmul<InT, WtT, OutT>::matmul_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag matmul<InT, WtT, OutT>::instr_reg_flag_;

template class matmul<std::uint16_t, std::uint16_t, std::uint16_t>;
} // namespace sd
} // namespace ryzenai
