/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

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

#include "sd_helper.hpp"
#include <ops/op_interface.hpp>
#include <ops/ops_common/help_file.hpp>
#include <ops/sd/conv2d.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>

#include "txn/txn_utils.hpp"
#include "txn_helper/txn_helper.hpp"
#include "utils/dpu_mdata.hpp"

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <ops/conv/conv_lp.h>

using std::vector;

namespace ryzenai {

namespace sd {

template <typename InT, typename WtT, typename BiasT, typename OutT>
void conv<InT, WtT, BiasT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<sd_conv2d_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key = get_key(txn_fname_prefix_, mat.OC, mat.IC, mat.IH, mat.IW,
                       mat.OH, mat.OW, mat.kh, mat.kw);
    instructions.push_back(std::make_pair(key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
std::string
conv<InT, WtT, BiasT, OutT>::get_key(std::string prefix, int64_t OC, int64_t IC,
                                     int64_t IH, int64_t IW, int64_t OH,
                                     int64_t OW, int64_t kh, int64_t kw) const {
  return prefix + "_" + std::to_string(OC) + "_" + std::to_string(IC) + "_" +
         std::to_string(IH) + "_" + std::to_string(IW) + "_" +
         std::to_string(OH) + "_" + std::to_string(OW) + "_" +
         std::to_string(kh) + "_" + std::to_string(kw);
}

// conv class constructor
template <typename InT, typename WtT, typename BiasT, typename OutT>
conv<InT, WtT, BiasT, OutT>::conv(const std::string &ifm_dtype,
                                  const std::string &weight_dtype,
                                  const std::string &bias_dtype,
                                  const std::string &out_dtype, bool load_xrt,
                                  const std::map<std::string, std::any> &attr)
    : attr_(attr) {
  if (Utils::get_env_var("DEBUG_SD", "0") != "0") {
    this->debug_ = true;
  }
  txnbin_a_header = {{"bfloat16", "a16bf"}, {"bfp16ebs8", "a16bfp"}};
  txnbin_b_header = {
      {"float32", "w16bfp"}, {"bfloat16", "w16bf"}, {"bfp16ebs8", "w16bfp"}};
  txnbin_acc_header = {{"bfloat16", "acc16bf"}};

  ifmDtype_ = ifm_dtype;
  weightDtype_ = weight_dtype;
  biasDtype_ = bias_dtype;
  ofmDtype_ = out_dtype;
  ifmDtypeSize_ = sizeof(InT);
  weightDtypeSize_ = sizeof(WtT);
  ofmDtypeSize_ = sizeof(OutT);
  biasDtypeSize_ = sizeof(BiasT);
  conv_id_ = conv_count++;

  /*TODO:
   * select xclbin based on the input/output types,
   * currently only need one xclbin for one type of ops.
   */
  XCLBIN_FNAME_ =
      OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\SDConv2d.xclbin";
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME_));
  txn_fname_prefix_ = sd_conv_key_ + txnbin_a_header.at(ifmDtype_) +
                      txnbin_b_header.at(weightDtype_) +
                      txnbin_acc_header.at(ofmDtype_);
  // Shape for Unetsdconv2d_a16bfw16bfpacc16bf
  default_shapes_["sd_conv2d_a16bfw16bfpacc16bf"].emplace_back(128, // OC
                                                               256, // IC
                                                               512, // IH
                                                               8,   // IW
                                                               512, // OH
                                                               8,   // OW
                                                               1,   // kh
                                                               1);  // kw

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  ifmCopyTime_ = 0;
  ifmSyncTime_ = 0;
  weightCopyTime_ = 0;
  weightFormatTime_ = 0;
  weightSyncTime_ = 0;
  ofmCopyTime_ = 0;
  ofmSyncTime_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;

  if (attr.count("weight_shape") &&
      attr.at("weight_shape").type() == typeid(std::vector<int>)) {
    const auto &weight_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("weight_shape"));

    if (weight_shape_vector.size() >= 4) {
      weightShape_[0] = weight_shape_vector[0]; // OC
      weightShape_[1] = weight_shape_vector[1]; // kh
      weightShape_[2] = weight_shape_vector[2]; // kw
      weightShape_[3] = weight_shape_vector[3]; // IC
      if (weightShape_[1] != weightShape_[2]) {
        throw std::runtime_error(
            "SD conv only support IHWO data format for weight");
      }
    } else {
      RYZENAI_LOG_INFO("Weight Shape attribute does not have enough elements.");
    }
    RYZENAI_LOG_TRACE(
        "Conv: WeightShape: " + std::to_string(weight_shape_vector[0]) + ", " +
        std::to_string(weight_shape_vector[1]) + ", " +
        std::to_string(weight_shape_vector[2]) + ", " +
        std::to_string(weight_shape_vector[3]));
  } else {
    RYZENAI_LOG_INFO(
        "Weight Shape attribute not found or not of correct type.");
  }

  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));
    if (input_shape_vector[0] != 1) {
      RYZENAI_LOG_INFO(
          "The N dimension for conv2d only support 1 now, but N given is " +
          std::to_string(input_shape_vector[0]));
    }

    if (input_shape_vector.size() == 4) {
      inputShape_[0] = input_shape_vector[1]; // IH
      inputShape_[1] = input_shape_vector[2]; // IW
      inputShape_[2] = input_shape_vector[3]; // IC
    } else {
      RYZENAI_LOG_INFO(
          "Input Shape attribute does not have the expected number of "
          "elements.Number of passed : " +
          std::to_string(input_shape_vector.size()) + ", Expected:4");
    }
    RYZENAI_LOG_TRACE(
        "Conv: InputShape: " + std::to_string(input_shape_vector[0]) + ", " +
        std::to_string(input_shape_vector[1]) + ", " +
        std::to_string(input_shape_vector[2]) + ", " +
        std::to_string(input_shape_vector[3]));
  } else {
    RYZENAI_LOG_INFO("Input Shape attribute not found or not of correct type.");
  }

  if (attr.count("output_shape") &&
      attr.at("output_shape").type() == typeid(std::vector<int>)) {
    const auto &output_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("output_shape"));

    if (output_shape_vector.size() == 4) {
      outputShape_[0] = output_shape_vector[1]; // OH
      outputShape_[1] = output_shape_vector[2]; // OW
      outputShape_[2] = output_shape_vector[3]; // OC
    } else {
      RYZENAI_LOG_INFO(
          "Input Shape attribute does not have the expected number of "
          "elements.Number of passed : " +
          std::to_string(output_shape_vector.size()) + ", Expected:4");
    }
    RYZENAI_LOG_TRACE(
        "Conv: OutputShape: " + std::to_string(output_shape_vector[0]) + ", " +
        std::to_string(output_shape_vector[1]) + ", " +
        std::to_string(output_shape_vector[2]) + ", " +
        std::to_string(output_shape_vector[3]));
  } else {
    RYZENAI_LOG_INFO(
        "Output Shape attribute not found or not of correct type.");
  }

  OC_ = weightShape_[0];
  IC_ = weightShape_[3];
  IH_ = inputShape_[0];
  IW_ = inputShape_[1];
  OH_ = outputShape_[0];
  OW_ = outputShape_[1];
  kh_ = weightShape_[1];
  kw_ = weightShape_[2];
  if (bias_en_) {
    biasShape_[0] = OC_;
  }

  // TODO: support stride not 1, N not 1
  stride_ = 1;
  N_ = 1;

  // TODO: align ifm, weights C, W to 8
  kernelInputShape_[0] = inputShape_[0];     // H
  kernelInputShape_[1] = inputShape_[2] / 8; // IC/8
  kernelInputShape_[2] = inputShape_[1];     // W
  kernelInputShape_[3] = 8;                  // IC parallelism

  kernelWeightShape_[0] = weightShape_[0] / 16; // OCG
  kernelWeightShape_[1] = weightShape_[3] / 8;  // ICG
  kernelWeightShape_[2] = weightShape_[1];      // kh
  kernelWeightShape_[3] = weightShape_[2];      // kw
  kernelWeightShape_[4] = 16;                   // OC parallelism
  kernelWeightShape_[5] = 8;                    // IC parallelism

  kernelBiasShape_[0] = OC_ / 8;
  kernelBiasShape_[1] = 4;
  kernelBiasShape_[2] = 8;

  kernelOutputShape_[0] = outputShape_[0];     // H
  kernelOutputShape_[1] = outputShape_[2] / 8; // C / 8
  kernelOutputShape_[2] = outputShape_[1];     // W
  kernelOutputShape_[3] = 8;                   // C parallelism

  std::call_once(logger_flag_, []() {
    std::string header = "sd_conv2d_id (OC IC IH IW OH OW kh kw) Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "IFM_copy_time(ns) IFM_sync_time(ns) "
                         "OFM_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[SD_CONV2D] ID: " + std::to_string(conv_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME_ +
                    ", (a_dtype, b_dtype, c_dtype): (" + ifmDtype_ + ", " +
                    weightDtype_ + ", " + ofmDtype_ + ")");
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
size_t
conv<InT, WtT, BiasT, OutT>::get_const_bo_size(uint32_t ifm_sv_depth) const {
  auto wts_size = OC_ * kh_ * kw_ * IC_ / 8 * 9;
  // from python, casc_len == 1
  // depth_iter_casc = ceil(ifm_depth/ifm_sv_depth)/casc_len
  size_t depth_iter_casc =
      static_cast<size_t>(std::ceil(IC_ * 1.0 / ifm_sv_depth));
  auto bias_size = OC_ * depth_iter_casc * 4 * sizeof(float);
  return wts_size + bias_size;
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
void conv<InT, WtT, BiasT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Conv initialize_const_params ...");
  auto super_kernel_params = get_super_kernel_params();
  auto super_kernel_size = super_kernel_params.size();
  DD_ASSERT(
      super_kernel_size >= 64,
      OpsFusion::dd_format("sdconv2d load {} bytes lp bin less than 64 bytes",
                           super_kernel_size));

  auto lp_data_ptr = reinterpret_cast<uint32_t *>(super_kernel_params.data());
  ryzenai::sd_helper::layer_params lp(lp_data_ptr);
  lp.print_param();

  if (weightDtype_ == "float32") {
    auto weights = (WtT *)const_params.at(0).data;
    auto wts_total_elements = OC_ * IC_ * kh_ * kw_;
    std::vector<WtT> wgt_reshaped;
    wgt_reshaped.resize(wts_total_elements);

    for (size_t b = 0; b < (size_t)OC_; ++b) {
      for (size_t c = 0; c < (size_t)IC_; ++c) {
        for (size_t h = 0; h < (size_t)kh_; ++h) {
          for (size_t w = 0; w < (size_t)kw_; ++w) {
            // Calculate indices for NHWC and NCHW
            size_t nhwc_idx = b * kh_ * kw_ * IC_ + h * kw_ * IC_ + w * IC_ + c;
            size_t nchw_idx = b * IC_ * kh_ * kw_ + c * kh_ * kw_ + h * kw_ + w;
            // Copy data
            wgt_reshaped[nchw_idx] = weights[nhwc_idx];
          }
        }
      }
    }
    auto wts_tensor = const_params.at(0);
    auto bias_tensor = const_params.at(1);
    std::vector<float> bias_data((float *)bias_tensor.data,
                                 (float *)bias_tensor.data +
                                     bias_tensor.shape[0]);

    std::vector<size_t> wt_shape_oihw{(size_t)OC_, (size_t)IC_, (size_t)kh_,
                                      (size_t)kw_};
    int num_words_bias = 1;
    int num_words_wts = 2;
    std::vector<uint32_t> buffer;
    std::string wts32_file = "cpp_wts32.txt";

    ryzenai::sd_helper::write_datafmt_wts(buffer, wgt_reshaped, wt_shape_oihw,
                                          bias_data, lp, wts32_file, 8,
                                          num_words_bias, num_words_wts);

    io.write(0, buffer.data(), buffer.size() * sizeof(uint32_t));

  } else if (weightDtype_ == "bfp16ebs8") {
    auto const_params_bo_size = get_const_bo_size(lp.ifm_sv_depth);
    DD_ASSERT(const_params_bo_size == const_params.at(0).shape.at(0),
              OpsFusion::dd_format("unexpected const params size: {} vs {}",
                                   const_params_bo_size,
                                   const_params.at(0).shape.at(0)));
    io.write(0, const_params.at(0).data, const_params_bo_size);
  }
  RYZENAI_LOG_TRACE("Conv initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
void conv<InT, WtT, BiasT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  std::vector<Tensor> input;
  std::vector<Tensor> output;
  size_t CONST_BO_SIZE, IFM_BO_SIZE, OFM_BO_SIZE;
  CONST_BO_SIZE = IFM_BO_SIZE = OFM_BO_SIZE = 0;
  auto args_map_list = this->get_buffer_reqs(input, output, attr);
  for (const auto &args_map : args_map_list) {
    if (args_map.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
      CONST_BO_SIZE = args_map.size;
    }
    if (args_map.arg_type == OpArgMap::OpArgType::INPUT) {
      IFM_BO_SIZE = args_map.size;
    }
    if (args_map.arg_type == OpArgMap::OpArgType::OUTPUT) {
      OFM_BO_SIZE = args_map.size;
    }
  }
  RYZENAI_LOG_TRACE("SD_CONV2D: IFM_BO_SIZE:" + std::to_string(IFM_BO_SIZE) +
                    " CONST_BO_SIZE:" + std::to_string(CONST_BO_SIZE) +
                    " OFM_BO_SIZE:" + std::to_string(OFM_BO_SIZE));
  constBo_ =
      xrt::bo(xrt_ctx_->get_device(), CONST_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
  ifmBo_ = xrt::bo(xrt_ctx_->get_device(), IFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));
  ofmBo_ = xrt::bo(xrt_ctx_->get_device(), OFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));

  weightCopyTime_ = 0;
  weightFormatTime_ = 0;
  weightSyncTime_ = 0;
  auto weightCopyStart = GET_ELAPSED_TIME_NS();
  auto weightFormatStart = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = constBo_.map<WtT *>();
  auto bo_const = BoConst(b_bo_map);
  initialize_const_params(bo_const, const_params, attr);
  auto weightFormatStop = GET_ELAPSED_TIME_NS();
  weightFormatTime_ +=
      static_cast<int64_t>(weightFormatStop - weightFormatStart);
  auto weightCopyStop = GET_ELAPSED_TIME_NS();
  auto weightSyncStart = GET_ELAPSED_TIME_NS();
  constBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto weightSyncStop = GET_ELAPSED_TIME_NS();
  weightCopyTime_ = static_cast<int64_t>(weightCopyStop - weightCopyStart);
  weightSyncTime_ = static_cast<int64_t>(weightSyncStop - weightSyncStart);
  RYZENAI_LOG_TRACE("Conv initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
void conv<InT, WtT, BiasT, OutT>::execute(std::vector<Tensor> &input,
                                          std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("Conv execute ...");
  ifmBo_.write(input.at(0).data);
  ifmBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto txnData = this->get_transaction_bin();

  auto i_buf = transaction_op(txnData);
  size_t instr_bo_words = i_buf.get_txn_instr_size();
  xrt::bo instr_bo =
      xrt::bo(xrt_ctx_->get_context(), instr_bo_words,
              xrt::bo::flags::cacheable, xrt_ctx_->get_kernel().group_id(1));
  instr_bo.write(i_buf.get_txn_op().data());
  instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();
  xrt::run run;

  auto run_aie_start = GET_ELAPSED_TIME_NS();
  // TODO: figure out the Bo order
  run = kernel_(2, instr_bo, instr_bo_words,
                ifmBo_.address() + DDR_AIE_ADDR_OFFSET,
                constBo_.address() + DDR_AIE_ADDR_OFFSET,
                ofmBo_.address() + DDR_AIE_ADDR_OFFSET, 0, 0);
  run.wait2();
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);

  /* sync output activation to host memory */
  ofmBo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  ofmBo_.read(output.at(0).data);

  RYZENAI_LOG_TRACE("Conv execute ... DONE");
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
void conv<InT, WtT, BiasT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
const std::vector<uint8_t>
conv<InT, WtT, BiasT, OutT>::get_transaction_bin() const {
  std::string txn_key =
      get_key(txn_fname_prefix_, OC_, IC_, IH_, IW_, OH_, OW_, kh_, kw_);
  Transaction &txn = Transaction::getInstance();
  std::vector<uint8_t> txnData = txn.get_txn_bvec(txn_key);
  return txnData;
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
const std::vector<uint8_t> conv<InT, WtT, BiasT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return this->get_transaction_bin();
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
void conv<InT, WtT, BiasT, OutT>::set_params(
    const std::string &modelName, const sd_conv2d_shapes &shape_info) {
  // TODO: use modelName to find xclbin when need.
  OC_ = shape_info.OC;
  IC_ = shape_info.IC;
  IH_ = shape_info.IH;
  IW_ = shape_info.IW;
  OH_ = shape_info.OH;
  OW_ = shape_info.OW;
  kh_ = shape_info.kh;
  kw_ = shape_info.kw;

  inputShape_[0] = IH_;
  inputShape_[1] = IW_;
  inputShape_[2] = IC_;

  weightShape_[0] = OC_;
  weightShape_[1] = kh_;
  weightShape_[2] = kw_;
  weightShape_[3] = IC_;

  outputShape_[0] = OH_;
  outputShape_[1] = OW_;
  outputShape_[2] = OC_;

  biasShape_[0] = OC_;

  // TODO: align ifm, weights C, W to 8
  kernelInputShape_[0] = inputShape_[0];     // H
  kernelInputShape_[1] = inputShape_[2] / 8; // IC/8
  kernelInputShape_[2] = inputShape_[1];     // W
  kernelInputShape_[3] = 8;                  // IC parallelism

  kernelWeightShape_[0] = weightShape_[0] / 16; // OCG
  kernelWeightShape_[1] = weightShape_[3] / 8;  // ICG
  kernelWeightShape_[2] = weightShape_[1];      // kh
  kernelWeightShape_[3] = weightShape_[2];      // kw
  kernelWeightShape_[4] = 16;                   // OC parallelism
  kernelWeightShape_[5] = 8;                    // IC parallelism

  kernelBiasShape_[0] = OC_ / 8;
  kernelBiasShape_[1] = 4; // repeat 4 times
  kernelBiasShape_[2] = 8;

  kernelOutputShape_[0] = outputShape_[0];     // H
  kernelOutputShape_[1] = outputShape_[2] / 8; // C / 8
  kernelOutputShape_[2] = outputShape_[1];     // W
  kernelOutputShape_[3] = 8;                   // C parallelism

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
std::vector<OpArgMap> conv<InT, WtT, BiasT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  size_t ifm_bo_size =
      (inputShape_[0] * inputShape_[1] * inputShape_[2] * ifmDtypeSize_);
  auto super_kernel_params = get_super_kernel_params();
  auto super_kernel_size = super_kernel_params.size();
  DD_ASSERT(
      super_kernel_size >= 64,
      OpsFusion::dd_format("sdconv2d load {} bytes lp bin less than 64 bytes",
                           super_kernel_size));
  auto lp_data_ptr = reinterpret_cast<uint32_t *>(super_kernel_params.data());
  ryzenai::sd_helper::layer_params lp(lp_data_ptr);
  lp.print_param();
  auto const_params_bo_size = get_const_bo_size(lp.ifm_sv_depth);

  size_t ofm_bo_size =
      ((outputShape_[0]) * outputShape_[1] * (outputShape_[2]) * ofmDtypeSize_);

  RYZENAI_LOG_TRACE("SDConv: IFM_BO_SIZE:" + std::to_string(ifm_bo_size) +
                    " CONST_BO_SIZE:" + std::to_string(const_params_bo_size) +
                    " OFM_BO_SIZE:" + std::to_string(ofm_bo_size));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, ifm_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, ofm_bo_size}};

  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("SD_Conv Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename BiasT, typename OutT>
const std::vector<uint8_t>
conv<InT, WtT, BiasT, OutT>::get_super_kernel_params() const {
  auto param_key =
      get_key(txn_fname_prefix_, OC_, IC_, IH_, IW_, OH_, OW_, kh_, kw_) +
      "_param";
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(param_key);
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
const std::vector<uint8_t> conv<InT, WtT, BiasT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return {};
};

template <typename InT, typename WtT, typename BiasT, typename OutT>
std::once_flag conv<InT, WtT, BiasT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename BiasT, typename OutT>
uint64_t conv<InT, WtT, BiasT, OutT>::conv_count = 0;

template <typename InT, typename WtT, typename BiasT, typename OutT>
std::once_flag conv<InT, WtT, BiasT, OutT>::instr_reg_flag_;

template class conv<std::uint16_t, std::uint8_t, float, std::uint16_t>;
template class conv<std::uint16_t, float, float, std::uint16_t>;

} // namespace sd

} // namespace ryzenai
