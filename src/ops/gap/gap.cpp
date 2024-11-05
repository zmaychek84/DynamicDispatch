/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>

#include <iomanip>
#include <iterator>
#include <string>

#include <ops/gap/gap.hpp>
#include <ops/op_interface.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>

#include "utils/dpu_mdata.hpp"

#include <ops/gap/gap_lp.h>
#include <txn_helper/txn_helper.hpp>

namespace ryzenai {
/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename OutT>
void gap<InT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<conv_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;

  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key =
        "gap_" + get_instr_key(txn_fname_prefix_, mat.Z, mat.F, mat.K, mat.N);
    instructions.push_back(std::make_pair(key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename OutT>
std::string gap<InT, OutT>::get_instr_key(std::string prefix, int64_t zp,
                                          int64_t K, int64_t Mi0,
                                          int64_t Mi1) const {
  return prefix + "_" + std::to_string(zp) + "_" + std::to_string(K) + "_" +
         std::to_string(Mi0) + "_" + std::to_string(Mi1);
}

static std::string GetParamKey(std::string prefix, int64_t zp, int64_t K,
                               int64_t Mi0, int64_t Mi1) {
  return prefix + "_" + std::to_string(zp) + "_" + std::to_string(Mi0) + "_" +
         std::to_string(Mi1) + "_" + std::to_string(K);
}

/*
 * Global Average Pool (GAP) class constructor
 */
template <typename InT, typename OutT>
gap<InT, OutT>::gap(const std::string &ifmDtype, const std::string &ofmDtype,
                    bool load_xrt,
                    const std::map<std::string, std::any> &attr) {
  this->compute_lp = true;
  if (!Utils::get_env_var("DISABLE_COMPUTE_GAP_LP", "").empty()) {
    this->compute_lp = false;
  }
  this->debug_lp = false;
  if (!Utils::get_env_var("DEBUG_GAP_LP", "").empty()) {
    this->debug_lp = true;
  }
  this->use_runtime_qdq = false;
  if (!Utils::get_env_var("USE_GAP_RUNTIME_QDQ", "").empty()) {
    this->use_runtime_qdq = true;
  }
  this->patch_txn_bin = false;
  if (!Utils::get_env_var("PATCH_GAP_TXN_BIN", "").empty()) {
    this->patch_txn_bin = true;
  }

  txnbin_a_header = {{"uint16", "a16"}};
  txnbin_acc_header = {{"uint16", "c16"}};
  xclbin_a_header = {{"int16", "a16"}, {"int8", "a8"}};
  xclbin_acc_header = {{"int16", "acc16"}, {"int8", "acc8"}};

  ifmDtype_ = ifmDtype;
  ofmDtype_ = ofmDtype;
  ifmDtypeSize_ = sizeof(InT);
  ofmDtypeSize_ = sizeof(OutT);

  gap_id_ = gap_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\ConvDwc.xclbin";

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME));
  txn_fname_prefix_ =
      "gap_" + txnbin_a_header.at(ifmDtype_) + txnbin_acc_header.at(ofmDtype_);

  default_shapes_["gap_a16c16"] = std::vector<conv_shapes>{};
  default_shapes_["gap_a16c16"].emplace_back(35881, 49, 1, 1024);

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
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

  /* New params based on attributes */
  if (attr.count("zero_point") &&
      attr.at("zero_point").type() == typeid(std::vector<int>)) {
    const auto &zp_vector =
        std::any_cast<const std::vector<int> &>(attr.at("zero_point"));
    for (const auto &zp : zp_vector) {
      zp_ = zp;
    }
  } else {
    std::cout << "Zero Point not found or not of correct type." << std::endl;
  }

  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));

    if (input_shape_vector.size() == 4) {
      inputShape_[0] = input_shape_vector[1];
      inputShape_[1] = input_shape_vector[2];
      inputShape_[2] = input_shape_vector[3];
    } else {
      std::cout << "Input Shape attribute does not have the expected number of "
                   "elements."
                << std::endl;
    }
    RYZENAI_LOG_TRACE(
        "Conv: InputShape: " + std::to_string(input_shape_vector[0]) + ", " +
        std::to_string(input_shape_vector[1]) + ", " +
        std::to_string(input_shape_vector[2]) + ", " +
        std::to_string(input_shape_vector[3]));
  } else {
    std::cout << "Input Shape attribute not found or not of correct type."
              << std::endl;
  }

  if (attr.count("output_shape") &&
      attr.at("output_shape").type() == typeid(std::vector<int>)) {
    const auto &output_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("output_shape"));

    if (output_shape_vector.size() == 4) {
      outputShape_[0] = output_shape_vector[1];
      outputShape_[1] = output_shape_vector[2];
      outputShape_[2] = output_shape_vector[3];
    } else {
      std::cout << "Output Shape attribute does not have the expected number "
                   "of elements."
                << std::endl;
    }
    RYZENAI_LOG_TRACE(
        "Conv: OutputShape: " + std::to_string(output_shape_vector[0]) + ", " +
        std::to_string(output_shape_vector[1]) + ", " +
        std::to_string(output_shape_vector[2]) + ", " +
        std::to_string(output_shape_vector[3]));
  } else {
    std::cout << "Output Shape attribute not found or not of correct type."
              << std::endl;
  }

  if (inputShape_[0] >= 8) {
    kernelInputShape_[0] = inputShape_[1];
    kernelInputShape_[1] = inputShape_[0] / 8;
    kernelInputShape_[2] = inputShape_[2];
    kernelInputShape_[3] = 8;
  } else {
    kernelInputShape_[0] = inputShape_[1];
    kernelInputShape_[1] = inputShape_[0] / 4;
    kernelInputShape_[2] = inputShape_[2];
    kernelInputShape_[3] = 4;
  }

  kernelOutputShape_[0] = outputShape_[1];
  kernelOutputShape_[1] = outputShape_[0] / 8;
  kernelOutputShape_[2] = outputShape_[2];
  kernelOutputShape_[3] = 8;
  Transaction &txn = Transaction::getInstance();
  if (this->compute_lp) {
    // std::cout << "Computing Layer Params..." << std::endl;
    this->lp = get_layer_params(attr);
    if (this->debug_lp) {
      std::cout << "Layer Params:" << std::endl;
      for (auto &param : this->lp) {
        std::cout << param << std::endl;
      }
    }
  } else {
    lp.resize(64);
    std::string lp_key = GetParamKey("gapData_layer", zp_, inputShape_[0],
                                     inputShape_[1], inputShape_[2]) +
                         "_lp";
    std::string lp_binary = txn.get_txn_str(lp_key);
    txn.GetBinData(lp_binary, lp, false);
  }

  std::call_once(logger_flag_, []() {
    std::string header = "gap_id (K Mi0 Mi1) Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "IFM_copy_time(ns) IFM_sync_time(ns) "
                         "OFM_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[Gap] ID: " + std::to_string(gap_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME + ", (a_dtype, c_dtype): (" +
                    ifmDtype_ + ", " + ofmDtype_ + ")");
}

template <typename InT, typename OutT>
void gap<InT, OutT>::WriteToFile(void *src, uint64_t length) {
  uint8_t *dataPtr = (uint8_t *)src;
  std::string testDataFolder = OpInterface::get_dd_base_dir() + "\\" + "tests" +
                               "\\" + "cpp" + "\\" + "unit_tests" + "\\" +
                               "testDataMladf" + "\\" + "GeneratedWeights";

  std::string fileName = testDataFolder + "\\" +
                         GetParamKey("wtsGenerated", zp_, inputShape_[0],
                                     inputShape_[1], inputShape_[2]) +
                         ".txt";
  std::ofstream wts32_fp(fileName);

  for (int i = 0; 4 * i + 3 < length; i++) {
    // print 4 nibbles (in reverse order)
    for (int j = 3; j >= 0; j--) {
      wts32_fp << std::setw(1) << std::hex << ((dataPtr[4 * i + j] & 0xF0) >> 4)
               << std::setw(0);
      wts32_fp << std::setw(1) << std::hex << (dataPtr[4 * i + j] & 0x0F)
               << std::setw(0);
    }
    wts32_fp << std::endl;
  }
  wts32_fp.close();
}

template <typename InT, typename OutT>
void gap<InT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("gap initialize_const_params(ptr) ...");
  auto rep_count = int(lp[8]);
  size_t write_offset = 0;
  size_t concatenateWeightParamsLength = 0;

  for (int i = 0; i < rep_count; i++) {
    io.write(write_offset, lp.data(), lp.size());
    write_offset += lp.size();
    concatenateWeightParamsLength += lp.size();
  }

  if (debug_) {
    auto val = io.read(0, concatenateWeightParamsLength);
    WriteToFile(val.data(), concatenateWeightParamsLength);
  }
  RYZENAI_LOG_TRACE("Gap initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename OutT>
void gap<InT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Gap initialize_const_params ...");
  int totalWtsSize = 64 * lp[8];

  weightCopyTime_ = 0;
  weightFormatTime_ = 0;
  weightSyncTime_ = 0;
  /* Create input/output BOs */
  const size_t CONST_BO_SIZE = totalWtsSize;
  const size_t IFM_BO_SIZE =
      (kernelInputShape_[0] * kernelInputShape_[1] * kernelInputShape_[2] *
       kernelInputShape_[3] * ifmDtypeSize_);
  const size_t OFM_BO_SIZE =
      ((kernelOutputShape_[0]) * kernelOutputShape_[1] *
       (kernelOutputShape_[2]) * kernelOutputShape_[3] * ofmDtypeSize_);
  RYZENAI_LOG_TRACE("Gap: IFM_BO_SIZE:" + std::to_string(IFM_BO_SIZE) +
                    " CONST_BO_SIZE:" + std::to_string(CONST_BO_SIZE) +
                    " OFM_BO_SIZE:" + std::to_string(OFM_BO_SIZE));
  constBo_ =
      xrt::bo(xrt_ctx_->get_device(), CONST_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
  ifmBo_ = xrt::bo(xrt_ctx_->get_device(), IFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));
  ofmBo_ = xrt::bo(xrt_ctx_->get_device(), OFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));
  auto weightCopyStart = GET_ELAPSED_TIME_NS();
  auto weightFormatStart = GET_ELAPSED_TIME_NS();
  uint8_t *b_bo_map = constBo_.map<uint8_t *>();
  auto bo_const = BoConst(b_bo_map);
  initialize_const_params(bo_const, const_params);
  auto weightFormatStop = GET_ELAPSED_TIME_NS();
  weightFormatTime_ +=
      static_cast<int64_t>(weightFormatStop - weightFormatStart);
  auto weightCopyStop = GET_ELAPSED_TIME_NS();
  auto weightSyncStart = GET_ELAPSED_TIME_NS();
  constBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto weightSyncStop = GET_ELAPSED_TIME_NS();
  weightCopyTime_ = static_cast<int64_t>(weightCopyStop - weightCopyStart);
  weightSyncTime_ = static_cast<int64_t>(weightSyncStop - weightSyncStart);
  RYZENAI_LOG_TRACE("Gap initialize_const_params ... DONE");
}

/*
 * perform gap c = a * w. w is stored in the object with initilize_weights
 * method.
 *
 * @param a pointer to activation matrix
 * @param a_shape tuple containing the shape of the activation matrix
 * @param c pointer to store the result of gap
 *
 * @return none
 */
template <typename InT, typename OutT>
void gap<InT, OutT>::execute(std::vector<Tensor> &input,
                             std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("Gap execute ...");

  xrt::bo param_bo;

  ifmBo_.write(input.at(0).data);
  ifmBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto instr_bo_key = "gap_" + txn_fname_prefix_ + "_" + std::to_string(zp_) +
                      "_" + std::to_string(inputShape_[1]) + "_" +
                      std::to_string(inputShape_[2]) + "_" +
                      std::to_string(inputShape_[0]);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));

  auto kernel_ = xrt_ctx_->get_kernel();
  xrt::run run;
  // launch the Gap kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  // kernel call for Gap that supports transaction binary flow
  run = kernel_(2, instr_bo, instr_bo_words,
                constBo_.address() + DDR_AIE_ADDR_OFFSET,
                ifmBo_.address() + DDR_AIE_ADDR_OFFSET,
                ofmBo_.address() + DDR_AIE_ADDR_OFFSET, 0, 0);
  run.wait2();
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);

  // sync output activation to host memory
  ofmBo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  ofmBo_.read(output.at(0).data);

  RYZENAI_LOG_TRACE("Gap execute ... DONE");
}

/*
 * method to set debug flag
 *
 * When the debug flag is set, execute method will write input, weights and
 * output matricies to a filed. the filename will be
 * ryzenai_qlinear2_<execute_num>_<matrix>.txt
 *
 * @param debug bool value to enable disable debug feature. turned off by
 * default
 *
 * @return none
 */
template <typename InT, typename OutT> void gap<InT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename OutT>
const std::vector<uint8_t> gap<InT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return this->get_transaction_bin();
}

template <typename InT, typename OutT>
void gap<InT, OutT>::set_params(const std::string &modelName) {
  std::string XCLBIN_FNAME;
  if (modelName == "m3uec") {
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() + "\\xclbin\\" + "stx" +
                   "\\ConvDwcGap_Psi.xclbin";
  } else if (modelName == "pst") {
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() + "\\xclbin\\" + "stx" +
                   "\\Conv_Pst.xclbin";
  }
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename OutT>
std::vector<OpArgMap> gap<InT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  int totalWtsSize = 64 * lp[8];

  size_t const_params_bo_size = totalWtsSize;
  size_t ifm_bo_size =
      (inputShape_[0] * inputShape_[1] * inputShape_[2] * ifmDtypeSize_);
  size_t ofm_bo_size =
      ((outputShape_[0]) * outputShape_[1] * (outputShape_[2]) * ofmDtypeSize_);

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, ifm_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 0, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, ofm_bo_size}};

  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("Gap Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename OutT>
std::vector<uint8_t>
gap<InT, OutT>::get_layer_params(const std::map<std::string, std::any> &attrs) {
  gap_lp::LayerInfo layer_info;
  // TODO: get batch_size
  layer_info.batch_size = 1;

  layer_info.ifm_height = (uint16_t)this->inputShape_[2];
  layer_info.ifm_width = (uint16_t)this->inputShape_[1];
  layer_info.ifm_depth = (uint16_t)this->inputShape_[0];

  layer_info.ofm_height = (uint16_t)this->outputShape_[2];
  layer_info.ofm_width = (uint16_t)this->outputShape_[1];
  layer_info.ofm_depth = (uint16_t)this->outputShape_[0];

  layer_info.div_shift =
      (uint8_t)std::any_cast<const std::vector<int> &>(attrs.at("div_shift"))
          .at(0);
  layer_info.div_factor =
      (uint32_t)std::any_cast<const std::vector<int> &>(attrs.at("div_factor"))
          .at(0);
  int lsb =
      std::any_cast<const std::vector<int> &>(attrs.at("offset_lsb")).at(0);
  int msb =
      std::any_cast<const std::vector<int> &>(attrs.at("offset_msb")).at(0);

  // TODO: Below is a hack to pass 64 bit integer as DD schema currently only
  // supports 32 bit integer in the attributes. Require support for 64 bit int
  // in DD
  uint64_t offset = (uint32_t)lsb;
  uint64_t temp = (uint32_t)msb;
  temp <<= 32;
  offset |= temp;
  layer_info.offset = offset;

  return computeLayerParams(layer_info);
}

/**
 * Helper function to read txn binary from file, embed zp in it (if rt_const_pad
 * is true) and return it
 * */
template <typename InT, typename OutT>
std::vector<uint8_t> gap<InT, OutT>::get_transaction_bin() const {
  std::string txn_key =
      "gap_" + txn_fname_prefix_ + "_" +
      (!this->patch_txn_bin ? (std::to_string(zp_) + "_") : "") +
      std::to_string(inputShape_[1]) + "_" + std::to_string(inputShape_[2]) +
      "_" + std::to_string(inputShape_[0]);

  Transaction &txn = Transaction::getInstance();
  std::vector<uint8_t> txnData = txn.get_txn_bvec(txn_key);

  if (this->patch_txn_bin) {
    // Runtime constant padding
    uint32_t zp = uint16_t(zp_);
    uint32_t pad_val = zp | (zp << 16);
    auto patchedTxnData = prepend_mtile_const_pad_txn(txnData, pad_val, 8, 1);
    if (this->debug_) {
      // Dump paddedTxnData
      std::string filePath = OpInterface::get_dd_base_dir() + "\\" + "tests" +
                             "\\" + "cpp" + "\\" + "unit_tests" + "\\" +
                             "testDataMladf" + "\\" +
                             "gap_patched_txn_dump.bin";
      if (!patchedTxnData.empty()) {
        Utils::dumpBinary(patchedTxnData.data(),
                          patchedTxnData.size() * sizeof(patchedTxnData[0]),
                          filePath);
      }
    }
    return patchedTxnData;
  }

  return txnData;
}

template <typename InT, typename OutT>
std::once_flag gap<InT, OutT>::logger_flag_;

template <typename InT, typename OutT> uint64_t gap<InT, OutT>::gap_count = 0;

template <typename InT, typename OutT>
std::once_flag gap<InT, OutT>::instr_reg_flag_;

template class gap<uint16_t, uint16_t>;

} // namespace ryzenai
