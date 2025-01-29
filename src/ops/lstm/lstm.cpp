/*
 Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 Licensed under the MIT License.
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

#include <ops/lstm/lstm.hpp>
#include <ops/op_interface.hpp>
#include <ops/ops_common/help_file.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>

#include "txn_helper/txn_helper.hpp"
#include "utils/dpu_mdata.hpp"

#include "lstm_util.hpp"

namespace ryzenai {
/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename WtT, typename OutT>
void lstm<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;

  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key = get_instr_key(txn_fname_prefix_, mat.M, mat.K, mat.N);
    instructions.push_back(std::make_pair(key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
std::string lstm<InT, WtT, OutT>::get_instr_key(std::string prefix, size_t m,
                                                size_t k, size_t n) const {
  return "lstm_" + prefix + "_" + std::to_string(m) + "_" + std::to_string(k) +
         "_" + std::to_string(n);
}

/*
 * lstm class constructor
 *
 * @param kernel_x_shape tuple containing of M x K dimension base lstm
 * supported on IPU
 * @param kernel_y_shape tuple containing of K x N dimension base lstm
 * supported on IPU
 *
 */
template <typename InT, typename WtT, typename OutT>
lstm<InT, WtT, OutT>::lstm(const std::string &ifmDtype,
                           const std::string &weightDtype,
                           const std::string &ofmDtype, bool load_xrt,
                           const std::map<std::string, std::any> &attr)
    : attr_(attr) {
  txnbin_a_header = {{"uint16", "a16"}};
  txnbin_b_header = {{"uint16", "w16"}};
  txnbin_acc_header = {{"uint16", "c16"}};

  modelNum_ = 320;
  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));

    if (input_shape_vector.size() == 3) {
      inputShape_[0] = input_shape_vector[0];
      inputShape_[1] = input_shape_vector[1];
      inputShape_[2] = input_shape_vector[2];
      modelNum_ = inputShape_[0] * 4;
    } else {
      std::cout
          << "Input Shape attribute does not have the expected number of "
             "elements.Number of passed : input_shape_vector.size(), Expected:3"
          << std::endl;
    }
  }
  if (attr.count("output_shape") &&
      attr.at("output_shape").type() == typeid(std::vector<int>)) {
    const auto &output_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("output_shape"));

    if (output_shape_vector.size() == 3) {
      outputShape_[0] = output_shape_vector[0];
      outputShape_[1] = output_shape_vector[1];
      outputShape_[2] = output_shape_vector[2];
    } else {
      std::cout
          << "Output Shape attribute does not have the expected number of "
             "elements.Number of passed : output_shape_vector.size(), "
             "Expected:3"
          << std::endl;
    }
  }

  std::string model_variant;
  if (attr.end() != attr.find("model_variant")) {
    model_variant = std::any_cast<std::string>(attr.at("model_variant"));
  }

  if (model_variant.size()) {
    std::stringstream ss(model_variant);
    std::vector<std::string> model_variant_tokens;
    std::string token;
    while (std::getline(ss, token, '_')) {
      model_variant_tokens.push_back(token);
    }
    model_variant_ = model_variant_tokens[0];
    modelNum_ = std::stoi(model_variant_tokens[1]);
  }

  DPU_DIR =
      OpInterface::get_dd_base_dir() + "//transaction//" + "stx" + "//lstm//";

  ifmDtype_ = ifmDtype;
  weightDtype_ = weightDtype;
  ofmDtype_ = ofmDtype;
  ifmDtypeSize_ = sizeof(InT);
  weightDtypeSize_ = sizeof(WtT);
  ofmDtypeSize_ = sizeof(OutT);

  lstm_id_ = lstm_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                             "\\xclbin\\stx\\4x2_pso2_model_a16w16_qdq.xclbin";

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME));
  txn_fname_prefix_ = "lstm_" + txnbin_a_header.at(ifmDtype_) +
                      txnbin_b_header.at(weightDtype_) +
                      txnbin_acc_header.at(ofmDtype_);

  default_shapes_["lstm_a16w16c16"] = std::vector<matrix_shapes>{};

  /* Shapes for mswbjvw-320 */
  default_shapes_["lstm_a16w16c16"].emplace_back(80, 1, 64);

  /* Shapes for mswbjvw-640 */
  default_shapes_["lstm_a16w16c16"].emplace_back(160, 1, 64);

  /* Shapes for mswbjvw-1280 */
  default_shapes_["lstm_a16w16c16"].emplace_back(320, 1, 64);

  /* Shapes for mswbjvw-2560 */
  default_shapes_["lstm_a16w16c16"].emplace_back(640, 1, 64);

  weightShape_[0] = 1;
  weightShape_[1] = 1;
  weightShape_[2] = (model_variant_ == "02" ? 721664 : 656128);

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

  this->SetNumConst(6);
  std::call_once(logger_flag_, []() {
    std::string header = "lstm_id (Mi0 Mi1 Mi2 Mo0, Mo1, Mo2) Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "IFM_copy_time(ns) IFM_sync_time(ns) "
                         "OFM_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[lstm] ID: " + std::to_string(lstm_id_) + ", XCLBIN: " +
                    XCLBIN_FNAME + ", (a_dtype, b_dtype, c_dtype): (" +
                    ifmDtype_ + ", " + weightDtype_ + ", " + ofmDtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void lstm<InT, WtT, OutT>::set_params(const int model_num,
                                      std::vector<size_t> input_shape,
                                      std::vector<size_t> weight_shape,
                                      std::vector<size_t> output_shape) {
  modelNum_ = model_num;
  inputShape_[0] = input_shape.at(0);
  inputShape_[1] = input_shape.at(1);
  inputShape_[2] = input_shape.at(2);

  weightShape_[0] = weight_shape.at(0);
  weightShape_[1] = weight_shape.at(1);
  weightShape_[2] = weight_shape.at(2);

  outputShape_[0] = output_shape.at(0);
  outputShape_[1] = output_shape.at(1);
  outputShape_[2] = output_shape.at(2);

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                             "\\xclbin\\stx\\4x2_pso2_model_a16w16_qdq.xclbin";

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME));
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

/*
 * copy weight matrix into XRT BOs with padding and tiling
 *
 * this method copies the weight matrix into XRT BOs. This is re-used for
 * every lstm performed for this object with different activations. weight
 * matrix is padded, tiled and reformatted while copying to XRT BOs. padding
 * is done to align with kernel_y_shape each tile of the weight matrix is of
 * shape kernel_y_shape this method also reformats the matrix b/weight matrix
 * as required by AIE/IPU lstm implementation
 *
 * @param weights pointer to the weight matrix
 * @param w_shape tuple containing the shape of the weight matrix
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void lstm<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("LSTM initialize_const_params(ptr) ...");

  int16_t *lstm0_wts_ptr = nullptr;
  int16_t *lstm1_wts_ptr =
      lstm0_wts_ptr + (model_variant_ == "02" ? 295296 : 229760);

  std::vector<Tensor> lstm0_const_params;
  lstm0_const_params.push_back(const_params[0]);
  lstm0_const_params.push_back(const_params[1]);
  lstm0_const_params.push_back(const_params[2]);

  std::vector<Tensor> lstm1_const_params;
  lstm1_const_params.push_back(const_params[3]);
  lstm1_const_params.push_back(const_params[4]);
  lstm1_const_params.push_back(const_params[5]);

  std::map<std::string, std::any> lstm0_attr;
  std::vector<std::string> lstm0_akeys = {"convfeat", "lstm_0__W", "lstm_0__R",
                                          "lstm_0__B", "lstm_0"};
  std::map<std::string, std::any> lstm1_attr;
  std::vector<std::string> lstm1_akeys = {"lstm_0", "lstm_1__W", "lstm_1__R",
                                          "lstm_1__B", "lstm_1"};

  if (attr_.count("scales") &&
      attr_.at("scales").type() == typeid(std::vector<float>)) {
    const auto &scales_vector =
        std::any_cast<const std::vector<float> &>(attr_.at("scales"));

    if (scales_vector.size() == 10) {
      for (int i = 0; i < 5; i++) {
        auto key0 = lstm0_akeys[i] + "_scale";
        lstm0_attr[key0] = scales_vector[i];
        auto key1 = lstm1_akeys[i] + "_scale";
        lstm1_attr[key1] = scales_vector[5 + i];
      }
    } else {
      std::cout << "Scales attribute does not have the expected number "
                   "of elements.Number of passed : "
                << scales_vector.size() << ",Expected : 10" << std::endl;
    }
  } else {
    std::cout << "Scales attribute not found or not of correct type."
              << std::endl;
  }

  if (attr_.count("zero_points") &&
      attr_.at("zero_points").type() == typeid(std::vector<int>)) {
    const auto &zero_points_vector =
        std::any_cast<const std::vector<int> &>(attr_.at("zero_points"));

    if (zero_points_vector.size() == 10) {
      for (int i = 0; i < 5; i++) {
        auto key0 = lstm0_akeys[i] + "_zero_point";
        lstm0_attr[key0] = (uint16_t)(zero_points_vector[i]);
        auto key1 = lstm1_akeys[i] + "_zero_point";
        lstm1_attr[key1] = (uint16_t)(zero_points_vector[5 + i]);
      }
    } else {
      std::cout << "ZPs attribute does not have the expected number "
                   "of elements.Number of passed : "
                << zero_points_vector.size() << ",Expected : 10" << std::endl;
    }
  } else {
    std::cout << "ZPs attribute not found or not of correct type." << std::endl;
  }

  int16_t seq_len;
  if (model_variant_ == "02") {
    seq_len = (int16_t)(modelNum_ / 8);
  } else {
    seq_len = (int16_t)(modelNum_ / 4);
    if (modelNum_ == 80) {
      seq_len = (int16_t)((modelNum_ / 4) + 4);
    }
  }
  lstm0_attr["seq_len"] = (int16_t)seq_len;
  lstm1_attr["seq_len"] = (int16_t)seq_len;
  lstm0_attr["model_variant"] = model_variant_;
  lstm1_attr["model_variant"] = model_variant_;

  LSTMUtil lstmUtil;

  lstmUtil.generateL1InitData(io, lstm0_wts_ptr, lstm0_const_params,
                              lstm0_attr);
  lstmUtil.generateL1InitData(io, lstm1_wts_ptr, lstm1_const_params,
                              lstm1_attr);
#if 0
  auto testDataFolder = OpInterface::get_dd_base_dir() +
                        "\\tests\\cpp\\unit_tests\\testDataMladf\\lstm_" +
                        std::to_string(modelNum_);

  auto fileName = testDataFolder + "\\" + "WTS" + ".txt";

  std::vector<WtT> weights;
  std::ifstream ifs(fileName);
  int val;
  while (ifs >> val)
    weights.push_back(val);
  ifs.close();

  //std::vector<WtT> weights = OpsFusion::read_bin_file<WtT>(
  //    fileName);

  int weightsSize =
      weightShape_[0] * weightShape_[1] * weightShape_[2] * sizeof(WtT);

  WtT* wts_gen = (WtT*)dest;
  int errcount = 0;
  for(int i=0; i < weights.size(); i++) {
    if(weights[i] != wts_gen[i]) {
      //std::cout << i << " : " << weights[i] << ", " << wts_gen[i] << std::endl;
      errcount++;
    }
  }
  if(errcount > 4096)
    std::cout << "Weights generation failed, errcount : " << errcount << std::endl;
  else
    std::cout << "Weights generation PASS, errcount : " << errcount << std::endl;

  auto gen_fileName = testDataFolder + "\\" + "wts_generated" + ".bin";
  //OpsFusion::write_bin_file(gen_fileName, (char*)dest, weightsSize);
  io.write(0, (void *)weights.data(), weightsSize);

#endif

  RYZENAI_LOG_TRACE("LSTM initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void lstm<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("lstm initialize_const_params ...");

  weightCopyTime_ = 0;
  weightFormatTime_ = 0;
  weightSyncTime_ = 0;

  /* Create input/output BOs */
  const size_t SCRATCH_BO_SIZE = 655360;
  const size_t CONST_BO_SIZE =
      (weightShape_[0] * weightShape_[1] * weightShape_[2] * weightDtypeSize_);
  const size_t IFM_BO_SIZE =
      (inputShape_[0] * inputShape_[1] * inputShape_[2] * ifmDtypeSize_);
  const size_t OFM_BO_SIZE =
      ((outputShape_[0]) * outputShape_[1] * (outputShape_[2]) * ofmDtypeSize_);
  RYZENAI_LOG_TRACE("lstm: IFM_BO_SIZE:" + std::to_string(IFM_BO_SIZE) +
                    " CONST_BO_SIZE:" + std::to_string(CONST_BO_SIZE) +
                    " OFM_BO_SIZE:" + std::to_string(OFM_BO_SIZE));
  constBo_ =
      xrt::bo(xrt_ctx_->get_device(), CONST_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
  ifmBo_ = xrt::bo(xrt_ctx_->get_device(), IFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));
  ofmBo_ = xrt::bo(xrt_ctx_->get_device(), OFM_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel().group_id(0));
  scratchBo_ =
      xrt::bo(xrt_ctx_->get_device(), SCRATCH_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));

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
  RYZENAI_LOG_TRACE("lstm initialize_const_params ... DONE");
}
/*
 * perform lstm c = a * w. w is stored in the object with initilize_weights
 * method.
 *
 * @param a pointer to activation matrix
 * @param a_shape tuple containing the shape of the activation matrix
 * @param c pointer to store the result of lstm
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void lstm<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                   std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("lstm execute ...");

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

  inputShape_[0] = input.at(0).shape.at(0);
  inputShape_[1] = input.at(0).shape.at(1);
  inputShape_[2] = input.at(0).shape.at(2);
  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  ifmBo_.write(input.at(0).data);
  auto a_copy_stop = GET_ELAPSED_TIME_NS();
  ifmCopyTime_ = static_cast<int64_t>(a_copy_stop - a_copy_start);

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  ifmBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();
  ifmSyncTime_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  xrt::bo param_bo;

  auto instr_bo_key = get_instr_key(txn_fname_prefix_, inputShape_[0],
                                    inputShape_[1], inputShape_[2]);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));

  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the lstm kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  // kernel call for lstm that supports transaction binary flow
  if (modelNum_ == 2560) {

    ryzenai::dynamic_dispatch::execute_kernel(
        kernel_, 2, instr_bo, instr_bo_words, constBo_, ifmBo_, scratchBo_,
        ofmBo_, 0, true, false);
  } else {

    ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                              instr_bo_words, constBo_, ifmBo_,
                                              ofmBo_, 0, 0, true, false);
  }
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);

  // sync output activation to host memory
  auto c_sync_start = GET_ELAPSED_TIME_NS();
  ofmBo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  auto c_sync_stop = GET_ELAPSED_TIME_NS();
  ofmSyncTime_ += static_cast<int64_t>(c_sync_stop - c_sync_start);

  auto c_copy_start = GET_ELAPSED_TIME_NS();
  ofmBo_.read(output.at(0).data);
  auto c_copy_stop = GET_ELAPSED_TIME_NS();
  ofmCopyTime_ = static_cast<int64_t>(c_copy_stop - c_copy_start);
  /*
    RYZENAI_LOG_INFO(
        std::to_string(matmul_id_) + " " + std::to_string(a_shape_[0]) + " " +
        std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) + " " +
        std::to_string(kernel_x_rows) + " " + std::to_string(kernel_x_shape_[1])
    + " " + std::to_string(kernel_y_shape_[1]) + " " + std::to_string(exec_end -
    exec_start) + " " + std::to_string(num_run_aie_) + " " +
    std::to_string(run_aie_time_) + " " + std::to_string(ifmCopyTime_) + " " +
    std::to_string(ifmCopyTime_) + " " + std::to_string(ofmCopyTime_) + " " +
    std::to_string(ofmCopyTime_) + " " + std::to_string((double)run_aie_time_ /
    num_run_aie_) + "\n");
  */
  RYZENAI_LOG_TRACE("lstm execute ... DONE");
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
template <typename InT, typename WtT, typename OutT>
void lstm<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> lstm<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  inputShape_[0] = input.at(0).shape[0];
  inputShape_[1] = input.at(0).shape[1];
  inputShape_[2] = input.at(0).shape[2];
  std::string txn_key =
      "lstm_" + txn_fname_prefix_ + "_" + std::to_string(inputShape_[0]) + "_" +
      std::to_string(inputShape_[1]) + "_" + std::to_string(inputShape_[2]);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> lstm<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  /*
    inputShape_[0] = input.at(0).shape[0];
    inputShape_[1] = input.at(0).shape[1];
    inputShape_[2] = input.at(0).shape[2];
    outputShape_[0] = input.at(7).shape[0];
    outputShape_[1] = input.at(7).shape[1];
    outputShape_[2] = input.at(7).shape[2];
  */
  size_t const_params_bo_size =
      (weightShape_[0] * weightShape_[1] * weightShape_[2] *
       weightDtypeSize_); // totalWtsSize;
  size_t ifm_bo_size =
      (inputShape_[0] * inputShape_[1] * inputShape_[2] * ifmDtypeSize_);
  size_t ofm_bo_size =
      ((outputShape_[0]) * outputShape_[1] * (outputShape_[2]) * ofmDtypeSize_);

  RYZENAI_LOG_TRACE("lstm: IFM_BO_SIZE:" + std::to_string(ifm_bo_size) +
                    " CONST_BO_SIZE:" + std::to_string(const_params_bo_size) +
                    " OFM_BO_SIZE:" + std::to_string(ofm_bo_size));

  size_t max_scratch_pad_size = 655360;

  std::vector<OpArgMap> arg_map;
  if (modelNum_ == 2560) {
    arg_map = {
        {OpArgMap::OpArgType::INPUT, 1, 0, 0, ifm_bo_size},
        {OpArgMap::OpArgType::CONST_INPUT, 0, 1, 0, const_params_bo_size},
        {OpArgMap::OpArgType::OUTPUT, 3, 7, 0, ofm_bo_size},
        {OpArgMap::OpArgType::SCRATCH_PAD, 2, 0, 0, max_scratch_pad_size}};
  } else {
    arg_map = {
        {OpArgMap::OpArgType::INPUT, 1, 0, 0, ifm_bo_size},
        {OpArgMap::OpArgType::CONST_INPUT, 0, 1, 0, const_params_bo_size},
        {OpArgMap::OpArgType::OUTPUT, 2, 7, 0, ofm_bo_size},
        {OpArgMap::OpArgType::SCRATCH_PAD, 2, 0, 0, max_scratch_pad_size}};
  }

  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("lstm Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
}

template <typename InT, typename WtT, typename OutT>
std::once_flag lstm<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t lstm<InT, WtT, OutT>::lstm_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag lstm<InT, WtT, OutT>::instr_reg_flag_;

template class lstm<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
