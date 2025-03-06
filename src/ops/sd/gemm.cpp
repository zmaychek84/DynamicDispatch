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

#include "sd_helper.hpp"
#include <ops/op_interface.hpp>
#include <ops/ops_common/help_file.hpp>
#include <ops/sd/gemm.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>

#include "txn/txn_utils.hpp"
#include "txn_helper/txn_helper.hpp"
#include "utils/dpu_mdata.hpp"

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

using std::vector;

namespace ryzenai {

namespace sd {

template <typename InT, typename WtT, typename BiasT, typename OutT>
void gemm<InT, WtT, BiasT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  auto supported_shapes = default_shapes_.find(txn_fname_prefix_)->second;
  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key = get_key(txn_fname_prefix_, mat);
    instructions.push_back(std::make_pair(key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
std::string
gemm<InT, WtT, BiasT, OutT>::get_key(std::string prefix,
                                     const std::vector<size_t> &mat) const {
  std::string out_str = prefix;
  for (size_t i = 0; i < mat.size(); i++) {
    out_str += "_" + std::to_string(mat[i]);
  }
  return out_str;
}

// gemm class constructor
template <typename InT, typename WtT, typename BiasT, typename OutT>
gemm<InT, WtT, BiasT, OutT>::gemm(const std::string &ifm_dtype,
                                  const std::string &weight_dtype,
                                  const std::string &bias_dtype,
                                  const std::string &out_dtype, bool load_xrt,
                                  const std::map<std::string, std::any> &attr) {
  if (Utils::get_env_var("DEBUG_SD", "0") != "0") {
    this->debug_ = true;
  }
  txnbin_a_header = {{"bfloat16", "a16bf"}};
  txnbin_b_header = {{"bfp16ebs8", "w16bfp"}, {"float32", "w16bfp"}};
  txnbin_acc_header = {{"bfloat16", "acc16bf"}};

  ifmDtype_ = ifm_dtype;
  weightDtype_ = weight_dtype;
  biasDtype_ = bias_dtype;
  ofmDtype_ = out_dtype;
  ifmDtypeSize_ = sizeof(InT);
  weightDtypeSize_ = sizeof(WtT);
  ofmDtypeSize_ = sizeof(OutT);
  biasDtypeSize_ = sizeof(BiasT);
  gemm_id_ = gemm_count++;
  txn_fname_prefix_ = sd_gemm_key_ + txnbin_a_header.at(ifmDtype_) +
                      txnbin_b_header.at(weightDtype_) +
                      txnbin_acc_header.at(ofmDtype_);

  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1024, 2560, 640});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1024, 640, 5120});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1024, 640, 640});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 256, 1280, 10240});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 256, 1280, 1280});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 256, 5120, 1280});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 4096, 1280, 320});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 4096, 320, 2560});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 4096, 320, 320});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 64, 1280, 10240});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 64, 1280, 1280});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 64, 5120, 1280});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 77, 768, 1280});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 77, 768, 320});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 77, 768, 640});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1, 320, 1280});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1, 1280, 320});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1, 1280, 640});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1, 1280, 1280});
  // default_shapes_[txn_fname_prefix_].emplace_back(
  //     std::vector<size_t>{1, 2, 2048, 1536});
  // default_shapes_[txn_fname_prefix_].emplace_back(
  //     std::vector<size_t>{1, 2, 256, 1536});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1, 2048, 1536});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1, 256, 1536});

  // from sd3 mmdit 512 layer1
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1024, 1536, 1536});
  // from sd3 mmdit 512 layer2
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1024, 1536, 6144});
  // from sd3 mmdit 512 layer3
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1024, 1536, 64});
  // from sd3 mmdit 512 layer4
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1024, 6144, 1536});

  // from sd3 mmdit 512 and 1024
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1, 1536, 1536});

  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 154, 1536, 1536});

  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 154, 1536, 6144});

  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 154, 4096, 1536});

  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 154, 6144, 1536});

  // from sd3 mmdit 1024 layer1
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 4096, 1536, 1536});
  // from sd3 mmdit 1024 layer2
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 4096, 1536, 6144});
  // from sd3 mmdit 1024 layer3
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 4096, 1536, 64});
  // from sd3 mmdit 1024 layer4
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 4096, 6144, 1536});
  // from sd3 vae decoder 1024 layer1
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{1, 16384, 512, 512});

  // from matmul_add_to_matmul pass
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 64, 1280, 5120});

  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 256, 1280, 5120});

  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1024, 640, 2560});

  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 4096, 320, 1280});

  // vae
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{1, 4096, 512, 512});

  // bf16-160 shapes
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 160, 1536, 1536});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 160, 1536, 6144});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 160, 4096, 1536});
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 160, 6144, 1536});

  if (load_xrt) {
    XCLBIN_FNAME_ =
        OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\SDGemm.xclbin";
    RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME_));
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
    if (weight_shape_vector.size() == 2) {
      weightShape_[0] = weight_shape_vector[0]; // K
      weightShape_[1] = weight_shape_vector[1]; // N
      K_ = weight_shape_vector[0];
      N_ = weight_shape_vector[1];

    } else {
      RYZENAI_LOG_INFO(
          "Weight Shape attribute does not have the expected number of "
          "elements.Number of passed : " +
          std::to_string(weight_shape_vector.size()) + ", Expected:2");
    }
    RYZENAI_LOG_TRACE(
        "Gemm: WeightShape: " + std::to_string(weight_shape_vector[0]) + ", " +
        std::to_string(weight_shape_vector[1]));
  } else {
    RYZENAI_LOG_INFO(
        "Weight Shape attribute not found or not of correct type.");
  }

  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));
    if (input_shape_vector.size() > 1) {
      inputShape_.assign(input_shape_vector.begin(), input_shape_vector.end());
    } else {
      RYZENAI_LOG_INFO(
          "Input Shape attribute does not have the expected number of "
          "elements.Number of passed : " +
          std::to_string(input_shape_vector.size()) + ", Expected: > 1");
    }
  } else {
    RYZENAI_LOG_INFO("Input Shape attribute not found or not of correct type.");
  }
  curr_txn_shape_.insert(curr_txn_shape_.end(), inputShape_.begin(),
                         inputShape_.end());
  curr_txn_shape_.push_back(N_);

  auto super_kernel_params = get_super_kernel_params();
  auto super_kernel_size = super_kernel_params.size();
  DD_ASSERT(
      super_kernel_size == 24,
      OpsFusion::dd_format("sdgemm load {} bytes lp bin not equal to 24 bytes",
                           super_kernel_size));
  auto lp_data_ptr = reinterpret_cast<uint32_t *>(super_kernel_params.data());
  // for wts shuffle
  sv_k_ = int(lp_data_ptr[4]);
  sv_n_ = int(lp_data_ptr[5]);

  if (attr.count("output_shape") &&
      attr.at("output_shape").type() == typeid(std::vector<int>)) {
    const auto &output_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("output_shape"));

    if (output_shape_vector.size() > 1) {
      outputShape_.assign(output_shape_vector.begin(),
                          output_shape_vector.end());
    } else {
      RYZENAI_LOG_INFO(
          "Input Shape attribute does not have the expected number of "
          "elements.Number of passed : " +
          std::to_string(output_shape_vector.size()) + ", Expected: > 1");
    }
  } else {
    RYZENAI_LOG_INFO(
        "Output Shape attribute not found or not of correct type.");
  }

  if (attr.count("bias_enable") &&
      attr.at("bias_enable").type() == typeid(bool)) {
    bias_en_ = std::any_cast<const bool &>(attr.at("bias_enable"));
    RYZENAI_LOG_TRACE("Gemm: BiasEn: " +
                      std::string(bias_en_ ? "true" : "false"));
  } else if (attr.count("bias_enable") &&
             attr.at("bias_enable").type() ==
                 typeid(std::vector<int>)) { // dd helper will convert bool type
                                             // into vector of int while
                                             // generating meta json
    bias_en_ =
        std::any_cast<const std::vector<int> &>(attr.at("bias_enable"))[0];
    RYZENAI_LOG_TRACE("Gemm: BiasEn: " +
                      std::string(bias_en_ ? "true" : "false"));
  } else {
    RYZENAI_LOG_INFO("Bias Enable attribute not found or not of correct type.");
  }

  CONST_BO_SIZE_ = get_const_bo_size(sv_k_, sv_n_);
  IFM_BO_SIZE_ = std::accumulate(inputShape_.begin(), inputShape_.end(), 1LL,
                                 [](int64_t acc, int64_t val) {
                                   return size_t(acc) * size_t(val);
                                 }) *
                 sizeof(InT);
  OFM_BO_SIZE_ = std::accumulate(outputShape_.begin(), outputShape_.end(), 1LL,
                                 [](int64_t acc, int64_t val) {
                                   return size_t(acc) * size_t(val);
                                 }) *
                 sizeof(OutT);
  std::call_once(logger_flag_, []() {
    std::string header =
        "sd_gemm_id | Execute time | total time | Avg_time_per_aie_run\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[SD_GEMM] ID: " + std::to_string(gemm_id_) +
                    ", (a_dtype, b_dtype, c_dtype): (" + ifmDtype_ + ", " +
                    weightDtype_ + ", " + ofmDtype_ + ")");
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
void gemm<InT, WtT, BiasT, OutT>::set_params(const std::string &xclbin,
                                             const std::string &pdi_name) {
  if (!xclbin.empty()) {
    XCLBIN_FNAME_ = OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\" + xclbin;
  }

  pdi_name_ = pdi_name;
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
size_t gemm<InT, WtT, BiasT, OutT>::get_const_bo_size(int sv_k,
                                                      int sv_n) const {

  int64_t iter_k = K_ / sv_k;
  int64_t iter_n = N_ / sv_n;
  int64_t bias_size = 0;
  if (bias_en_) {
    // GEMM  kernel expects bias in  the  format of  bf16
    bias_size = iter_k * iter_n * sv_n * sizeof(uint16_t);
  }
  size_t total_size = K_ * N_ / 8 * 9 + bias_size;
  return total_size;
}
template <typename InT, typename WtT, typename BiasT, typename OutT>
std::vector<uint8_t>
gemm<InT, WtT, BiasT, OutT>::shuffle_wts_bfp16(float *wts, float *bias) {
  float *b_ptr = nullptr;
  bool wts_transpose = true;
  if (bias_en_) {
    b_ptr = bias;
  }
  std::vector<uint8_t> w_b_vals;
  ryzenai::sd_helper::shuffle_gemm_wts(w_b_vals, wts, int(K_), int(N_),
                                       wts_transpose, b_ptr, sv_k_, sv_n_);
  return w_b_vals;
}
template <typename InT, typename WtT, typename BiasT, typename OutT>
void gemm<InT, WtT, BiasT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("gemm initialize_const_params ...");

  if (weightDtype_ == "float32") {
    auto weights = (float *)const_params.at(0).data;

    auto bias = (float *)const_params.at(1).data;

    std::vector<uint8_t> w_b_vals = shuffle_wts_bfp16(weights, bias);

    io.write(0, w_b_vals.data(), w_b_vals.size() * sizeof(uint8_t));
  } else if (weightDtype_ == "bfp16ebs8") {
    io.write(0, const_params.at(0).data, CONST_BO_SIZE_);
  }
  RYZENAI_LOG_TRACE("Gemm initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
void gemm<InT, WtT, BiasT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  std::vector<Tensor> input;
  std::vector<Tensor> output;

  RYZENAI_LOG_TRACE("SD_Gemm: IFM_BO_SIZE:" + std::to_string(IFM_BO_SIZE_) +
                    " CONST_BO_SIZE:" + std::to_string(CONST_BO_SIZE_) +
                    " OFM_BO_SIZE:" + std::to_string(OFM_BO_SIZE_));
  constBo_ =
      xrt::bo(xrt_ctx_->get_device(), CONST_BO_SIZE_, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel(pdi_name_).group_id(0));
  ifmBo_ = xrt::bo(xrt_ctx_->get_device(), IFM_BO_SIZE_, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel(pdi_name_).group_id(0));
  ofmBo_ = xrt::bo(xrt_ctx_->get_device(), OFM_BO_SIZE_, XRT_BO_FLAGS_HOST_ONLY,
                   xrt_ctx_->get_kernel(pdi_name_).group_id(0));

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
  RYZENAI_LOG_TRACE("gemm initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
void gemm<InT, WtT, BiasT, OutT>::execute(std::vector<Tensor> &input,
                                          std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("gemm execute ...");

  ifmBo_.write(input.at(0).data);
  ifmBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto txnData = this->get_transaction_bin();

  auto instr_bo_key = get_key(txn_fname_prefix_, curr_txn_shape_);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  size_t instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel(pdi_name_);

  auto run_aie_start = GET_ELAPSED_TIME_NS();
  // TODO: figure out the Bo order

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, ifmBo_, constBo_,
                                            ofmBo_, 0, 0, true, false);
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);
  /* sync output activation to host memory */
  ofmBo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  ofmBo_.read(output.at(0).data);

  RYZENAI_LOG_INFO(std::to_string(gemm_id_) + " " +
                   std::to_string(num_run_aie_) + " " +
                   std::to_string(run_aie_time_) + " " +
                   std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
void gemm<InT, WtT, BiasT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
const std::vector<uint8_t>
gemm<InT, WtT, BiasT, OutT>::get_transaction_bin() const {
  std::string txn_key = get_key(txn_fname_prefix_, curr_txn_shape_);
  Transaction &txn = Transaction::getInstance();
  std::vector<uint8_t> txnData = txn.get_txn_bvec(txn_key);
  return txnData;
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
const std::vector<uint8_t> gemm<InT, WtT, BiasT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return this->get_transaction_bin();
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
std::vector<OpArgMap> gemm<InT, WtT, BiasT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  RYZENAI_LOG_TRACE("SDGEMM: IFM_BO_SIZE:" + std::to_string(IFM_BO_SIZE_) +
                    " CONST_BO_SIZE:" + std::to_string(CONST_BO_SIZE_) +
                    " OFM_BO_SIZE:" + std::to_string(OFM_BO_SIZE_));
  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, IFM_BO_SIZE_},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, CONST_BO_SIZE_},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, OFM_BO_SIZE_}};

  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("SD_Gemm Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

// param bin container holding 24 bytes, each represented as a uint32 data type.
// The bytes correspond to the parameters: M, K, N, L1_M, L1_K, L1_N.
template <typename InT, typename WtT, typename BiasT, typename OutT>
const std::vector<uint8_t>
gemm<InT, WtT, BiasT, OutT>::get_super_kernel_params() const {
  auto param_key = get_key(txn_fname_prefix_, curr_txn_shape_) + "_param";
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(param_key);
}

template <typename InT, typename WtT, typename BiasT, typename OutT>
std::once_flag gemm<InT, WtT, BiasT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename BiasT, typename OutT>
uint64_t gemm<InT, WtT, BiasT, OutT>::gemm_count = 0;

template <typename InT, typename WtT, typename BiasT, typename OutT>
std::once_flag gemm<InT, WtT, BiasT, OutT>::instr_reg_flag_;

template class gemm<std::uint16_t, std::uint8_t, std::uint16_t, std::uint16_t>;
template class gemm<std::uint16_t, float, float, std::uint16_t>;

} // namespace sd

} // namespace ryzenai
