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

#include <ops/convForMatmulAdd/convForMatmulAdd.hpp>
#include <ops/op_interface.hpp>
#include <ops/ops_common/help_file.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>

#include "txn/txn_utils.hpp"
#include "txn_helper/txn_helper.hpp"
#include "utils/dpu_mdata.hpp"
#include <txn_container.hpp>

namespace ryzenai {
/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename WtT, typename OutT>
void convForMatmulAdd<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<conv_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;

  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key =
        "conv_" + get_instr_key(txn_fname_prefix_, mat.Z, mat.F, mat.K, mat.N);
    instructions.push_back(std::make_pair(key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
std::string convForMatmulAdd<InT, WtT, OutT>::get_instr_key(
    std::string prefix, int64_t zp, int64_t F, int64_t K, int64_t N) const {
  if (zp == NO_ZP) {
    return prefix + "_" + std::to_string(F) + "_" + std::to_string(K) + "_" +
           std::to_string(N);
  } else {
    return prefix + "_" + std::to_string(zp) + "_" + std::to_string(F) + "_" +
           std::to_string(K) + "_" + std::to_string(N);
  }
}

static std::string GetParamKey(std::string prefix, int64_t zp, int64_t K,
                               int64_t N, int64_t F0) {
  return prefix + "_" + std::to_string(zp) + "_" + std::to_string(F0) + "_" +
         std::to_string(K) + "_" + std::to_string(N);
}

/*
 * convForMatmulAdd class constructor
 *
 * @param kernel_x_shape tuple containing of M x K dimension base
 * convForMatmulAdd supported on IPU
 * @param kernel_y_shape tuple containing of K x N dimension base
 * convForMatmulAdd supported on IPU
 *
 */
template <typename InT, typename WtT, typename OutT>
convForMatmulAdd<InT, WtT, OutT>::convForMatmulAdd(
    const std::string &ifmDtype, const std::string &weightDtype,
    const std::string &ofmDtype, bool load_xrt,
    const std::map<std::string, std::any> &attr)
    : attr_(attr) {

  this->debug_lp = false;
  this->compute_lp = false;
  if (Utils::get_env_var("DEBUG_LP", "0") != "0") {
    this->debug_lp = true;
  }
  if (Utils::get_env_var("COMPUTE_LP", "0") != "0") {
    this->compute_lp = true;
  }

  /* Wts, qdq, qdq_params */
  this->SetNumConst(3);

  /* By default use txn binaries without zp */
  this->useTxnBinWithZp_ = false;

  txnbin_a_header = {{"uint16", "a16"}};
  txnbin_b_header = {{"uint8", "w8"}, {"uint16", "w16"}};
  txnbin_acc_header = {{"uint16", "c16"}};
  xclbin_a_header = {{"int16", "a16"}, {"int8", "a8"}};
  xclbin_b_header = {{"int8", "w8"}, {"uint16", "w16"}};
  xclbin_acc_header = {{"int16", "acc16"}, {"int8", "acc8"}};

  ifmDtype_ = ifmDtype;
  weightDtype_ = weightDtype;
  ofmDtype_ = ofmDtype;
  ifmDtypeSize_ = sizeof(InT);
  weightDtypeSize_ = sizeof(WtT);
  ofmDtypeSize_ = sizeof(OutT);

  conv_id_ = conv_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\ConvDwc.xclbin";

  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME));
  txn_fname_prefix_ = "conv_" + txnbin_a_header.at(ifmDtype_) +
                      txnbin_b_header.at(weightDtype_) +
                      txnbin_acc_header.at(ofmDtype_);

  /* Shapes for mswbjvw 320 */
  default_shapes_["conv_a16w16c16"] = std::vector<conv_shapes>{};
  default_shapes_["conv_a16w16c16"].emplace_back(32771, 1, 256, 548);

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

  /* Attribute Parsing */
  if (attr.count("group") &&
      attr.at("group").type() == typeid(std::vector<int>)) {
    const auto &group_vector =
        std::any_cast<const std::vector<int> &>(attr.at("group"));
    for (const auto &group_id : group_vector) {
      groupId_ = group_id;
    }
  } else {
    std::cout << "Group ID not found or not of correct type." << std::endl;
  }

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

  if (attr.count("weight_shape") &&
      attr.at("weight_shape").type() == typeid(std::vector<int>)) {
    const auto &weight_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("weight_shape"));

    if (weight_shape_vector.size() >= 2) {
      weightShape_[0] = weight_shape_vector[0];
      weightShape_[1] = weight_shape_vector[1];
    } else {
      std::cout << "Weight Shape attribute does not have enough elements."
                << std::endl;
    }
    RYZENAI_LOG_TRACE(
        "Conv: WeightShape: " + std::to_string(weight_shape_vector[0]) + ", " +
        std::to_string(weight_shape_vector[1]) + ", " +
        std::to_string(weight_shape_vector[2]) + ", " +
        std::to_string(weight_shape_vector[3]));
  } else {
    std::cout << "Weight Shape attribute not found or not of correct type."
              << std::endl;
  }

  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));

    if (input_shape_vector.size() == 3) {
      inputShape_[0] = input_shape_vector[2];
      inputShape_[1] = input_shape_vector[1];
      inputShape_[2] = input_shape_vector[0];
    } else {
      std::cout
          << "Input Shape attribute does not have the expected number of "
             "elements.Number of passed : input_shape_vector.size(), Expected:3"
          << std::endl;
    }
    RYZENAI_LOG_TRACE(
        "Conv: InputShape: " + std::to_string(input_shape_vector[0]) + ", " +
        std::to_string(input_shape_vector[1]) + ", " +
        std::to_string(input_shape_vector[2]));
  } else {
    std::cout << "Input Shape attribute not found or not of correct type."
              << std::endl;
  }

  if (attr.count("output_shape") &&
      attr.at("output_shape").type() == typeid(std::vector<int>)) {
    const auto &output_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("output_shape"));

    if (output_shape_vector.size() == 3) {
      outputShape_[0] = output_shape_vector[2];
      outputShape_[1] = output_shape_vector[1];
      outputShape_[2] = output_shape_vector[0];
    } else {
      std::cout << "Output Shape attribute does not have the expected number "
                   "of elements.Number of passed : input_shape_vector.size(), "
                   "Expected:3"
                << std::endl;
    }
    RYZENAI_LOG_TRACE(
        "Conv: OutputShape: " + std::to_string(output_shape_vector[0]) + ", " +
        std::to_string(output_shape_vector[1]) + ", " +
        std::to_string(output_shape_vector[2]));
  } else {
    std::cout << "Output Shape attribute not found or not of correct type."
              << std::endl;
  }

  if (attr.count("orig_output_shape") &&
      attr.at("orig_output_shape").type() == typeid(std::vector<int>)) {
    const auto &orig_output_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("orig_output_shape"));

    if (orig_output_shape_vector.size() == 3) {
      origOutputShape_[0] = orig_output_shape_vector[2];
      origOutputShape_[1] = orig_output_shape_vector[1];
      origOutputShape_[2] = orig_output_shape_vector[0];
    } else {
      std::cout
          << "Orig Output Shape attribute does not have the expected number "
             "of elements.Number of passed : input_shape_vector.size(), "
             "Expected:3"
          << std::endl;
    }
    RYZENAI_LOG_TRACE(
        "FC: OrigOutputShape: " + std::to_string(orig_output_shape_vector[0]) +
        ", " + std::to_string(orig_output_shape_vector[1]) + ", " +
        std::to_string(orig_output_shape_vector[2]));
  } else {
    std::cout << "Output Shape attribute not found or not of correct type."
              << std::endl;
    origOutputShape_[0] = outputShape_[0];
    origOutputShape_[1] = outputShape_[1];
    origOutputShape_[2] = outputShape_[2];
  }

  std::string model_variant;

  if (attr.end() != attr.find("model_variant")) {
    model_variant = std::any_cast<std::string>(attr.at("model_variant"));
  }

  if (model_variant.size()) {
    convData_ = "convData_" + model_variant + "_layer";
    std::stringstream ss(model_variant);
    std::vector<std::string> model_variant_tokens;
    std::string token;
    while (std::getline(ss, token, '_')) {
      model_variant_tokens.push_back(token);
    }
    model_variant_ = model_variant_tokens[0];
  } else if (attr.count("width") &&
             attr.at("width").type() == typeid(std::vector<int>)) {
    const auto &width_vector =
        std::any_cast<const std::vector<int> &>(attr.at("width"));
    for (const auto &width : width_vector) {
      convData_ = "convData" + std::to_string(width) + "_layer";
    }
  } else {
    convData_ = "convData_layer";
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

  kernelWeightShape_[0] = weightShape_[1];
  kernelWeightShape_[1] = weightShape_[0];

  kernelOutputShape_[0] = outputShape_[1];
  kernelOutputShape_[1] = outputShape_[0] / 8;
  kernelOutputShape_[2] = outputShape_[2];
  kernelOutputShape_[3] = 8;

  lp.resize(128, 0);
  /* Update the layer params binary */
  Transaction &txn = Transaction::getInstance();
  if (this->compute_lp) {
    updateLayerParams(lp, this->attr_);
  } else {
    std::string lp_key =
        GetParamKey(convData_, zp_, inputShape_[0], origOutputShape_[0], 1) +
        "_lp";
    std::string lp_binary = txn.get_txn_str(lp_key);
    txn.GetBinData(lp_binary, lp, false);
  }

  /* Comparison with lp_ref */
  // TODO: once the lp computation are finalized we need to remove the
  // comparison along with the lp bin files
  if (this->compute_lp && this->debug_lp) {
    std::vector<uint8_t> lp_ref(128, 0);
    std::string lp_key =
        GetParamKey(convData_, zp_, inputShape_[0], origOutputShape_[0], 1) +
        "_lp";
    std::string lp_binary = txn.get_txn_str(lp_key);
    txn.GetBinData(lp_binary, lp_ref, false);

    compare_vectors_and_print(lp, lp_ref);
  }

  foldWts_ = lp[19];

  std::call_once(logger_flag_, []() {
    std::string header = "conv_id (Mi0 Mi1 F0 F1 K N Mo0 Mo1) Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "IFM_copy_time(ns) IFM_sync_time(ns) "
                         "OFM_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[Conv] ID: " + std::to_string(conv_id_) + ", XCLBIN: " +
                    XCLBIN_FNAME + ", (a_dtype, b_dtype, c_dtype): (" +
                    ifmDtype_ + ", " + weightDtype_ + ", " + ofmDtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void convForMatmulAdd<InT, WtT, OutT>::WriteToFile(void *src, uint64_t length) {
  uint8_t *dataPtr = (uint8_t *)src;
  std::string testDataFolder = OpInterface::get_dd_base_dir() + "\\" + "tests" +
                               "\\" + "cpp" + "\\" + "unit_tests" + "\\" +
                               "testDataMladf" + "\\" + "GeneratedWeights";
  std::string fileName =
      testDataFolder + "\\" +
      GetParamKey("wtsGenerated", zp_, inputShape_[0], outputShape_[0], 1) +
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

/* Concat weight params for convA16W8 */
template <typename InT, typename WtT, typename OutT>
void convForMatmulAdd<InT, WtT, OutT>::dumpBinary(void *src, size_t length,
                                                  std::string &filePath) const {
  std::ofstream ofs(filePath, std::ios::binary);
  size_t chunk_size = 1024;
  char *ptr = (char *)src;
  for (int i = 0; i < length / chunk_size; ++i) {
    ofs.write((char *)src, 1024);
    ptr += chunk_size;
  }
  ofs.write(ptr, length % chunk_size);
}

/* Concat weight params for convA16W16 */
template <typename InT, typename WtT, typename OutT>
int64_t convForMatmulAdd<InT, WtT, OutT>::ConcatenateWeightParams(
    ConstBufferIO &io, const std::vector<WtsListType> &wts_list,
    const std::vector<std::vector<int32_t>> &qdq_list,
    /* uint8_t *lp,*/
    int ifm_depth, int cstride, int ksize_x, int ksize_y) {

  size_t concatenateWeightParamsLength = 0;
  size_t offset = 0;
  for (size_t it = 0; it < wts_list.size(); ++it) {
    // for (size_t it = 0; it < 1; ++it) {
    int id = 0;
    std::vector<WtT> w_vals;

    // dd lp
    std::vector<uint8_t> lpTemporary;
    lpTemporary.resize(128);
    memcpy(lpTemporary.data(), lp.data(), 128 * sizeof(uint8_t));
    lpTemporary[82] = lpTemporary[83] = 0;
    io.write(offset, lpTemporary.data(), 128 * sizeof(uint8_t));
    concatenateWeightParamsLength += 128 * sizeof(uint8_t);
    offset += 128 * sizeof(uint8_t);

    // qdq
    io.write(offset, (void *)qdq_list[it].data(),
             sizeof(int32_t) * qdq_list[it].size());
    concatenateWeightParamsLength += sizeof(int32_t) * qdq_list[it].size();
    offset += sizeof(int32_t) * qdq_list[it].size();

    int istride = std::min(ifm_depth, cstride);
    int ostride = 8;
    int Cout = static_cast<int>(wts_list[it].size());
    int Cin = static_cast<int>(wts_list[it][0].size());

    auto wtsListMember = wts_list[it];
    for (int o = 0; o < Cout; o += ostride) {
      for (int i = 0; i < Cin; i += istride) {
        for (int y = 0; y < ksize_y; ++y) {
          for (int x = 0; x < ksize_x; ++x) {
            for (int i_idx = i; i_idx < i + istride; ++i_idx) {
              for (int o_idx = 0; o_idx < 8; ++o_idx) {
                auto w_val = wts_list[it][o + o_idx][i_idx][y][x];
                w_vals.push_back(w_val);
              }
            }
          }
        }
      }
    }
    io.write(offset, w_vals.data(), sizeof(WtT) * w_vals.size());
    concatenateWeightParamsLength += sizeof(WtT) * w_vals.size();
    offset += sizeof(WtT) * w_vals.size();
  }
  return concatenateWeightParamsLength;
}

/* qdq header for convA16W16 */
template <typename InT, typename WtT, typename OutT>
std::vector<int32_t> convForMatmulAdd<InT, WtT, OutT>::qdq_header(
    int64_t *qdq, int32_t ofm_height, int32_t ofm_width,
    int32_t ofm_depth_start, int32_t ofm_depth_end) {
  int64_t *c0 = qdq;

  std::vector<int32_t> header;
  for (int32_t i = ofm_depth_start; i < ofm_depth_end; i++) {
    int64_t ci = c0[i];
    int64_t temp = static_cast<uint64_t>(ci) & 0x00000000FFFFFFFF;
    header.push_back(static_cast<int32_t>(temp));
    temp = (static_cast<uint64_t>(ci) >> 32) & 0x00000000FFFFFFFF;
    header.push_back(static_cast<int32_t>(temp));
  }

  return header;
}

/*
 * copy weight matrix into XRT BOs with padding and tiling
 *
 * this method copies the weight matrix into XRT BOs. This is re-used for
 * every convForMatmulAdd performed for this object with different activations.
 * weight matrix is padded, tiled and reformatted while copying to XRT BOs.
 * padding is done to align with kernel_y_shape each tile of the weight matrix
 * is of shape kernel_y_shape this method also reformats the matrix b/weight
 * matrix as required by AIE/IPU convForMatmulAdd implementation
 *
 * @param weights pointer to the weight matrix
 * @param w_shape tuple containing the shape of the weight matrix
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void convForMatmulAdd<InT, WtT, OutT>::initialize_const_params_conv(
    ConstBufferIO &io, const std::vector<Tensor> &const_params) {

  auto wtsIn = (WtT *)const_params.at(0).data;
  auto wtsInShape = const_params.at(0).shape;
  auto wtsInType = const_params.at(0).dtype;
  Wts2DListType wts(std::vector<std::vector<WtT>>(
      wtsInShape[0], std::vector<WtT>(wtsInShape[1])));

  for (int i = 0; i < wtsInShape[0]; ++i) {
    for (int j = 0; j < wtsInShape[1]; ++j) {
      wts[i][j] = wtsIn[(i * wtsInShape[1]) + j];
    }
  }

  int stride = 1 << lp[5];
  int fold_wts = static_cast<int>(lp[19]);
  int ifm_depth = static_cast<int>(wts.size());
  int ofm_depth = static_cast<int>(wts[0].size());
  int ifm_sv_depth = static_cast<int>(lp[2] * 8);
  int ofm_sv_depth = static_cast<int>(lp[3] * 8);
  int ofm_depth_padded = static_cast<int>(
      ofm_sv_depth * 2 * lp[67] * (model_variant_ == "02" ? lp[65] : 1));

  int wts_zp = lp[82] + (lp[83] << 8);
  WtsListType wt_pad(
      1, std::vector<std::vector<std::vector<WtT>>>(
             1, std::vector<std::vector<WtT>>(
                    ifm_depth, Wts1DListType(ofm_depth_padded, wts_zp))));
  for (int i = 0; i < ifm_depth; ++i) {
    for (int j = 0; j < ofm_depth; ++j) {
      wt_pad[0][0][i][j] = wts[i][j];
    }
  }

  WtsListType wts_new(
      ofm_depth_padded,
      std::vector<std::vector<std::vector<WtT>>>(
          ifm_depth, std::vector<std::vector<WtT>>(1, Wts1DListType(1, 0))));
  for (int i = 0; i < ofm_depth_padded; ++i) {
    for (int j = 0; j < ifm_depth; ++j) {
      for (int k = 0; k < 1; ++k) {
        for (int l = 0; l < 1; ++l) {
          wts_new[i][j][k][l] = wt_pad[k][l][j][i];
        }
      }
    }
  }

  int ksize_x = 1;
  int ksize_y = 1;

  int ofm_sv_width = (lp[0] - ksize_x + 1) / stride;
  int ofm_sv_height = (lp[1] - ksize_y + 1) / stride;

  int ofm_depth_iters = static_cast<int>(lp[67]);
  int num_adf_rows =
      static_cast<int>((ofm_depth_padded / ofm_sv_depth) / ofm_depth_iters);
  int ch_in_depth_split = static_cast<int>(lp[24]);
  int num_wt_streams = num_adf_rows;
  int cout_per_ch_iter = static_cast<int>(ofm_sv_depth * num_adf_rows);
  int cout_per_stream = static_cast<int>(cout_per_ch_iter / num_wt_streams);
  int ifm_depth_iters = lp[66];
  int dp = static_cast<int>(ifm_depth_iters * ifm_sv_depth / ch_in_depth_split);
  int cstride = 8;

  int64_t *qdq_data;
  std::vector<int64_t> qdq;

  Transaction &txn = Transaction::getInstance();
  if (this->compute_lp) {
    qdq_data = (int64_t *)const_params.at(1).data;
  } else {
    auto qdqSize = ofm_depth_padded;
    qdq.resize(qdqSize);
    std::string qdq_key =
        GetParamKey(convData_, zp_, inputShape_[0], origOutputShape_[0], 1) +
        "_qdq";
    std::string qdq_binary = txn.get_txn_str(qdq_key);
    txn.GetBinData(qdq_binary, qdq, false);
    qdq_data = qdq.data();
  }

  std::vector<WtsListType> wts_list;
  std::vector<std::vector<int32_t>> qdq_list;
  for (int32_t och_iter = 0; och_iter < ofm_depth_iters; och_iter++) {
    auto start_och = och_iter * cout_per_ch_iter;
    auto end_och = (och_iter + 1) * cout_per_ch_iter;
    auto wt_och_iter =
        WtsListType(wts_new.begin() + start_och, wts_new.begin() + end_och);

    for (int32_t kk = 0; kk < ch_in_depth_split; kk++) {
      for (int32_t wt_strms = 0; wt_strms < num_wt_streams; wt_strms++) {
        auto start_wt_strms = wt_strms * cout_per_stream;
        auto end_wt_strms = (wt_strms + 1) * cout_per_stream;
        auto wt_strm_data = WtsListType(wt_och_iter.begin() + start_wt_strms,
                                        wt_och_iter.begin() + end_wt_strms);

        int start = 0;
        int end = static_cast<int>(
            ceil(ifm_depth_iters / static_cast<double>(ch_in_depth_split)));

        for (int32_t ich_iter = start; ich_iter < end; ich_iter++) {
          qdq_list.push_back(qdq_header(
              (int64_t *)qdq_data, ofm_sv_height, ofm_sv_width,
              och_iter * cout_per_ch_iter + wt_strms * cout_per_stream,
              och_iter * cout_per_ch_iter + end_wt_strms));

          WtsListType sub_tensor;
          for (const auto &o : wt_strm_data) {
            std::vector<std::vector<std::vector<WtT>>> wts_o;
            for (int i = kk * dp + ich_iter * ifm_sv_depth;
                 i < kk * dp + (ich_iter + 1) * ifm_sv_depth &&
                 i < static_cast<int>(o.size());
                 i++) {
              wts_o.push_back(o[i]);
            }
            sub_tensor.push_back(wts_o);
          }
          wts_list.push_back(sub_tensor);
        }
      }
    }
  }

  size_t concatenateWeightParamsLength = 0;
  concatenateWeightParamsLength = ConcatenateWeightParams(
      io, wts_list, qdq_list, (int)wts_new[0].size(), cstride,
      (int)wts_new[0][0][0].size(), (int)wts_new[0][0].size());

  if (debug_ == true) {
    auto val = io.read(0, concatenateWeightParamsLength);
    WriteToFile(val.data(), concatenateWeightParamsLength);
  }
}

template <typename InT, typename WtT, typename OutT>
int64_t convForMatmulAdd<InT, WtT, OutT>::ConcatenateWeightParams_dwc(
    void *dest, const std::vector<WtsListType> &wts_list,
    const std::vector<std::vector<int32_t>> &qdq_list, int ifm_depth,
    int cstride, int ksize_x, int ksize_y) {

  size_t concatenateWeightParamsLength = 0;
  WtT *dstWeightBuffer = (WtT *)dest;
  for (size_t it = 0; it < wts_list.size(); ++it) {
    std::vector<WtT> w_vals;

    memcpy(dstWeightBuffer, lp.data(), 64 * sizeof(WtT));
    concatenateWeightParamsLength += 64 * sizeof(WtT);
    dstWeightBuffer += 64 * sizeof(WtT);

    int istride = std::min(ifm_depth, cstride);
    int ostride = 8;
    int Cout = static_cast<int>(wts_list[it].size());
    int Cin = static_cast<int>(wts_list[it][0].size());

    auto wtsListMember = wts_list[it];
    for (int o = 0; o < Cout; o += ostride) {
      for (int i = 0; i < Cin; i += istride) {
        for (int y = 0; y < ksize_y; ++y) {
          for (int x = 0; x < ksize_x; ++x) {
            for (int o_idx = 0; o_idx < 8; ++o_idx) {
              auto w_val = wts_list[it][o + o_idx][i][y][x];
              w_vals.push_back(w_val);
            }
          }
        }
      }
    }

    int zeroPadForAlignmentLength = (64 - (w_vals.size() % 64));

    memcpy(dstWeightBuffer, w_vals.data(), sizeof(WtT) * w_vals.size());
    concatenateWeightParamsLength +=
        sizeof(WtT) * (w_vals.size() + zeroPadForAlignmentLength);
    dstWeightBuffer +=
        sizeof(WtT) * (w_vals.size() + zeroPadForAlignmentLength);

    size_t qdqSizeInBytes = qdq_list[it].size() * sizeof(int32_t);
    zeroPadForAlignmentLength = (64 - (qdqSizeInBytes % 64));

    memcpy(dstWeightBuffer, qdq_list[it].data(),
           sizeof(int32_t) * qdq_list[it].size());
    concatenateWeightParamsLength +=
        sizeof(int32_t) * qdq_list[it].size() + zeroPadForAlignmentLength;
    dstWeightBuffer +=
        sizeof(int32_t) * qdq_list[it].size() + zeroPadForAlignmentLength;
  }
  return concatenateWeightParamsLength;
}

template <typename InT, typename WtT, typename OutT>
void convForMatmulAdd<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Conv initialize_const_params(ptr) ...");
  initialize_const_params_conv(io, const_params);

  RYZENAI_LOG_TRACE("Conv initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void convForMatmulAdd<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Conv initialize_const_params ...");

  /* Get buffer sizes required for this operators. We are not using input and
   * output tenosrs in get_buffer_req(). So calling with dummy tensors */
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

  RYZENAI_LOG_TRACE("Conv: IFM_BO_SIZE:" + std::to_string(IFM_BO_SIZE) +
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
/*
 * perform convForMatmulAdd c = a * w. w is stored in the object with
 * initilize_weights method.
 *
 * @param a pointer to activation matrix
 * @param a_shape tuple containing the shape of the activation matrix
 * @param c pointer to store the result of convForMatmulAdd
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void convForMatmulAdd<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
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
  /* kernel call for Conv that supports transaction binary flow. For single
   * convolution there can't be any time requirement of scratch pad buffer. So
   * in below executiion scratch pad is not used */
  run = kernel_(2, instr_bo, instr_bo_words,
                constBo_.address() + DDR_AIE_ADDR_OFFSET,
                ifmBo_.address() + DDR_AIE_ADDR_OFFSET,
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

/*
 * method to set debug flag
 *
 * @param debug bool value to enable disable debug feature. turned off by
 * default
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void convForMatmulAdd<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

/**
 * Helper function to read txn binary from file, embed zp in it (if rt_const_pad
 * is true) and return it
 * */
template <typename InT, typename WtT, typename OutT>
std::vector<uint8_t>
convForMatmulAdd<InT, WtT, OutT>::get_transaction_bin() const {
  std::string txn_key =
      "conv_" + txn_fname_prefix_ + "_" +
      (this->useTxnBinWithZp_ ? (std::to_string(zp_) + "_") : "") +
      std::to_string(1) + "_" + std::to_string(inputShape_[0]) + "_" +
      std::to_string(outputShape_[0]);
  std::cout << txn_key << std::endl;
  Transaction &txn = Transaction::getInstance();
  std::string txn_string = txn.get_txn_str(txn_key);
  std::istringstream txn_stream(txn_string, std::ios::binary);
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("Instruction fname : {}", txn_key));
  std::vector<uint8_t> txnData((std::istreambuf_iterator<char>(txn_stream)),
                               std::istreambuf_iterator<char>());

  if (!this->useTxnBinWithZp_) {
    // Runtime constant padding
    uint32_t zp = uint16_t(zp_);
    uint32_t pad_val = zp | (zp << 16);
    auto paddedTxnData = prepend_mtile_const_pad_txn(txnData, pad_val, 6, 2);
    if (this->debug_) {
      // Dump paddedTxnData
      std::string filePath =
          OpInterface::get_dd_base_dir() + "\\" + "tests" + "\\" + "cpp" +
          "\\" + "unit_tests" + "\\" + "testDataMladf" + "\\" +
          "GeneratedWeights" +
          GetParamKey("padded_conv", zp_, inputShape_[0], outputShape_[0], 1) +
          ".bin";
      if (!paddedTxnData.empty()) {
        dumpBinary(paddedTxnData.data(),
                   paddedTxnData.size() * sizeof(paddedTxnData[0]), filePath);
      }
    }
    return paddedTxnData;
  }
  return txnData;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
convForMatmulAdd<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return this->get_transaction_bin();
}

template <typename InT, typename WtT, typename OutT>
void convForMatmulAdd<InT, WtT, OutT>::set_params(const std::string &modelName,
                                                  bool useTxnBinWithZp) {
  this->useTxnBinWithZp_ = useTxnBinWithZp;
  std::string XCLBIN_FNAME;
  if (modelName == "m3uec") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\ConvDwcGap_Psi.xclbin";
  } else if ((modelName == "pst") || (modelName == "pss")) {
    if (zp_ == 699) { /* Temporary until xclbin is merged */
      XCLBIN_FNAME =
          OpInterface::get_dd_base_dir() +
          "\\xclbin\\stx\\tempXclbinFiles\\conv_699_3_512_512.xclbin";
    } else {
      XCLBIN_FNAME =
          OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\ConvPssPst.xclbin";
    }
  } else if ((modelName == "mswbjvw") || (modelName == "mswbjvw640") ||
             (modelName == "mswbjvw1280") || (modelName == "mswbjvw2560")) {
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() + "\\xclbin\\" + "stx" +
                   "\\4x2_pso2_320_model_lstm_a16w16_qdq.xclbin";
  }
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
#if 0
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
#endif
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> convForMatmulAdd<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  int totalWtsSize = 0;

  int ofm_depth = static_cast<int>(weightShape_[1]);
  int ifm_sv_depth = static_cast<int>(lp[2] * 8);
  int ofm_sv_depth = static_cast<int>(lp[3] * 8);
  int ofm_depth_padded = static_cast<int>(
      ofm_sv_depth * 2 * lp[67] * (model_variant_ == "02" ? lp[65] : 1));

  int kx = 1;
  int ky = 1;
  int ifm_depth_iters = lp[66];
  int ofm_depth_iters = static_cast<int>(lp[67]);
  int num_adf_rows =
      static_cast<int>((ofm_depth_padded / ofm_sv_depth) / ofm_depth_iters);
  totalWtsSize =
      (128 + ofm_sv_depth * 8 + kx * ky * ifm_sv_depth * ofm_sv_depth * 2) *
      ifm_depth_iters * ofm_depth_iters * num_adf_rows;
  size_t const_params_bo_size = totalWtsSize;
  size_t ifm_bo_size =
      (inputShape_[0] * inputShape_[1] * inputShape_[2] * ifmDtypeSize_);
  size_t ofm_bo_size =
      ((outputShape_[0]) * outputShape_[1] * (outputShape_[2]) * ofmDtypeSize_);
  RYZENAI_LOG_TRACE("Conv: IFM_BO_SIZE:" + std::to_string(ifm_bo_size) +
                    " CONST_BO_SIZE:" + std::to_string(const_params_bo_size) +
                    " OFM_BO_SIZE:" + std::to_string(ofm_bo_size));
  /* convForMatmulAdd operator is used in concate as well. Here we are assuming
   * that if each convForMatmulAdd layer's output is supposed to spill over in
   * DDR, than each layer needs ifm_bo_size + ofm_bo_size scratch buffer. */
  size_t scratch_bo_size = ifm_bo_size + ofm_bo_size;

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, ifm_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 0, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 4, 0, ofm_bo_size},
      {OpArgMap::OpArgType::SCRATCH_PAD, 3, 0, 0, scratch_bo_size}};
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("Conv Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
std::once_flag convForMatmulAdd<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t convForMatmulAdd<InT, WtT, OutT>::conv_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag convForMatmulAdd<InT, WtT, OutT>::instr_reg_flag_;

std::string to_hex_string(const std::vector<uint8_t> &data) {
  std::stringstream ss;
  ss << std::hex << std::setfill('0');

  for (uint8_t byte : data) {
    ss << std::setw(2) << (int)byte;
  }

  return ss.str();
}

void updateLayerParams(std::vector<uint8_t> &layer_params,
                       const std::map<std::string, std::any> &attrs) {

  // NOTE: an string object must be passed as value (not a const char *)
  auto graph_id = std::to_string(std::any_cast<int64_t>(attrs.at("graph_id")));
  if (graph_id != "320" && graph_id != "640" && graph_id != "1280" &&
      graph_id != "2560" && graph_id != "5120" && graph_id != "8000") {
    throw new std::logic_error(
        "Unsupported graph_id/network passed to updateLayerParams");
  }

  auto layer_dims = fc_layer_dims[graph_id];
  auto srs_shifts = mswbjvw_srs_shifts[graph_id];

  bool alpha_scaled = std::round(
      FCLayerDims::ALPHA * std::pow(2, FCLayerDims::CONV2D_LRELU_SHIFT_ALPHA));

  int lrelu_alpha_scaled = 0;
  int shift_lrelu_alpha = 0;
  int act = FCLayerDims::ACT;

  // Adjust lrelu_alpha_scaled based on act value
  if (act >= 2) {
    lrelu_alpha_scaled = alpha_scaled ? 1 : 0;
  }

  std::vector<int> filt_dim = {
      FCLayerDims::KSIZE_X,          FCLayerDims::KSIZE_Y,
      FCLayerDims::OFM_SV_DEPTH,     FCLayerDims::STRIDE,
      FCLayerDims::MAXPOOL_2D_KSIZE, FCLayerDims::MAXPOOL_2D_STRIDE};
  auto stride_width = filt_dim[3];

  // Extract parameters from the map
  std::string dtype_in = "uint16";
  auto kernel_shape = std::any_cast<std::vector<int>>(attrs.at("weight_shape"));
  auto input_shape = std::any_cast<std::vector<int>>(attrs.at("input_shape"));
  auto output_shape = std::any_cast<std::vector<int>>(attrs.at("output_shape"));
  uint16_t wzp =
      (uint16_t)std::any_cast<std::vector<int>>(attrs.at("wts_zp")).at(0);
  int32_t c1 = std::any_cast<std::vector<int>>(attrs.at("c1")).at(0);
  int32_t c2 = std::any_cast<std::vector<int>>(attrs.at("c2")).at(0);
  auto kernel_height = FCLayerDims::KERNEL_H;
  auto kernel_width = FCLayerDims::KERNEL_W;
  // int ofm_depth = output_shape[2];
  // TODO: check why is this required to be constant;
  int ofm_depth = 576;
  // int kernel_width = std::any_cast<int>(attrs.at("kernel_width"));
  // int ofm_depth = std::any_cast<int>(attrs.at("ofm_depth"));
  // int kernel_height = std::any_cast<int>(attrs.at("kernel_height"));

  int width_gran = 8 / stride_width;
  int height_gran = FCLayerDims::HEIGHT_GRAN;
  int ifm_depth_gran = FCLayerDims::IFM_DEPTH_GRAN;
  int ofm_depth_gran = FCLayerDims::OFM_DEPTH_GRAN;
  int ifm_bytes = (dtype_in == "uint16") ? 2 : 1; // Assuming uint16 input

  // Calculate num_chout_iters using local variables
  int num_chout_iters;
  if (FCLayerDims::DWC_CH_SPLIT == 0) {
    num_chout_iters = (int)std::ceil(ofm_depth / FCLayerDims::OFM_SV_DEPTH) /
                      FCLayerDims::NUM_ADF_COLS;
  } else {
    num_chout_iters = (int)std::ceil(ofm_depth / FCLayerDims::OFM_SV_DEPTH) /
                      FCLayerDims::NUM_ADF_ROWS;
  }

  std::vector<int> ifm_sv_dim = {
      layer_dims.ifm_sv_height, layer_dims.ifm_sv_width,
      layer_dims.ifm_sv_depth, FCLayerDims::CONV_2D_OFM_PAD};
  int ofm_sv_height =
      (int)std::ceil((ifm_sv_dim[0] - filt_dim[0] + 1) / (double)filt_dim[3]);
  int ofm_sv_width =
      (int)std::ceil((ifm_sv_dim[1] - filt_dim[1] + 1) / (double)filt_dim[3]);
  int ofm_height = (int)std::ceil((layer_dims.ifm_height - filt_dim[0] + 1) /
                                  (double)filt_dim[3]);
  int ofm_width = (int)std::ceil((layer_dims.ifm_width - filt_dim[1] + 1) /
                                 (double)filt_dim[3]);

  int ifm_iter_w = (int)std::ceil(
      (float)ofm_width / (ofm_sv_width - FCLayerDims::OFM_SV_OVERLAP_WIDTH));
  int ifm_iter_h = (int)std::ceil(
      (float)ofm_height / (ofm_sv_height - FCLayerDims::OFM_SV_OVERLAP_HEIGHT));

  auto temp_iters = ifm_iter_w;
  if (temp_iters == 1) {
    ifm_iter_w = 1;
  } else {
    ifm_iter_w =
        (int)std::ceil((double)ifm_iter_w / FCLayerDims::NUM_ADF_COLS) *
        FCLayerDims::NUM_ADF_COLS;
  }

  int ifm_x_iter = int(ifm_iter_w * FCLayerDims::HEIGHT_UNROLL +
                       (1 - FCLayerDims::HEIGHT_UNROLL) * (double)ifm_iter_w /
                           FCLayerDims::NUM_ADF_COLS);
  int ifm_y_iter = int(ifm_iter_h * (1 - FCLayerDims::HEIGHT_UNROLL) +
                       FCLayerDims::HEIGHT_UNROLL * (double)ifm_iter_h /
                           FCLayerDims::NUM_ADF_COLS);

  // Calculate ifm_depth_iter and width_iter using local variables
  int ifm_depth_iter =
      (int)std::ceil(layer_dims.ifm_depth / (double)layer_dims.ifm_sv_depth);
  int width_iter = (FCLayerDims::NO_X_ITER == 0) ? ifm_x_iter : 1;

  // Adjust width_iter based on hardcoded value for mswbjvw_8000
  if (ofm_width == 2000) {
    width_iter = (int)((double)ofm_width / 80);
  }

  int height_iter = ifm_y_iter;
  int depth_scale_fac = 8;
  int ifm_sign = dtype_in == "int16" || dtype_in == "int8";

  layer_params[0] = layer_dims.ifm_sv_width;
  layer_params[1] = layer_dims.ifm_sv_height;
  layer_params[2] = layer_dims.ifm_sv_depth / depth_scale_fac;
  layer_params[3] = (uint8_t)(FCLayerDims::OFM_SV_DEPTH / depth_scale_fac);
  layer_params[24] = FCLayerDims::SUPER_ITER_0;
  layer_params[25] = FCLayerDims::NUM_ADF_ROWS;
  layer_params[26] = FCLayerDims::CONV_2D_OFM_PAD;
  if (act == 0 || act == 1) {
    layer_params[27] = act;
  }
  // layer_params[28] = FCLayerDims::CONV_MODE;
  layer_params[30] = ifm_sign * 16 + FCLayerDims::ELTADD_IFM2_SIGN;

  size_t offset = 64;
  // Assign values to layer_params
  layer_params[offset + 0] = width_iter;
  layer_params[offset + 1] = height_iter;
  layer_params[offset + 2] = ifm_depth_iter;
  layer_params[offset + 3] = num_chout_iters;
  layer_params[offset + 4] = kernel_width;
  layer_params[offset + 5] = kernel_height;
  layer_params[offset + 6] =
      int(std::ceil(layer_dims.ifm_sv_depth / float(ifm_depth_gran)));
  layer_params[offset + 7] = stride_width;
  layer_params[offset + 8] = FCLayerDims::BATCH_SIZE;

  layer_params[offset + 9] = int(
      std::ceil(FCLayerDims::BATCH_SIZE * ofm_sv_width / float(width_gran)));
  layer_params[offset + 10] =
      int(std::ceil(ofm_sv_height / float(height_gran)));
  layer_params[offset + 11] =
      int(std::ceil(FCLayerDims::OFM_SV_DEPTH / float(ofm_depth_gran)));

  // Calculate inner and outer granularity
  int inner_g = int(layer_params[offset + 4] * layer_params[offset + 5] *
                    layer_params[offset + 6]);
  int outer_g = int(layer_params[offset + 9] * layer_params[offset + 10] *
                    layer_params[offset + 11]);
  layer_params[offset + 12] = inner_g & 0xFF;
  layer_params[offset + 13] = (inner_g >> 8) & 0xFF;
  layer_params[offset + 14] = outer_g & 0xFF;
  layer_params[offset + 15] = (outer_g >> 8) & 0xFF;

  // Assign SRS shifts
  layer_params[offset + 16] = srs_shifts[0];
  layer_params[offset + 17] = srs_shifts[1];

  // Calculate step sizes using local variables
  int step_kx = ifm_depth_gran * ifm_bytes;
  int step_ky = int(std::ceil((ofm_sv_width * stride_width + kernel_width - 1) *
                              ifm_depth_gran * ifm_bytes / 64.0) *
                    64.0 * layer_dims.ifm_sv_depth / ifm_depth_gran);
  int step_Ci = int(std::ceil((ofm_sv_width * stride_width + kernel_width - 1) *
                              ifm_depth_gran * ifm_bytes / 64.0) *
                    64.0);
  int step_Xi = step_kx;
  int step_Yi = step_ky;
  int step_Xo = ofm_depth_gran * ifm_bytes;
  int step_Yo = ofm_sv_width * FCLayerDims::OFM_SV_DEPTH * ifm_bytes;
  int step_Co = FCLayerDims::OFM_SV_DEPTH * ofm_depth_gran * ifm_bytes;

  layer_params[offset + 20] = step_kx & 0xFF;
  layer_params[offset + 21] = (step_kx >> 8) & 0xFF;
  layer_params[offset + 22] = step_ky & 0xFF;
  layer_params[offset + 23] = (step_ky >> 8) & 0xFF;
  layer_params[offset + 24] = step_Ci & 0xFF;
  layer_params[offset + 25] = (step_Ci >> 8) & 0xFF;
  layer_params[offset + 26] = step_Xi & 0xFF;
  layer_params[offset + 27] = (step_Xi >> 8) & 0xFF;
  layer_params[offset + 28] = step_Yi & 0xFF;
  layer_params[offset + 29] = (step_Yi >> 8) & 0xFF;
  layer_params[offset + 30] = step_Xo & 0xFF;
  layer_params[offset + 31] = (step_Xo >> 8) & 0xFF;
  layer_params[offset + 32] = step_Yo & 0xFF;
  layer_params[offset + 33] = (step_Yo >> 8) & 0xFF;
  layer_params[offset + 34] = step_Co & 0xFF;
  layer_params[offset + 35] = (step_Co >> 8) & 0xFF;

  // Assign zero_init, sign_N, sign_O, sign_W, sign_A, skip_casc_in,
  // skip_casc_out, norm_ch_g
  int zero_init = 1;
  int sign_N = 0;
  int sign_O = 0;
  int sign_W = 0;
  int sign_A = 0;
  int skip_casc_in = 0;
  int skip_casc_out = 0;
  int norm_ch_g = 0;
  layer_params[offset + 40] =
      (zero_init & 0x01) + ((sign_N << 1) & 0x02) + ((sign_O << 2) & 0x04) +
      ((skip_casc_in << 6) & 0x40) + ((skip_casc_out << 7) & 0x80);
  layer_params[offset + 41] = (sign_W & 0x01) + ((sign_A << 1) & 0x02);
  layer_params[offset + 42] = 0;
  layer_params[offset + 43] = norm_ch_g;

  // Extract and assign qdq coefficients
  // int c1 = qdq_params.at(1);
  // int c2 = qdq_params.at(2);
  layer_params[offset + 44] = c1 & 0xFF;
  layer_params[offset + 45] = (c1 >> 8) & 0xFF;
  layer_params[offset + 46] = (c1 >> 16) & 0xFF;
  layer_params[offset + 47] = (c1 >> 24) & 0xFF;

  layer_params[offset + 48] = c2 & 0xFF;
  layer_params[offset + 49] = (c2 >> 8) & 0xFF;
  layer_params[offset + 50] = (c2 >> 16) & 0xFF;
  layer_params[offset + 51] = (c2 >> 24) & 0xFF;

  // Assign the last SRS shift
  // layer_params[offset + 53] =
  // std::any_cast<std::vector<int>>(params.at("srs_shifts"))[1];
  layer_params[offset + 53] = srs_shifts[1];

  layer_params[82] = wzp & 0xFF;
  layer_params[83] = (wzp >> 8) & 0xFF;
}

void compare_vectors_and_print(const std::vector<uint8_t> &vec1,
                               const std::vector<uint8_t> &vec2) {
  // Ensure vectors have equal sizes
  if (vec1.size() != vec2.size()) {
    std::cerr << "Error: Vectors must have the same size." << std::endl;
    return;
  }

  const std::string GREEN = "\033[92m";
  const std::string RED = "\033[91m";
  const std::string RESET = "\033[0m";

  for (std::size_t i = 0; i < vec1.size(); ++i) {
    std::string equal_str = (vec1[i] == vec2[i]) ? GREEN + "Equal" + RESET
                                                 : RED + "Not Equal" + RESET;
    std::cout << std::setw(5) << i;
    std::cout << "  " << std::setw(3) << (int)vec1[i] << "       "
              << std::setw(3) << (int)vec2[i] << "                " << equal_str
              << std::endl;
  }
}

template class convForMatmulAdd<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
