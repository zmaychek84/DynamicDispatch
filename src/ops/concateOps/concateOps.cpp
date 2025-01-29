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

#include "utils/dpu_mdata.hpp"
#include <nlohmann/json.hpp>
#include <ops/concateOps/concateOps.hpp>
#include <ops/conv/conv.hpp>
#include <ops/convForMatmulAdd/convForMatmulAdd.hpp>
#include <ops/lstm/lstm.hpp>
#include <ops/maxpool/maxpool.hpp>
#include <ops/op_interface.hpp>
#include <ops/ops_common/help_file.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/tfuncs.hpp>

using json = nlohmann::json;

std::any json_to_any(const json &j);

std::map<std::string, std::any> json_to_map(const json &j) {
  std::map<std::string, std::any> result;

  for (auto it = j.begin(); it != j.end(); ++it) {
    result[it.key()] = json_to_any(it.value());
  }

  return result;
}

std::any json_to_any(const json &j) {
  if (j.is_object()) {
    return json_to_map(j);
  } else if (j.is_array()) {
    std::vector<std::any> array;
    for (const auto &item : j) {
      array.push_back(json_to_any(item));
    }
    return array;
  } else if (j.is_string()) {
    return j.get<std::string>();
  } else if (j.is_boolean()) {
    return j.get<bool>();
  } else if (j.is_number_integer()) {
    return j.get<int>();
  } else if (j.is_number_unsigned()) {
    return j.get<unsigned int>();
  } else if (j.is_number_float()) {
    return j.get<double>();
  } else {
    return {};
  }
}

namespace ryzenai {
/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename OutT>
void concateOps<InT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;

  for (int i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key =
        "concatenate_" + get_instr_key(txn_fname_prefix_, mat.M, mat.K, mat.N);
    instructions.push_back(std::make_pair(key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

/*
 * concateOps class constructor
 */
template <typename InT, typename OutT>
concateOps<InT, OutT>::concateOps(const std::string &ifmDtype,
                                  const std::string &ofmDtype, bool load_xrt,
                                  const std::map<std::string, std::any> &attr) {

  std::string json_attr;

  if (attr.count("list_attrs") &&
      attr.at("list_attrs").type() == typeid(std::vector<std::string>)) {
    const auto &attrs_vec =
        std::any_cast<const std::vector<std::string> &>(attr.at("list_attrs"));
    json_attr = attrs_vec[0];
  }
  std::vector<std::map<std::string, std::any>> attrsVec;

  // Set default to 320
  graphId_ = 320;
  inChannels_ = 8;
  outChannels_ = 16;

  if (attr.end() != attr.find("model_variant")) {
    const auto &model_variant_vec =
        std::any_cast<const std::vector<std::string> &>(
            attr.at("model_variant"));
    model_variant_ = model_variant_vec[0];
  }

  json data;
  try {
    data = json::parse(json_attr, nullptr, true);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
    throw std::runtime_error("Failed to parse JSON");
  }

  // Fill the attributes
  for (auto &elem : data) {
    std::map<std::string, std::any> attr_map;

    for (auto it = elem.begin(); it != elem.end(); ++it) {
      if (it.key() == "opType" || it.key() == "opIfmDtype" ||
          it.key() == "opWtsDtype" || it.key() == "opOfmDtype") {
        attr_map[it.key()] = json_to_any(it.value());
      } else if (it.key() == "group" || it.key() == "zero_point" ||
                 it.key() == "width" || it.key() == "c1" || it.key() == "c2" ||
                 it.key() == "wts_zp" || it.key() == "shift_conv" ||
                 it.key() == "shift_out") {
        auto value = json_to_any(it.value());
        std::vector<int> x;
        x.push_back(std::any_cast<int>(value));
        attr_map[it.key()] = x;
      } else if (it.key() == "graphID") {
        auto value = json_to_any(it.value());
        graphId_ = std::any_cast<int>(value);
      } else if (it.key() == "inChannels") {
        auto value = json_to_any(it.value());
        inChannels_ = std::any_cast<int>(value);
      } else if (it.key() == "outChannels") {
        auto value = json_to_any(it.value());
        outChannels_ = std::any_cast<int>(value);
      } else if (it.key() == "list_scale") {
        auto value = json_to_any(it.value());
        std::vector<float> x;
        for (const auto &elem : std::any_cast<std::vector<std::any>>(value)) {
          if (elem.type() == typeid(float)) {
            x.push_back(std::any_cast<float>(elem));
          } else if (elem.type() == typeid(double)) {
            x.push_back(static_cast<float>(std::any_cast<double>(elem)));
          }
        }
        attr_map[it.key()] = x;
      } else {
        auto value = json_to_any(it.value());
        std::vector<int> x;
        for (const auto &elem : std::any_cast<std::vector<std::any>>(value)) {
          if (elem.type() == typeid(int)) {
            x.push_back(std::any_cast<int>(elem));
          }
        }
        attr_map[it.key()] = x;
      }
    }
    attrsVec.push_back(attr_map);
  }

  int ortOutIdx = 0;
  if (attr.count("ort_out_index") &&
      attr.at("ort_out_index").type() == typeid(std::vector<int>)) {
    const auto &idx_vector =
        std::any_cast<const std::vector<int> &>(attr.at("ort_out_index"));
    for (const auto &idx : idx_vector) {
      ortOutIdx = idx;
    }
  } else {
    std::cout << "Ort output index not found or not of correct type."
              << std::endl;
  }

  concatenate_id_ = concatenate_count++;

  txn_fname_prefix_ = "concatenate";
  default_shapes_["concatenate"] = std::vector<matrix_shapes>{};
  default_shapes_["concatenate"].emplace_back(320, 8, 16);
  default_shapes_["concatenate"].emplace_back(640, 8, 16);
  default_shapes_["concatenate"].emplace_back(1280, 8, 16);
  default_shapes_["concatenate"].emplace_back(2560, 8, 16);
  run_aie_time_ = 0;

  /* Attribute Parsing */
  for (const auto &attrs : attrsVec) {
    std::string opType = std::any_cast<std::string>(attrs.at("opType"));
    if (opType == "conv") {
      CreateConvOperator(attrs);
    } else if (opType == "maxpool") {
      CreateMaxpoolOperator(attrs);
    } else if (opType == "lstm") {
      CreateLstmOperator(attrs);
    } else if (opType == "convformatmuladd") {
      CreateConvForMatmulAddOperator(attrs);
    } else {
      std::cout << "Error: Concatenate does't support this operator"
                << std::endl;
    }
  }

  std::call_once(logger_flag_, []() {
    std::string header = "concatenate_id Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "IFM_copy_time(ns) IFM_sync_time(ns) "
                         "OFM_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });
}

/* CreateConvOperator private function
Below function creates a conv operator and pushed it's instance in
op_interfaces_ */
template <typename InT, typename OutT>
void concateOps<InT, OutT>::CreateConvOperator(
    const std::map<std::string, std::any> &attrs) {
  std::string opIfmDtype = std::any_cast<std::string>(attrs.at("opIfmDtype"));
  std::string opWtsDtype = std::any_cast<std::string>(attrs.at("opWtsDtype"));
  std::string opOfmDtype = std::any_cast<std::string>(attrs.at("opOfmDtype"));

  std::map<std::string, std::any> attr;
  attr["group"] = attrs.at("group");
  attr["input_shape"] = attrs.at("input_shape");
  attr["output_shape"] = attrs.at("output_shape");
  attr["weight_shape"] = attrs.at("weight_shape");
  attr["zero_point"] = attrs.at("zero_point");
  attr["c1"] = attrs.at("c1");
  attr["c2"] = attrs.at("c2");
  attr["shift_out"] = attrs.at("shift_out");
  attr["shift_conv"] = attrs.at("shift_conv");
  attr["model_variant"] = model_variant_;
  if (attrs.count("width")) {
    attr["width"] = attrs.at("width");
  }
  if (attrs.find("maxpool_kernel_shape") != attrs.end()) {
    attr["maxpool_kernel_shape"] = attrs.at("maxpool_kernel_shape");
    attr["maxpool_stride"] = attrs.at("maxpool_kernel_shape");
  }

  /* Sandip TBD: In below datatype should come from the opIfmDtype, opOfmDtype,
   * and opWtsDtype Store the std::unique_ptr<OpInterface> in the private member
   */
  op_interfaces_.push_back(
      std::make_unique<ryzenai::conv<uint16_t, uint16_t, uint16_t>>(
          opIfmDtype, opWtsDtype, opOfmDtype, false, attr));
}

/* CreateMaxpoolOperator private function
Below function creates a maxpool operator and pushed it's instance in
op_interfaces_ */
template <typename InT, typename OutT>
void concateOps<InT, OutT>::CreateMaxpoolOperator(
    const std::map<std::string, std::any> &attrs) {
  std::string opIfmDtype = std::any_cast<std::string>(attrs.at("opIfmDtype"));
  std::string opWtsDtype = std::any_cast<std::string>(attrs.at("opWtsDtype"));
  std::string opOfmDtype = std::any_cast<std::string>(attrs.at("opOfmDtype"));

  std::map<std::string, std::any> attr;
  attr["group"] = attrs.at("group");
  attr["input_shape"] = attrs.at("input_shape");
  attr["output_shape"] = attrs.at("output_shape");
  attr["weight_shape"] = attrs.at("weight_shape");
  attr["zero_point"] = attrs.at("zero_point");

  /* Sandip TBD: In below datatype should come from the opIfmDtype, opOfmDtype,
   * and opWtsDtype Store the std::unique_ptr<OpInterface> in the private member
   */
  op_interfaces_.push_back(
      std::make_unique<ryzenai::maxpool<uint16_t, uint16_t>>(opIfmDtype,
                                                             opOfmDtype, attr));
}

/* CreateLstmOperator private function
Below function creates a lstm operator and pushed it's instance in
op_interfaces_ */
template <typename InT, typename OutT>
void concateOps<InT, OutT>::CreateLstmOperator(
    const std::map<std::string, std::any> &attrs) {
  std::string opIfmDtype = std::any_cast<std::string>(attrs.at("opIfmDtype"));
  std::string opWtsDtype = std::any_cast<std::string>(attrs.at("opWtsDtype"));
  std::string opOfmDtype = std::any_cast<std::string>(attrs.at("opOfmDtype"));

  std::map<std::string, std::any> attr;
  attr["input_shape"] = attrs.at("input_shape");
  attr["output_shape"] = attrs.at("output_shape");
  attr["scales"] = attrs.at("list_scale");
  attr["zero_points"] = attrs.at("list_zero_point");
  attr["model_num"] = graphId_;
  attr["model_variant"] = model_variant_;

  /* Sandip TBD: In below datatype should come from the opIfmDtype, opOfmDtype,
   * and opWtsDtype Store the std::unique_ptr<OpInterface> in the private member
   */
  op_interfaces_.push_back(
      std::make_unique<ryzenai::lstm<uint16_t, uint16_t, uint16_t>>(
          opIfmDtype, opWtsDtype, opOfmDtype, false, attr));
}

/*CreateConvOperator private function Below function creates a conv operator and
    pushed it's instance in op_interfaces_ */
template <typename InT, typename OutT>
void concateOps<InT, OutT>::CreateConvForMatmulAddOperator(
    const std::map<std::string, std::any> &attrs) {
  std::string opIfmDtype = std::any_cast<std::string>(attrs.at("opIfmDtype"));
  std::string opWtsDtype = std::any_cast<std::string>(attrs.at("opWtsDtype"));
  std::string opOfmDtype = std::any_cast<std::string>(attrs.at("opOfmDtype"));

  std::map<std::string, std::any> attr;
  attr["group"] = attrs.at("group");
  attr["input_shape"] = attrs.at("input_shape");
  attr["output_shape"] = attrs.at("output_shape");
  attr["orig_output_shape"] = attrs.at("orig_output_shape");
  attr["weight_shape"] = attrs.at("weight_shape");
  attr["zero_point"] = attrs.at("zero_point");
  attr["c1"] = attrs.at("c1");
  attr["c2"] = attrs.at("c2");
  attr["wts_zp"] = attrs.at("wts_zp");
  attr["graph_id"] = this->graphId_;
  attr["model_variant"] = model_variant_;
  if (attrs.count("width")) {
    attr["width"] = attrs.at("width");
  }

  /* Sandip TBD: In below datatype should come from the opIfmDtype, opOfmDtype,
   * and opWtsDtype Store the std::unique_ptr<OpInterface> in the private member
   */
  op_interfaces_.push_back(
      std::make_unique<ryzenai::convForMatmulAdd<uint16_t, uint16_t, uint16_t>>(
          opIfmDtype, opWtsDtype, opOfmDtype, false, attr));
}

template <typename InT, typename OutT>
void concateOps<InT, OutT>::set_params(const std::string &modelName,
                                       bool debugFlag) {
  std::string modelNameLowerCase = modelName;
  std::transform(modelNameLowerCase.begin(), modelNameLowerCase.end(),
                 modelNameLowerCase.begin(), ::tolower);
  std::string XCLBIN_FNAME;
  if (modelNameLowerCase == "mswbjvw") {
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                   "\\xclbin\\stx\\tempXclbinFiles\\pso2ConvLstmFc_320.xclbin";
  }
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  debug_ = debugFlag;
}

template <typename InT, typename OutT>
std::string concateOps<InT, OutT>::get_instr_key(std::string prefix,
                                                 int64_t graphId,
                                                 int64_t inChannels,
                                                 int64_t outChannels) const {
  return prefix + "_" + std::to_string(graphId) + "_" +
         std::to_string(inChannels) + "_" + std::to_string(outChannels);
}

/* Below function is not using input tensor and output tensor. So it is OK to
 * call this functin with dummy input and ouput tensors */
template <typename InT, typename OutT>
std::vector<OpArgMap> concateOps<InT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  size_t const_params_bo_size = 0;
  size_t max_scratch_pad_size = 74448896; // 71MB
  size_t ifm_bo_size = 0;
  size_t ofm_bo_size = 0;
  size_t ort_out_index = 1;
  for (size_t i = 0; i < op_interfaces_.size(); ++i) {
    auto &op_interface = op_interfaces_[i];
    auto args_map_list =
        op_interface->get_buffer_reqs(input, output, op_interface->get_attr());

    ort_out_index += op_interface->GetNumConst();
    for (const auto &args_map : args_map_list) {
      if (args_map.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
        const_params_bo_size += args_map.size;
      }
      if (args_map.arg_type == OpArgMap::OpArgType::SCRATCH_PAD) {
        max_scratch_pad_size = std::max(max_scratch_pad_size, args_map.size);
      }
      if ((i == 0) && (args_map.arg_type == OpArgMap::OpArgType::INPUT)) {
        ifm_bo_size = args_map.size;
      }
      if ((i == (op_interfaces_.size() - 1)) &&
          (args_map.arg_type == OpArgMap::OpArgType::OUTPUT)) {
        ofm_bo_size = args_map.size;
      }
    }
  }
  // This block will be taken care of when the fc layer is added in concateops
  // ofm_bo_size = (graphId_ / 80) * 21920;

  /* Sandip TBD: Below if else condition is a workaround. The proper fix is the
   * ofm bo should be given xrt id 2 and scratch bo should be given xrt id 3 for
   * all graphs */
  std::vector<OpArgMap> arg_map;
  if ((graphId_ == 1280) || (graphId_ == 2560) || (graphId_ == 5120) ||
      (graphId_ == 8000)) {
    arg_map = {
        {OpArgMap::OpArgType::INPUT, 1, 0, 0, ifm_bo_size},
        {OpArgMap::OpArgType::CONST_INPUT, 0, 1, 0, const_params_bo_size},
        {OpArgMap::OpArgType::OUTPUT, 3, ort_out_index, 0, ofm_bo_size},
        {OpArgMap::OpArgType::SCRATCH_PAD, 2, 0, 0, max_scratch_pad_size}};
  } else {
    arg_map = {
        {OpArgMap::OpArgType::INPUT, 1, 0, 0, ifm_bo_size},
        {OpArgMap::OpArgType::CONST_INPUT, 0, 1, 0, const_params_bo_size},
        {OpArgMap::OpArgType::OUTPUT, 2, ort_out_index, 0, ofm_bo_size},
        {OpArgMap::OpArgType::SCRATCH_PAD, 3, 0, 0, max_scratch_pad_size}};
  }
  return arg_map;
}

static std::string GetParamKey(std::string prefix, int64_t graphId,
                               int64_t inChannels, int64_t outChannels) {
  return prefix + "_" + std::to_string(graphId) + "_" +
         std::to_string(inChannels) + "_" + std::to_string(outChannels);
}

template <typename InT, typename OutT>
void concateOps<InT, OutT>::WriteToFile(void *src, uint64_t length) {
  uint8_t *dataPtr = (uint8_t *)src;
  std::string testDataFolder = OpInterface::get_dd_base_dir() + "\\" + "tests" +
                               "\\" + "cpp" + "\\" + "unit_tests" + "\\" +
                               "testDataMladf" + "\\" + "GeneratedWeights";
  std::string fileName =
      testDataFolder + "\\" +
      GetParamKey("wtsGenerated", graphId_, inChannels_, outChannels_) + ".txt";
  write32BitHexTxtFile<uint16_t>(fileName, (uint16_t *)src, length);
}

template <typename InT, typename OutT>
void concateOps<InT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  size_t generatedWeightSize = 0;

  int constParamIndex = 0;
  for (size_t i = 0; i < op_interfaces_.size(); ++i) {
    auto &op_interface = op_interfaces_[i];
    std::vector<Tensor> sub_const_params;
    for (int j = 0; j < op_interface->GetNumConst(); j++) {
      sub_const_params.push_back(const_params.at(constParamIndex));
      constParamIndex++;
    }
    op_interface->initialize_const_params(io, sub_const_params,
                                          op_interface->get_attr());

    /* Get buffer sizes required for this operators. We are not using input and
     * output tenosrs in get_buffer_req(). So calling with dummy tensors */
    std::vector<Tensor> input;
    std::vector<Tensor> output;
    auto args_map_list =
        op_interface->get_buffer_reqs(input, output, op_interface->get_attr());
    for (const auto &args_map : args_map_list) {
      if (args_map.arg_type == OpArgMap::OpArgType::CONST_INPUT) {
        io.update_offset(args_map.size);
        generatedWeightSize += args_map.size;
      }
    }
  }

  if (debug_) {
    auto val = io.read(0, generatedWeightSize);
    WriteToFile(val.data(), (generatedWeightSize >> 1));
  }
}

template <typename InT, typename OutT>
void concateOps<InT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("concateOps initialize_const_params ...");

  /* Get buffer sizes required for this operators. We are not using input and
   * output tenosrs in get_buffer_req(). So calling with dummy tensors */
  std::vector<Tensor> input;
  std::vector<Tensor> output;
  size_t CONST_BO_SIZE, IFM_BO_SIZE, OFM_BO_SIZE, SCRATCH_BO_SIZE;
  CONST_BO_SIZE = IFM_BO_SIZE = OFM_BO_SIZE = SCRATCH_BO_SIZE = 0;
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
    if (args_map.arg_type == OpArgMap::OpArgType::SCRATCH_PAD) {
      SCRATCH_BO_SIZE = args_map.size;
    }
  }

  RYZENAI_LOG_TRACE("Concatenate: IFM_BO_SIZE:" + std::to_string(IFM_BO_SIZE) +
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

  /* Sandip TBD : uint16_t should not be hardcoded */
  uint16_t *b_bo_map = constBo_.map<uint16_t *>();
  auto bo_const = BoConst(b_bo_map);
  initialize_const_params(bo_const, const_params);
  constBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  RYZENAI_LOG_TRACE("Concatenate initialize_const_params ... DONE");
}

template <typename InT, typename OutT>
const std::vector<uint8_t> concateOps<InT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  bool elf_flow = false;
  if (attr.end() != attr.find("elf_flow")) {
    elf_flow = std::any_cast<uint32_t>(attr.at("elf_flow"));
  }

  std::string txn_key;
  if (model_variant_.size()) {
    txn_key = "concatenate_" + txn_fname_prefix_ + "_" + model_variant_ + "_" +
              std::to_string(inChannels_) + "_" + std::to_string(outChannels_);
  } else {
    txn_key = "concatenate_" + txn_fname_prefix_ + "_" +
              std::to_string(graphId_) + "_" + std::to_string(inChannels_) +
              "_" + std::to_string(outChannels_);
  }
  if (elf_flow) {
    txn_key += "_elf";
  }

  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename OutT>
void concateOps<InT, OutT>::execute(std::vector<Tensor> &input,
                                    std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("Conv execute ...");

  xrt::bo param_bo;

  auto ifmDtype = input.at(0).dtype;
  size_t ifmDSize;
  if (ifmDtype == "uint16") {
    ifmDSize = 2;
  } else if (ifmDtype == "uint8") {
    ifmDSize = 1;
  }
  size_t ifmDataSize = input.at(0).shape[0] * input.at(0).shape[1] *
                       input.at(0).shape[2] * ifmDSize;
  ifmBo_.write(input.at(0).data, ifmDataSize, 0);
  ifmBo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto instr_bo_key =
      "concatenate_" + txn_fname_prefix_ + "_" + std::to_string(graphId_) +
      "_" + std::to_string(inChannels_) + "_" + std::to_string(outChannels_);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));

  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the Conv kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  /* kernel call for Conv that supports transaction binary flow */

  /* Sandip TBD: Below if else condition is a workaround. The proper fix is the
   * ofm bo should be given xrt id 2 and scratch bo should be given xrt id 3 for
   * all graphs */
  if ((graphId_ == 1280) || (graphId_ == 2560)) {

    ryzenai::dynamic_dispatch::execute_kernel(
        kernel_, 2, instr_bo, instr_bo_words, constBo_, ifmBo_, scratchBo_,
        ofmBo_, 0, true, false);
  } else {

    ryzenai::dynamic_dispatch::execute_kernel(
        kernel_, 2, instr_bo, instr_bo_words, constBo_, ifmBo_, ofmBo_,
        scratchBo_, 0, true, false);
  }
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);

  // sync output activation to host memory
  ofmBo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  ofmBo_.read(output.at(0).data);

  RYZENAI_LOG_TRACE("Conv execute ... DONE");
}

template <typename InT, typename OutT>
std::once_flag concateOps<InT, OutT>::logger_flag_;

template <typename InT, typename OutT>
uint64_t concateOps<InT, OutT>::concatenate_count = 0;

template <typename InT, typename OutT>
std::once_flag concateOps<InT, OutT>::instr_reg_flag_;

template class concateOps<uint16_t, uint16_t>;

} // namespace ryzenai
