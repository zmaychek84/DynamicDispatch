/*
 Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 Licensed under the MIT License.
 */
#include <any>
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

#include <ops/bmm/bmm.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>
#include <xclbin_container.hpp>

// AIE Driver header
#include "xaiengine.h"

namespace ryzenai {

namespace {
std::string getXCLBinName(std::string op_version) {
  if (op_version == "v1" || op_version == "flat") {
    return LLAMA2_MLADF_2x4x4_V1_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_NAME;
  } else {
    return LLAMA2_MLADF_2x4x4_GEMMBFP16_SILU_MUL_MHA_RMS_ROPE_XCLBIN_NAME;
  }
}
} // namespace

static std::tuple<size_t, size_t, size_t>
extract_MKN(const std::vector<Tensor> &inputs) {
  size_t M;
  if (inputs.at(0).shape.size() == 2) {
    M = inputs.at(0).shape.at(0);
  } else if (inputs.at(0).shape.size() == 3) { // has batch_dim
    M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(1);
  } else {
    throw std::runtime_error("Input Shape is not supported");
  }
  size_t K = inputs.at(0).shape.at(1);
  size_t N = inputs.at(1).shape.at(1);
  return std::make_tuple(M, K, N);
}

/*
 * bmm is an experimental class to offload bf16 * bf16  (int16_t) matrix
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
void bmm<InT, WtT, OutT>::set_kernel_shapes() {
  // Use largest M dimension as the default
  //    NOTE: smaller M's can be selected in run_aie if needed
  kernel_x_shape_[0] = a_shape_[0]; // B
  kernel_x_shape_[1] = a_shape_[1]; // M
  kernel_x_shape_[2] = a_shape_[2]; // K

  kernel_y_shape_[0] = w_shape_[0]; // B
  kernel_y_shape_[1] = w_shape_[1]; // K
  kernel_y_shape_[2] = w_shape_[2]; // N

  kernel_z_shape_[0] = a_shape_[0];
  kernel_z_shape_[1] = a_shape_[1];
  kernel_z_shape_[2] = w_shape_[2];
}

template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::setup_instr_init() {}
/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::setup_instr_registry(
    const std::map<std::string, std::any> &attr) {
  std::vector<std::vector<size_t>> supported_shapes;
  std::vector<std::pair<std::string, bool>> instructions;
  if (attr.find("shapes") != attr.end()) {
    raw_shapes_[txn_fname_prefix_] = std::vector<std::vector<size_t>>{};
    auto shapes = std::any_cast<std::vector<std::vector<int>>>(
        attr.find("shapes")->second);
    for (auto sh : shapes) {
      raw_shapes_[txn_fname_prefix_].emplace_back(
          std::vector<size_t>{(size_t)sh[0], (size_t)sh[1], (size_t)sh[2],
                              (size_t)sh[3], (size_t)sh[4], (size_t)sh[5]});
      supported_shapes.emplace_back(
          std::vector<size_t>{(size_t)sh[0], (size_t)sh[1], (size_t)sh[2],
                              (size_t)sh[3], (size_t)sh[4], (size_t)sh[5]});
    }
  } else {
    supported_shapes = default_shapes_.find(txn_fname_prefix_)->second;
  }
  for (auto &sh : supported_shapes) {
    // raw shape is the actual shape from ONNX
    auto key = get_instr_key(txn_fname_prefix_, sh[0], sh[1], sh[2], sh[3],
                             sh[4], sh[5]);
    instructions.push_back(std::make_pair(key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
std::string bmm<InT, WtT, OutT>::get_instr_key(std::string prefix, size_t m,
                                               size_t k, size_t n) const {
  return "bmm_" + prefix + "_" + std::to_string(m) + "_" + std::to_string(k) +
         "_" + std::to_string(n);
}

template <typename InT, typename WtT, typename OutT>
std::string bmm<InT, WtT, OutT>::get_instr_key(std::string prefix, size_t b0,
                                               size_t m0, size_t k0, size_t b1,
                                               size_t k1, size_t n1) const {
  return "bmm_" + prefix + "_" + std::to_string(b0) + "_" + std::to_string(m0) +
         "_" + std::to_string(k0) + "_" + std::to_string(b1) + "_" +
         std::to_string(k1) + "_" + std::to_string(n1);
}

/*
 * bmm class constructor
 *
 * @param kernel_x_shape tuple containing of M x K dimension base bmm
 * supported on IPU
 * @param kernel_y_shape tuple containing of K x N dimension base bmm
 * supported on IPU
 *
 * NOTE: If the input shape has a smaller M dimension than the kernel
 * shape initialized here, the execute function can transparently
 * call a smaller BMM to reduce padding overhead. The kernel
 * shape passed here should have largest supported M dimension.
 *
 */
template <typename InT, typename WtT, typename OutT>
bmm<InT, WtT, OutT>::bmm(const std::string &a_dtype, const std::string &b_dtype,
                         const std::string &c_dtype, bool load_xrt,
                         bool transpose,
                         const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"uint16_t", "a16"}, {"bfloat16", "a16"}};
  txnbin_b_header = {{"uint16_t", "w16"}, {"bfloat16", "w16"}};

  op_version_ = "v1";
  if (attr.find("op_version") != attr.end()) {
    op_version_ = std::any_cast<std::string>(attr.find("op_version")->second);
    if (op_version_ != "v1" && op_version_ != "flat") {
      throw std::runtime_error("The selected op version does not exist");
    }
  }

  std::string txn_key =
      "bmm_v1_" + txnbin_a_header.at(a_dtype) + txnbin_b_header.at(b_dtype);
  std::string txn_key_transpose = "bmm_v1_trans_" +
                                  txnbin_a_header.at(a_dtype) +
                                  txnbin_b_header.at(b_dtype);

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_[txn_key_transpose] = std::vector<std::vector<size_t>>{};

  // psu1 bmm1
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 96, 32, 96, 128});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 96, 32, 96, 256});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 96, 32, 96, 384});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 96, 32, 96, 512});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 96, 32, 96, 640});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 96, 32, 96, 768});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 96, 32, 96, 896});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 96, 32, 96, 1024});

  // psu1 bmm2
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 128, 32, 128, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 256, 32, 256, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 384, 32, 384, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 512, 32, 512, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 640, 32, 640, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 768, 32, 768, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 896, 32, 896, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 1024, 32, 1024, 96});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 2048, 128, 32, 128, 2048});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1024, 128, 32, 128, 1024});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 512, 128, 32, 128, 512});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 256, 128, 32, 128, 256});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 128});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 128, 128, 8, 128, 128});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 128, 128, 8, 128, 256});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 128, 128, 8, 128, 384});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 128, 128, 8, 128, 512});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 128, 128, 8, 128, 640});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 128, 128, 8, 128, 768});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 128, 128, 8, 128, 896});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 128, 128, 8, 128, 1024});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 128, 128, 8, 128, 1536});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 128, 128, 8, 128, 2048});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 256, 128, 8, 128, 256});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 384, 128, 8, 128, 384});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 512, 128, 8, 128, 512});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 640, 128, 8, 128, 640});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 768, 128, 8, 128, 768});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 896, 128, 8, 128, 896});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 1024, 128, 8, 128, 1024});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 1536, 128, 8, 128, 1536});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{8, 2048, 128, 8, 128, 2048});

  // phi3 shapes
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 96, 8, 96, 128});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 96, 32, 96, 128});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 256, 96, 8, 96, 256});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 256, 96, 32, 96, 256});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 512, 96, 8, 96, 512});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 512, 96, 32, 96, 512});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1024, 96, 8, 96, 1024});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1024, 96, 32, 96, 1024});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 2048, 96, 8, 96, 2048});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 2048, 96, 32, 96, 2048});

  default_shapes_[txn_key] = std::vector<std::vector<size_t>>{};
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 2048, 2048, 32, 2048, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1024, 1024, 32, 1024, 128});

  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 896, 896, 32, 896, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 768, 768, 32, 768, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 640, 640, 32, 640, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 512, 512, 32, 512, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 384, 384, 32, 384, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 256, 32, 256, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 384, 32, 384, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 512, 32, 512, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 640, 32, 640, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 768, 32, 768, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 896, 32, 896, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 256, 256, 32, 256, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 256, 8, 256, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 384, 8, 384, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 512, 8, 512, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 640, 8, 640, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 768, 8, 768, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 896, 8, 896, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1024, 8, 1024, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 256, 256, 8, 256, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 384, 384, 8, 384, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 512, 512, 8, 512, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 640, 640, 8, 640, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 768, 768, 8, 768, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 896, 896, 8, 896, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1024, 1024, 8, 1024, 128});

  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1152, 1152, 32, 1152, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1280, 1280, 32, 1280, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1024, 32, 1024, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1152, 32, 1152, 128});

  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1280, 32, 1280, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1408, 32, 1408, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1536, 32, 1536, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1664, 32, 1664, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1792, 32, 1792, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1920, 32, 1920, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 2048, 32, 2048, 128});

  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1408, 1408, 32, 1408, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1536, 1536, 32, 1536, 128});

  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1152, 1152, 8, 1152, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1280, 1280, 8, 1280, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1152, 8, 1152, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1280, 8, 1280, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1408, 8, 1408, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1536, 8, 1536, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1664, 8, 1664, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1792, 8, 1792, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 1920, 8, 1920, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 2048, 8, 2048, 128});

  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1408, 1408, 8, 1408, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1536, 1536, 8, 1536, 128});

  // phi3 shapes
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 256, 256, 8, 256, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 256, 256, 32, 256, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 512, 512, 8, 512, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 512, 512, 32, 512, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1024, 1024, 8, 1024, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1024, 1024, 32, 1024, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 2048, 2048, 8, 2048, 96});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 2048, 2048, 32, 2048, 96});

  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 1024, 8, 1024, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 1152, 8, 1152, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 1280, 8, 1280, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 128});

  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 1408, 8, 1408, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 1536, 8, 1536, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 2048, 8, 2048, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 256, 8, 256, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 384, 8, 384, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 512, 8, 512, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 640, 8, 640, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 768, 8, 768, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 896, 8, 896, 128});

  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1024, 128, 8, 128, 1024});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1152, 128, 32, 128, 1152});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1152, 128, 8, 128, 1152});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1280, 128, 32, 128, 1280});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1280, 128, 8, 128, 1280});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 1024});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 1152});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 1280});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 1408});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 1536});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 1664});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 1792});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 1920});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 2048});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 256});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 384});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 512});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 640});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 768});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 896});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 1024});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 1152});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 128});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 1280});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 1408});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 1536});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 1664});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 1792});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 1920});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 2048});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 256});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 384});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 512});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 640});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 768});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 8, 128, 896});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1536, 128, 32, 128, 1536});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1536, 128, 8, 128, 1536});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 2048, 128, 8, 128, 2048});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 256, 128, 8, 128, 256});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 384, 128, 32, 128, 384});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 384, 128, 8, 128, 384});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 512, 128, 8, 128, 512});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 640, 128, 32, 128, 640});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 640, 128, 8, 128, 640});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 768, 128, 32, 128, 768});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 768, 128, 8, 128, 768});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 896, 128, 32, 128, 896});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 896, 128, 8, 128, 896});
  // ChatGLM3
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 3072, 32, 3072, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 3072, 3072, 32, 3072, 128});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 32, 128, 3072});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 3072, 128, 32, 128, 3072});
  // ChatGLM3 - kv heads = 2
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 128, 3072, 2, 3072, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 3072, 3072, 2, 3072, 128});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 128, 128, 2, 128, 3072});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 3072, 128, 2, 128, 3072});

  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 2176});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 2304});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 2432});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 2560});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 2688});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 2816});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 2944});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 3072});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 3200});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 3328});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 3456});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 3584});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 3712});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 3840});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 3968});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 4096});

  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 2176, 8, 2176, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 2304, 8, 2304, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 2432, 8, 2432, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 2560, 8, 2560, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 2688, 8, 2688, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 2816, 8, 2816, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 2944, 8, 2944, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 3072, 8, 3072, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 3200, 8, 3200, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 3328, 8, 3328, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 3456, 8, 3456, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 3584, 8, 3584, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 3712, 8, 3712, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 3840, 8, 3840, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 3968, 8, 3968, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 4096, 8, 4096, 128});

  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 128});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 256});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 384});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 512});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 640});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 768});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 896});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 1024});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 1152});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 1280});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 1408});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 1536});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 1664});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 1792});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 1920});
  default_shapes_[txn_key_transpose].emplace_back(
      std::vector<size_t>{32, 1, 128, 8, 128, 2048});

  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 1664, 8, 1664, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 1792, 8, 1792, 128});
  default_shapes_[txn_key].emplace_back(
      std::vector<size_t>{32, 1, 1920, 8, 1920, 128});

  raw_shapes_[txn_key_transpose] = std::vector<std::vector<size_t>>{};
  raw_shapes_[txn_key] = std::vector<std::vector<size_t>>{};

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);
  transpose_ = transpose;

  txn_fname_prefix_ = (transpose_) ? txn_key_transpose : txn_key;
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("txn_fname_prefix : {}", txn_fname_prefix_));
  bmm_id_ = bmm_count++;
  // KERNEL_M_MAX = 32 * 2048;
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
    std::string header = "bmm_id M K N kernel_m kernel_k kernel_n Execute"
                         "time(ns) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });
}

template <typename InT, typename WtT, typename OutT>
xrt::bo bmm<InT, WtT, OutT>::create_bo(void *usr_ptr, size_t size,
                                       int operand_index) {
  std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(usr_ptr);
  constexpr std::uint32_t MASK = ((1 << 12) - 1);
  DD_ASSERT((addr & MASK) == 0, "addr must be multiple of 4096 address.");
  auto bo =
      xrt::bo(xrt_ctx_->get_context(), usr_ptr, size, xrt::bo::flags::host_only,
              xrt_ctx_->get_kernel().group_id(0));
  if (operand_index == 0) {
    a_bo_ = bo;
  } else if (operand_index == 1) {
    b_bo_ = bo;
  } else if (operand_index == 2) {
    c_bo_ = bo;
  } else {
    throw std::runtime_error("create_bo with invalid operand_index " +
                             std::to_string(operand_index));
  }
  return bo;
}
template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::set_params(
    const std::string &model_name, std::vector<size_t> input_shape,
    const std::map<std::string, std::any> &attr) {
  a_shape_[0] = input_shape.at(0);
  a_shape_[1] = input_shape.at(1);
  a_shape_[2] = input_shape.at(2);

  // TODO this is for MHA for shor to call execute with xrt::bo
  if (transpose_) {
    w_shape_[0] = a_shape_[0];
    w_shape_[1] = a_shape_[2];
    w_shape_[2] = a_shape_[1];
  } else {
    w_shape_[0] = a_shape_[0];
    w_shape_[1] = a_shape_[1];
    w_shape_[2] = 128;
  }
  set_kernel_shapes();

  // KERNEL_M_MAX = (int)M;

  std::string XCLBIN_FNAME = getXCLBinName(op_version_);

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(
      XCLBIN_FNAME, 0, {},
      XclbinContainer::getInstance().get_xclbin_content(XCLBIN_FNAME));

  if (transpose_) {
    if (op_version_ == "v1") {
      std::call_once(trans_instr_reg_v1_flag_,
                     [this, &attr]() { setup_instr_init(); });
    } else {
      std::call_once(trans_instr_reg_flag_,
                     [this, &attr]() { setup_instr_init(); });
    }
  } else {
    if (op_version_ == "v1") {
      std::call_once(instr_reg_v1_flag_,
                     [this, &attr]() { setup_instr_init(); });
    } else {
      std::call_once(instr_reg_flag_, [this, &attr]() { setup_instr_init(); });
    }
  }
  setup_instr_registry(attr);
  instr_bo_key_ =
      get_instr_key(txn_fname_prefix_, a_shape_[0], a_shape_[1], a_shape_[2],
                    w_shape_[0], w_shape_[1], w_shape_[2]);

  auto kernel_ = xrt_ctx_->get_kernel();
  size_t B_BO_SIZE = kernel_y_shape_[0] * kernel_y_shape_[1] *
                     kernel_y_shape_[2] * b_dtype_size_;
  size_t A_BO_SIZE = kernel_x_shape_[0] * kernel_x_shape_[1] *
                     kernel_x_shape_[2] * a_dtype_size_;
  size_t C_BO_SIZE = kernel_z_shape_[0] * kernel_z_shape_[1] *
                     kernel_z_shape_[2] * c_dtype_size_;

  const bool skip_create_input_a =
      (attr.find("skip_create_input_a") != attr.end()) ||
      (attr.find("skip_create_input") != attr.end());

  const bool skip_create_input_b =
      (attr.find("skip_create_input_b") != attr.end()) ||
      (attr.find("skip_create_input") != attr.end());

  if (!skip_create_input_a) {
    a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                    kernel_.group_id(0));
  }

  if (!skip_create_input_b) {
    b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                    kernel_.group_id(0));
  }

  if (attr.find("skip_create_output") == attr.end()) {
    c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                    kernel_.group_id(0));
  }

  return;
}

template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::set_params(
    const std::string &model_name, std::vector<size_t> input_shape,
    std::vector<size_t> weight_shape,
    const std::map<std::string, std::any> &attr) {
  a_shape_[0] = input_shape.at(0);
  a_shape_[1] = input_shape.at(1);
  a_shape_[2] = input_shape.at(2);

  // TODO this is for MHA for shor to call execute with xrt::bo

  w_shape_[0] = weight_shape.at(0);
  w_shape_[1] = weight_shape.at(1);
  w_shape_[2] = weight_shape.at(2);

  set_kernel_shapes();

  // KERNEL_M_MAX = (int)M;

  std::string XCLBIN_FNAME = getXCLBinName(op_version_);

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(
      XCLBIN_FNAME, 0, {},
      XclbinContainer::getInstance().get_xclbin_content(XCLBIN_FNAME));

  if (transpose_) {
    if (op_version_ == "v1") {
      std::call_once(trans_instr_reg_v1_flag_,
                     [this, &attr]() { setup_instr_init(); });
    } else {
      std::call_once(trans_instr_reg_flag_,
                     [this, &attr]() { setup_instr_init(); });
    }
  } else {
    if (op_version_ == "v1") {
      std::call_once(instr_reg_v1_flag_,
                     [this, &attr]() { setup_instr_init(); });
    } else {
      std::call_once(instr_reg_flag_, [this, &attr]() { setup_instr_init(); });
    }
  }
  setup_instr_registry(attr);
  instr_bo_key_ =
      get_instr_key(txn_fname_prefix_, a_shape_[0], a_shape_[1], a_shape_[2],
                    w_shape_[0], w_shape_[1], w_shape_[2]);

  auto kernel_ = xrt_ctx_->get_kernel();
  size_t B_BO_SIZE = kernel_y_shape_[0] * kernel_y_shape_[1] *
                     kernel_y_shape_[2] * b_dtype_size_;
  size_t A_BO_SIZE = kernel_x_shape_[0] * kernel_x_shape_[1] *
                     kernel_x_shape_[2] * a_dtype_size_;
  size_t C_BO_SIZE = kernel_z_shape_[0] * kernel_z_shape_[1] *
                     kernel_z_shape_[2] * c_dtype_size_;

  const bool skip_create_input_a =
      (attr.find("skip_create_input_a") != attr.end()) ||
      (attr.find("skip_create_input") != attr.end());

  const bool skip_create_input_b =
      (attr.find("skip_create_input_b") != attr.end()) ||
      (attr.find("skip_create_input") != attr.end());

  if (!skip_create_input_a) {
    a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                    kernel_.group_id(0));
  }

  if (!skip_create_input_b) {
    b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                    kernel_.group_id(0));
  }

  if (attr.find("skip_create_output") == attr.end()) {
    c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                    kernel_.group_id(0));
  }

  return;
}

/*
 * copy weight matrix into XRT BOs with padding and tiling
 *
 * this method copies the weight matrix into XRT BOs. This is re-used for
 * every bmm performed for this object with different activations. weight
 * matrix is padded, tiled and reformatted while copying to XRT BOs. padding
 * is done to align with kernel_y_shape each tile of the weight matrix is of
 * shape kernel_y_shape this method also reformats the matrix b/weight matrix
 * as required by AIE/IPU bmm implementation
 *
 * @param weights pointer to the weight matrix
 * @param w_shape tuple containing the shape of the weight matrix
 *
 * @return none
 */

template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("bmm initialize_const_params ...");

  DD_THROW_IF(
      (const_params.size() != 1) || ((const_params.at(0).shape.size() != 2) &&
                                     (const_params.at(0).shape.size() != 3)),
      OpsFusion::dd_format("Unsupported const spec for bmm\n") +
          OpsFusion::dd_format(
              "(Details : #const params == 1 ({}), Const param dim == 2 ({})",
              const_params.size(), const_params.at(0).shape.size()));

  w_shape_[0] = const_params.at(0).shape.at(0);
  w_shape_[1] = const_params.at(0).shape.at(1);
  w_shape_[2] = const_params.at(0).shape.at(2);

  set_kernel_shapes();
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  auto b_copy_start = GET_ELAPSED_TIME_NS();
  auto b_format_start = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = b_bo_.map<WtT *>();
  auto weights = (WtT *)const_params.at(0).data;
  auto weight_size = w_shape_[0] * w_shape_[1] * w_shape_[2] * b_dtype_size_;
  memcpy((void *)(static_cast<WtT *>(b_bo_map)), (void *)weights, weight_size);
  auto b_format_stop = GET_ELAPSED_TIME_NS();
  b_format_time_ += static_cast<int64_t>(b_format_stop - b_format_start);
  auto b_copy_stop = GET_ELAPSED_TIME_NS();
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();
  b_copy_time_ = static_cast<int64_t>(b_copy_stop - b_copy_start);
  b_sync_time_ = static_cast<int64_t>(b_sync_stop - b_sync_start);
  RYZENAI_LOG_TRACE("bmm initialize_const_params ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                  std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("bmm execute ...");
  a_copy_time_ = 0;
  a_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;
  num_run_aie_ = 0;
  auto exec_start = GET_ELAPSED_TIME_NS();

  a_shape_[0] = input.at(0).shape.at(0);
  a_shape_[1] = input.at(0).shape.at(1);
  a_shape_[2] = input.at(0).shape.at(2);
  c_shape_[0] = a_shape_[0];
  c_shape_[1] = a_shape_[1];
  c_shape_[2] = w_shape_[2];
  auto aie_out = (OutT *)output.at(0).data;
  auto a = (InT *)input.at(0).data;

  auto a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();

  memcpy((void *)a_bo_map, (void *)a,
         (a_shape_[0] * a_shape_[1] * a_shape_[2] * a_dtype_size_));
  auto a_copy_stop = GET_ELAPSED_TIME_NS();
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();
  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  std::vector<xrt::bo> inputs = {a_bo_, b_bo_};
  std::vector<xrt::bo> outputs = {c_bo_};
  execute(inputs, outputs);
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  // sync output activation to host memory
  auto c_sync_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  OutT *c_bo_map = c_bo_.map<OutT *>();
  auto c_sync_stop = GET_ELAPSED_TIME_NS();
  c_sync_time_ += static_cast<int64_t>(c_sync_stop - c_sync_start);
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);
  auto c_copy_start = GET_ELAPSED_TIME_NS();
  memcpy((void *)aie_out, (void *)c_bo_map,
         (c_shape_[0] * c_shape_[1] * c_shape_[2] * c_dtype_size_));
  auto c_copy_end = GET_ELAPSED_TIME_NS();
  c_copy_time_ += static_cast<int64_t>(c_copy_end - c_copy_start);
  auto exec_end = GET_ELAPSED_TIME_NS();
  RYZENAI_LOG_INFO(
      std::to_string(bmm_id_) + " " + std::to_string(a_shape_[0]) + " " +
      std::to_string(a_shape_[1]) + " " + std::to_string(a_shape_[2]) + " " +
      std::to_string(w_shape_[1]) + " " + std::to_string(w_shape_[2]) + " " +
      std::to_string(kernel_x_rows) + " " + std::to_string(a_shape_[1]) + " " +
      std::to_string(w_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
  RYZENAI_LOG_TRACE("bmm execute ... DONE");
}

template <typename InT, typename WtT, typename OutT>
std::vector<xrt::bo> bmm<InT, WtT, OutT>::get_inputs() {
  return {a_bo_, b_bo_};
}

template <typename InT, typename WtT, typename OutT>
std::vector<xrt::bo> bmm<InT, WtT, OutT>::get_outputs() {
  return {c_bo_};
}

template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::set_execute_kernel_shape(
    const std::vector<size_t> &input_shape) {
  if (a_shape_[0] != input_shape.at(0) || a_shape_[1] != input_shape.at(1) ||
      a_shape_[2] != input_shape.at(2)) {
    a_shape_[0] = input_shape.at(0);
    a_shape_[1] = input_shape.at(1);
    a_shape_[2] = input_shape.at(2);
    if (transpose_) {
      w_shape_[0] = a_shape_[0];
      w_shape_[1] = a_shape_[2];
      w_shape_[2] = a_shape_[1];
    } else {
      w_shape_[0] = a_shape_[0];
      w_shape_[1] = a_shape_[1];
      w_shape_[2] = 128;
    }
    set_kernel_shapes();
    instr_bo_key_ =
        get_instr_key(txn_fname_prefix_, a_shape_[0], a_shape_[1], a_shape_[2],
                      w_shape_[0], w_shape_[1], w_shape_[2]);
  }
  return;
}

template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::set_execute_kernel_shape(
    const std::vector<size_t> &input_shape,
    const std::vector<size_t> &weight_shape) {
  if (a_shape_[0] != input_shape.at(0) || a_shape_[1] != input_shape.at(1) ||
      a_shape_[2] != input_shape.at(2) || w_shape_[0] != weight_shape.at(0) ||
      w_shape_[1] != weight_shape.at(1) || w_shape_[2] != weight_shape.at(2)) {
    a_shape_[0] = input_shape.at(0);
    a_shape_[1] = input_shape.at(1);
    a_shape_[2] = input_shape.at(2);

    w_shape_[0] = weight_shape[0];
    w_shape_[1] = weight_shape[1];
    w_shape_[2] = weight_shape[2];

    set_kernel_shapes();
    instr_bo_key_ =
        get_instr_key(txn_fname_prefix_, a_shape_[0], a_shape_[1], a_shape_[2],
                      w_shape_[0], w_shape_[1], w_shape_[2]);
  }
  return;
}

template <typename InT, typename WtT, typename OutT>
void bmm<InT, WtT, OutT>::execute(std::vector<xrt::bo> &input,
                                  std::vector<xrt::bo> &output, bool wait) {
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key_);
  auto instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, input[0], input[1],
                                            output[0], 0, 0, wait, false);
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
void bmm<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> bmm<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  // auto [M, K, N] = extract_MKN(input);
  // auto [Mo, Ko] = map_padded_shape(M, K);

  size_t B0 = input.at(0).shape.at(0);
  size_t M0 = input.at(0).shape.at(1);
  size_t K0 = input.at(0).shape.at(2);
  size_t B1 = input.at(1).shape.at(0);
  size_t K1 = input.at(1).shape.at(1);
  size_t N1 = input.at(1).shape.at(2);
  std::string txn_key =
      get_instr_key(txn_fname_prefix_, B0, M0, K0, B1, K1, N1);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> bmm<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  size_t B0 = input.at(0).shape.at(0);
  size_t M0 = input.at(0).shape.at(1);
  size_t K0 = input.at(0).shape.at(2);
  size_t B1 = input.at(1).shape.at(0);
  size_t K1 = input.at(1).shape.at(1);
  size_t N1 = input.at(1).shape.at(2);
  size_t input_bo_size = (B0 * M0 * K0 * a_dtype_size_);
  size_t const_params_bo_size = (B1 * K1 * N1 * b_dtype_size_);
  size_t output_bo_size = (B0 * M0 * N1 * c_dtype_size_);
  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::INPUT, 1, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 2, 2, 0, output_bo_size}};
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("bmm Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
std::once_flag bmm<InT, WtT, OutT>::logger_flag_;
template <typename InT, typename WtT, typename OutT>
uint64_t bmm<InT, WtT, OutT>::bmm_count = 0;
template <typename InT, typename WtT, typename OutT>
std::once_flag bmm<InT, WtT, OutT>::instr_reg_flag_;
template <typename InT, typename WtT, typename OutT>
std::once_flag bmm<InT, WtT, OutT>::trans_instr_reg_flag_;
template <typename InT, typename WtT, typename OutT>
std::once_flag bmm<InT, WtT, OutT>::instr_reg_v1_flag_;
template <typename InT, typename WtT, typename OutT>
std::once_flag bmm<InT, WtT, OutT>::trans_instr_reg_v1_flag_;
template class bmm<uint16_t, uint16_t, uint16_t>;

} // namespace ryzenai
