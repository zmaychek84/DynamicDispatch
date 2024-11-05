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

#include "txn/txn_utils.hpp"
#include "txn_container.hpp"
#include <ops/experimental/matmul_tile.hpp>
#include <ops/op_interface.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>
// AIE Driver header
#include "xaiengine.h"

#include "ops/ops_common/matmul_matrix.hpp"
#include <txn_helper/txn_tiling_util.hpp>

using namespace matmul_matrix;
using utils::txn_util;

namespace ryzenai {

static std::tuple<size_t, size_t, size_t>
extract_MKN(const std::vector<Tensor> &inputs) {
  size_t M;
  if (inputs.at(0).shape.size() == 2) {
    M = inputs.at(0).shape.at(0);
  } else if (inputs.at(0).shape.size() == 3) { // has batch_dim
    M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(1);
  } else if (inputs.at(0).shape.size() == 4) { // has batch_dim
    M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(1) *
        inputs.at(0).shape.at(2);
  } else {
    throw std::runtime_error("Input Shape is not supported");
  }

  size_t K = inputs.at(1).shape.at(0);
  size_t N = inputs.at(1).shape.at(1);

  return std::make_tuple(M, K, N);
}

/* map from onnx shapes to default shapes accepted by the kernel,
 * calculation tiling (along the m dimension currently) if necessary,
 * we prefer smaller numnber of tiles than larger, and there is no
 * support for combining kernels of different shapes
 *
 * @param M: actual shape M
 * @param K: actual shape K
 * @param N: actual shape N
 *
 * @returns: kernel shape M, K, N, and tiling information*/
template <typename InT, typename WtT, typename OutT>
std::tuple<size_t, size_t, size_t, matrix_shapes>
matmul_tile<InT, WtT, OutT>::map_padded_shape(size_t M, size_t K,
                                              size_t N) const {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<matrix_shapes> &supported_shapes = iter->second;
  size_t Mo = M;
  size_t Ko = K;
  size_t No = N;
  size_t fidx = 0;
  bool f_found = false;
  int64_t num_m_tiles = 1;
  int64_t num_n_tiles = 1;
  bool t_found = false;               // if possibvle to tile
  size_t min_num_tiles = UINTMAX_MAX; // tracking the smallest number of tiles
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    if (M == mat.M && K == mat.K && N == mat.N) {
      fidx = i;
      f_found = true;
      break;
    } else if (((M % mat.M) == 0 || M == 1) && (N % mat.N) == 0 && K == mat.K) {
      // checking if tiling along m / n dimension is possible
      // minimize the number of tiles
      size_t num_tiles = (size_t)(M / mat.M) * (size_t)(N / mat.N);
      if (num_tiles < min_num_tiles) {
        fidx = i;
        num_m_tiles = M / mat.M;
        num_n_tiles = N / mat.N;
        t_found = true;
        min_num_tiles = num_tiles;
      }
    }
  }

  if (!(f_found || t_found)) {
    throw std::runtime_error("Cannot find the shape");
  }

  iter = default_shapes_.find(txn_fname_prefix_);
  const std::vector<matrix_shapes> &actual_shapes = iter->second;
  auto mat = actual_shapes.at(fidx);
  Mo = mat.M;
  Ko = mat.K;
  No = mat.N;
  matrix_shapes tile_info(1, 1, 1);

  if (f_found) {
    return std::make_tuple(Mo, Ko, No, tile_info);
  }

  if (t_found && (M > Mo || N > No)) {
    // if padding is big enough then we don't need tiling
    RYZENAI_LOG_TRACE(
        "M: " + std::to_string(M) + ", K: " + std::to_string(K) +
        ", N: " + std::to_string(N) +
        " mapped to tiling kernel: " + "Mo: " + std::to_string(Mo) +
        ", Ko: " + std::to_string(Ko) + ", No: " + std::to_string(No));
    tile_info.M = num_m_tiles;
    tile_info.N = num_n_tiles;
  }
  return std::make_tuple(Mo, Ko, No, tile_info);
}

/*
 * matmul is an experimental class to offload int8_t * int8_t matrix
 * multiplications to AIE. this class uses lite runtime stack to interface with
 * XRT and submit jobs to IPU. Even though the instructions in this template
 * supports transaction format, it can be extended to support DPU sequence
 * format.
 */

/* Utility function to set the kernel shape based on the weights dimensions */
template <typename InT, typename WtT, typename OutT>
void matmul_tile<InT, WtT, OutT>::set_kernel_shapes() {
  // Use largest M dimension as the default
  //    NOTE: smaller M's can be selected in run_aie if needed
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;

  kernel_x_shape_[1] = w_shape_[0];
  kernel_y_shape_[0] = w_shape_[0];
  kernel_y_shape_[1] = w_shape_[1] / tile_info_.N;
  kernel_z_shape_[1] = w_shape_[1] / tile_info_.N;
}

/*
 * Utility function that setups the instruction registry with transaction
 * binaries.
 */
template <typename InT, typename WtT, typename OutT>
void matmul_tile<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  std::vector<matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    auto key = "gemm_" + get_instr_key(txn_fname_prefix_, mat.M, mat.K, mat.N);
    auto param_key = "gemm_" +
                     get_instr_key(param_fname_prefix_, mat.M, mat.K, mat.N) +
                     "_param";
    instructions.push_back(std::make_pair(key, false));
    layer_params.push_back(std::make_pair(param_key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
  xrt_ctx_->get_registry().add_layer_params(layer_params);
}
template <typename InT, typename WtT, typename OutT>
std::string matmul_tile<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                                       size_t m, size_t k,
                                                       size_t n) const {
  return prefix + "_" + std::to_string(m) + "_" + std::to_string(k) + "_" +
         std::to_string(n);
}

/*
 * matmul class constructor
 *
 * @param kernel_x_shape tuple containing of M x K dimension base matmul
 * supported on IPU
 * @param kernel_y_shape tuple containing of K x N dimension base matmul
 * supported on IPU
 *
 * NOTE: If the input shape has a smaller M dimension than the kernel
 * shape initialized here, the execute function can transparently
 * call a smaller GeMM to reduce padding overhead. The kernel
 * shape passed here should have largest supported M dimension.
 *
 */
template <typename InT, typename WtT, typename OutT>
matmul_tile<InT, WtT, OutT>::matmul_tile(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {

  txnbin_a_header = {{"int8", "a8"}, {"uint8", "a8"}, {"uint16", "a16"}};

  txnbin_b_header = {{"int8", "w8"}, {"uint8", "w8"}};

  txnbin_acc_header = {
      {"uint16", "acc16"}, {"int8", "acc8"}, {"uint8", "acc8"}};

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_["gemm_4x2_a8w8acc8"] = std::vector<matrix_shapes>{};
  default_shapes_["gemm_4x2_a16w8acc16"] = std::vector<matrix_shapes>{};

  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 1152, 1152);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 768, 1152);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 512, 1152);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 768, 768);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 768, 3072);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 3072, 768);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 768, 128);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 768, 128);

  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 1152, 1152);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 768, 1152);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 512, 1152);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 768, 768);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 768, 3072);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 3072, 768);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 768, 128);
  default_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 768, 128);

  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 1152, 1152);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 768, 1152);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 768, 1152);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 512, 1152);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 768, 768);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 768, 3072);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 768, 128);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 3072, 768);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 3072, 768);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 1152, 1152);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 768, 1152);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 512, 1152);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 768, 768);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 768, 3072);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 3072, 768);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 768, 128);

  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(64, 128, 128);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(64, 1024, 1024);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(64, 1024, 3072);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(64, 1024, 4096);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(64, 4096, 1024);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(256, 512, 512);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(256, 512, 1536);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(256, 512, 2048);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(256, 2048, 512);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(832, 256, 256);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(832, 256, 768);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(832, 256, 1024);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(832, 1024, 256);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(3136, 128, 128);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(3136, 128, 384);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(3136, 128, 512);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(3136, 512, 128);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(64, 1024, 768);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 1024, 1024);
  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 4096, 1024);

  default_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 768, 768);

  default_shapes_["gemm_4x4_a16w8acc16"] = std::vector<matrix_shapes>{};
  default_shapes_["gemm_4x4_a16w8acc16"].emplace_back(64, 1280, 10240);
  default_shapes_["gemm_4x4_a16w8acc16"].emplace_back(64, 1280, 1280);
  default_shapes_["gemm_4x4_a16w8acc16"].emplace_back(64, 5120, 1280);
  default_shapes_["gemm_4x4_a16w8acc16"].emplace_back(256, 5120, 1280);
  default_shapes_["gemm_4x4_a16w8acc16"].emplace_back(256, 1280, 10240);
  default_shapes_["gemm_4x4_a16w8acc16"].emplace_back(256, 1280, 1280);
  default_shapes_["gemm_4x4_a16w8acc16"].emplace_back(1024, 640, 5120);
  default_shapes_["gemm_4x4_a16w8acc16"].emplace_back(1024, 2560, 640);
  default_shapes_["gemm_4x4_a16w8acc16"].emplace_back(1024, 640, 640);
  default_shapes_["gemm_4x4_a16w8acc16"].emplace_back(4096, 1280, 320);
  default_shapes_["gemm_4x4_a16w8acc16"].emplace_back(4096, 320, 320);
  default_shapes_["gemm_4x4_a16w8acc16"].emplace_back(4096, 320, 2560);

  // raw shape is the actual shape from ONNX
  raw_shapes_["gemm_4x2_a8w8acc8"] = std::vector<matrix_shapes>{};
  raw_shapes_["gemm_4x2_a16w8acc16"] = std::vector<matrix_shapes>{};

  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 1152, 1152);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 768, 1152);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 512, 1152);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 768, 768);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 768, 3072);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 3072, 768);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 768, 128);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(512, 768, 26);

  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 1152, 1152);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 768, 1152);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 512, 1152);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 768, 768);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 768, 3072);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 3072, 768);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 768, 128);
  raw_shapes_["gemm_4x2_a8w8acc8"].emplace_back(256, 768, 26);

  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 1152, 1152);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(77, 768, 1152);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 768, 1152);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 512, 1152);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 768, 768);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 768, 3072);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 768, 128);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(77, 3072, 768);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(128, 3072, 768);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 1152, 1152);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 768, 1152);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 512, 1152);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 768, 768);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 768, 3072);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 3072, 768);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(512, 768, 128);

  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(49, 128, 128);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(49, 1024, 1024);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(49, 1024, 3072);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(49, 1024, 4096);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(49, 4096, 1024);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(196, 512, 512);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(196, 512, 1536);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(196, 512, 2048);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(196, 2048, 512);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(784, 256, 256);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(784, 256, 768);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(784, 256, 1024);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(784, 1024, 256);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(3136, 128, 128);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(3136, 128, 384);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(3136, 128, 512);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(3136, 512, 128);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(1, 1024, 768);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(77, 1024, 1024);
  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(77, 4096, 1024);

  raw_shapes_["gemm_4x2_a16w8acc16"].emplace_back(77, 768, 768);

  raw_shapes_["gemm_4x4_a16w8acc16"] = std::vector<matrix_shapes>{};
  raw_shapes_["gemm_4x4_a16w8acc16"].emplace_back(64, 1280, 10240);
  raw_shapes_["gemm_4x4_a16w8acc16"].emplace_back(64, 1280, 1280);
  raw_shapes_["gemm_4x4_a16w8acc16"].emplace_back(64, 5120, 1280);
  raw_shapes_["gemm_4x4_a16w8acc16"].emplace_back(256, 5120, 1280);
  raw_shapes_["gemm_4x4_a16w8acc16"].emplace_back(256, 1280, 10240);
  raw_shapes_["gemm_4x4_a16w8acc16"].emplace_back(256, 1280, 1280);
  raw_shapes_["gemm_4x4_a16w8acc16"].emplace_back(1024, 640, 5120);
  raw_shapes_["gemm_4x4_a16w8acc16"].emplace_back(1024, 2560, 640);
  raw_shapes_["gemm_4x4_a16w8acc16"].emplace_back(1024, 640, 640);
  raw_shapes_["gemm_4x4_a16w8acc16"].emplace_back(4096, 1280, 320);
  raw_shapes_["gemm_4x4_a16w8acc16"].emplace_back(4096, 320, 320);
  raw_shapes_["gemm_4x4_a16w8acc16"].emplace_back(4096, 320, 2560);

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  matmul_id_ = matmul_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dd_base_dir() + ryzenai::mdsqr_A8W8_QDQ_XCLBIN_PATH;

  if (a_dtype_ == "uint16") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mxpzi_A16W8_QDQ_XCLBIN_PATH;
  }

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
    RYZENAI_LOG_TRACE("iConv: DesignFormat: " + design_param_);
  }
  txn_fname_prefix_ = "gemm_4x2_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
  param_fname_prefix_ = "gemm_4x2_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    txn_fname_prefix_ = "gemm_4x4_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);
    param_fname_prefix_ = "gemm_4x4_" + txnbin_a_header.at(a_dtype_) +
                          txnbin_b_header.at(b_dtype_) +
                          txnbin_acc_header.at(c_dtype_);

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
        std::cout
            << "Input Shape attribute does not have the expected number of "
               "elements.Number of passed : input_shape_vector.size(), "
               "Expected:4"
            << std::endl;
      }
      RYZENAI_LOG_TRACE(
          "iConv: InputShape: " + std::to_string(input_shape_vector[0]) + ", " +
          std::to_string(input_shape_vector[1]) + ", " +
          std::to_string(input_shape_vector[2]) + ", " +
          std::to_string(input_shape_vector[3]));
    } else {
      std::cout << "Input Shape attribute not found or not of correct type."
                << std::endl;
    }
  }

  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("txn_fname_prefix : {}", txn_fname_prefix_));
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("param_fname_prefix : {}", param_fname_prefix_));

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

  std::call_once(logger_flag_, []() {
    std::string header = "matmul_id M K N kernel_m kernel_k kernel_n Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[GEMM] ID: " + std::to_string(matmul_id_) + ", XCLBIN: " +
                    XCLBIN_FNAME + ", (a_dtype, b_dtype, c_dtype): (" +
                    a_dtype_ + ", " + b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void matmul_tile<InT, WtT, OutT>::set_params(const std::string &model_name,
                                             std::vector<size_t> input_shape) {
  std::string XCLBIN_FNAME;
  if (model_name == "mdsqr") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mdsqr_A8W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "mxpzi") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mxpzi_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "mxgan") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mxgan_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "m3uec") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::m3uec_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "m7h4xjg") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::m7h4xjg_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "mzdk5") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mzdk5_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "4x4mzdk5") {
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                   ryzenai::mzdk54x4_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "mdsqrv1.1") {
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                   ryzenai::mdsqrv1_1_A8W8_QDQ_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }

  auto [M, K, N, tile_info] =
      map_padded_shape(input_shape.at(0), input_shape.at(1), input_shape.at(2));
  KERNEL_M_MAX = M;
  tile_info_ = tile_info;
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

/*
 * copy weight matrix into XRT BOs with padding and tiling
 *
 * this method copies the weight matrix into XRT BOs. This is re-used for
 * every matmul performed for this object with different activations. weight
 * matrix is padded, tiled and reformatted while copying to XRT BOs. padding
 * is done to align with kernel_y_shape each tile of the weight matrix is of
 * shape kernel_y_shape this method also reformats the matrix b/weight matrix
 * as required by AIE/IPU matmul implementation
 *
 * @param weights pointer to the weight matrix
 * @param w_shape tuple containing the shape of the weight matrix
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void matmul_tile<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Matmul initialize_const_params(ptr) ...");

  DD_THROW_IF(
      (const_params.size() != 3) || (const_params.at(0).shape.size() != 2),
      OpsFusion::dd_format("Unsupported const spec for Matmul\n") +
          OpsFusion::dd_format(
              "(Details : #const params == 1 ({}), Const param dim == 2 ({})",
              const_params.size(), const_params.at(0).shape.size()));
  const_params_ = const_params;

  w_shape_[0] = const_params.at(0).shape.at(0);
  w_shape_[1] = const_params.at(0).shape.at(1);

  auto K_raw = w_shape_[0];
  auto N_raw = w_shape_[1];

  auto weights = (WtT *)const_params.at(0).data;
  auto qdq = (int64_t *)const_params.at(1).data;
  auto qdq_params = (int32_t *)const_params.at(2).data;

  auto Ksubv = matmul_matrix::Ksubv;
  auto Msubv = matmul_matrix::Msubv;
  auto Nsubv = matmul_matrix::Nsubv;

  // if the shape of input not known, assume it is 1
  auto [Mo, Ko, No, tile_info] = map_padded_shape(
      1, const_params.at(0).shape.at(0), const_params.at(0).shape.at(1));
  // if tiling parameters are already calculated, use those
  if (tile_info_.M > 1 || tile_info_.N > 1) {
    tile_info = tile_info_;
  }
  tile_info_ = tile_info;

  if (design_param_.find("4x4") != std::string::npos) { // mzdk5 4x4 design
    set_kernel_shapes();
  } else {
    if (w_shape_[0] == 1024 &&
        (w_shape_[1] == 64 ||
         w_shape_[1] % tile_info.N == 64)) { // mzdk5 padding K case
      w_shape_[0] = 1040;
    }
    if (w_shape_[0] == 768 &&
        (w_shape_[1] == 26 ||
         w_shape_[1] % tile_info.N == 26)) { // mdsqr padding N case
      w_shape_[1] = 128 * tile_info.N;
    }
    set_kernel_shapes();
  }

  std::vector<WtT> buf(w_shape_[0] * w_shape_[1], 0);
  if (design_param_.find("4x4") != std::string::npos) { // mzdk5 4x4 design
    size_t M;
    if (inputShape_[2] == inputShape_[3]) {
      M = inputShape_[2] * inputShape_[3];
    } else {
      M = inputShape_[1] * inputShape_[2];
    }

    SUBV_T key = {(int)M, (int)w_shape_[0], (int)w_shape_[1]};
    auto [Mo, Ko, No, tile_info] = map_padded_shape(
        M, const_params.at(0).shape.at(0), const_params.at(0).shape.at(1));
    if (tile_info.M > 1 || tile_info.N > 1) {
      RYZENAI_LOG_TRACE("Input is tiled, tile M " +
                        std::to_string(tile_info.M) + " tile N " +
                        std::to_string(tile_info.N));
      DD_THROW_IF(
          (key.at(0) % tile_info.M != 0 || key.at(1) % tile_info.N != 0),
          "Tiling parameter is not valid. ");
      key[0] = (int)(key[0] / tile_info.M);
      key[1] = (int)(key[1] / tile_info.N);
    }
    auto subv_mode = search_subv_mode(key);

    format_wgt<WtT>(weights, buf.data(), subv_mode, (int)w_shape_[0],
                    (int)w_shape_[1]);
    SUBV_T subv = get_subv(subv_mode);
    Msubv = subv[0];
    Ksubv = subv[1];
    Nsubv = subv[2];
  } else {
    if (w_shape_[0] % Ksubv_mzdk5 == 0) {
      Ksubv = matmul_matrix::Ksubv_mzdk5;
      Msubv = matmul_matrix::Msubv_mzdk5;
      Nsubv = matmul_matrix::Nsubv_mzdk5;
      if (w_shape_[1] > 640) {
        Nsubv = matmul_matrix::Nsubv_mzdk5_LARGE;
      }
    } else {
      if (a_dtype_ == "uint16") {
        Msubv = matmul_matrix::Msubv_16;
      }
    }

    if (w_shape_[0] % Ksubv_mzdk5 == 0) { // mzdk5
      if (w_shape_[1] > 640) {
        matmul_matrix::WgtMatrix<WtT, matmul_matrix::Ksubv_mzdk5,
                                 matmul_matrix::Nsubv_mzdk5_LARGE>
            W((int)w_shape_[0], (int)w_shape_[1], buf.data());
        for (int r = 0; r < w_shape_[0]; ++r) {
          for (int c = 0; c < w_shape_[1]; ++c) {
            W.at(r, c) = weights[(r * w_shape_[1]) + c];
          }
        }
      } else {
        matmul_matrix::WgtMatrix<WtT, matmul_matrix::Ksubv_mzdk5,
                                 matmul_matrix::Nsubv_mzdk5>
            W((int)w_shape_[0], (int)w_shape_[1], buf.data());
        for (int r = 0; r < w_shape_[0]; ++r) {
          for (int c = 0; c < w_shape_[1]; ++c) {
            W.at(r, c) = weights[(r * w_shape_[1]) + c];
          }
        }
      }
    } else {
      matmul_matrix::WgtMatrix<WtT, matmul_matrix::Ksubv, matmul_matrix::Nsubv>
          W((int)w_shape_[0], (int)w_shape_[1], buf.data());
      for (int r = 0; r < K_raw; ++r) {
        for (int c = 0; c < N_raw; ++c) {
          W.at(r, c) = weights[(r * N_raw) + c];
        }
      }
    }
  }

  std::vector<int64_t> qdq_buf(w_shape_[1], 0);
  memcpy((void *)&qdq_buf[0], (void *)&qdq[0], N_raw * sizeof(int64_t));

  // padding Msubv and Nsubv
  qdq_params[qdq_Mv_idx] = Msubv;
  qdq_params[qdq_Nv_idx] = Nsubv;

  auto total_size = Ksubv * Nsubv;
  auto qdq_size = Nsubv * sizeof(int64_t);
  auto qdq_params_size = matmul_matrix::QDQparam_size * sizeof(int32_t);
  auto per_tile_weight_size = w_shape_[0] * w_shape_[1] / tile_info.N;
  auto per_tile_w1_size = w_shape_[1] / tile_info.N;
  //// WGT + Bias(all zeros)
  { // This section of the code interleaves bias with weights Nsubv of bias
    // with every K x N
    size_t write_offset = 0;
    for (int n = 0; n < tile_info_.N; n++) {
      for (int N_shard = 0; N_shard < (w_shape_[1] / tile_info_.N) / (Nsubv);
           N_shard++) {
        for (int K_shard = 0; K_shard < (w_shape_[0]) / (Ksubv); K_shard++) {
          io.write(
              write_offset,
              (void *)&buf[(N_shard * w_shape_[0] * Nsubv) +
                           (K_shard * total_size) + (n * per_tile_weight_size)],
              (total_size));
          write_offset += total_size;
          io.write(write_offset,
                   (void *)&qdq_buf[(N_shard * Nsubv) + (n * per_tile_w1_size)],
                   qdq_size);
          write_offset += qdq_size;
        }
      }
      io.write(write_offset, (void *)qdq_params, qdq_params_size);
      write_offset += qdq_params_size;
    }
  }

  RYZENAI_LOG_TRACE("Matmul initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void matmul_tile<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Matmul initialize_const_params ...");

  DD_THROW_IF(
      (const_params.size() != 3) || (const_params.at(0).shape.size() != 2),
      OpsFusion::dd_format("Unsupported const spec for Matmul\n") +
          OpsFusion::dd_format(
              "(Details : #const params == 1 ({}), Const param dim == 2 ({})",
              const_params.size(), const_params.at(0).shape.size()));
  const_params_ = const_params;
  w_shape_[0] = const_params.at(0).shape.at(0);
  w_shape_[1] = const_params.at(0).shape.at(1);
  int Ksubv;
  if (design_param_.find("4x4") != std::string::npos) { // mzdk5 4x4 design
    set_kernel_shapes();
    size_t M;
    if (inputShape_[2] == inputShape_[3]) {
      M = inputShape_[2] * inputShape_[3];
    } else {
      M = inputShape_[1] * inputShape_[2];
    }

    SUBV_T key = {(int)M, (int)w_shape_[0], (int)w_shape_[1]};
    if (tile_info_.M > 1 || tile_info_.N > 1) {
      RYZENAI_LOG_TRACE("Input is tiled, tile M " +
                        std::to_string(tile_info_.M) + " tile N " +
                        std::to_string(tile_info_.N));
      DD_THROW_IF(
          (key.at(0) % tile_info_.M != 0 || key.at(1) % tile_info_.N != 0),
          "Tiling parameter is not valid. ");
      key[0] = (int)(key[0] / tile_info_.M);
      key[1] = (int)(key[1] / tile_info_.N);
    }
    auto subv_mode = search_subv_mode(key);
    SUBV_T subv = get_subv(subv_mode);
    Ksubv = subv[1];
  } else {
    if (w_shape_[0] == 1024 && w_shape_[1] == 64) { // mtea0a padding K case
      w_shape_[0] = 1040;
    }
    if (w_shape_[0] == 768 && w_shape_[1] == 26) { // mdsqr padding N case
      w_shape_[1] = 128;
    }
    set_kernel_shapes();

    Ksubv = matmul_matrix::Ksubv;
    if (w_shape_[0] % Ksubv_mzdk5 == 0) {
      Ksubv = matmul_matrix::Ksubv_mzdk5;
    }
  }

  // qdqc
  size_t size_interleaved_qdq =
      w_shape_[0] * w_shape_[1] / Ksubv * sizeof(int64_t);

  size_interleaved_qdq +=
      matmul_matrix::QDQparam_size * sizeof(int32_t) * tile_info_.N;

  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  /* Create input/output BOs */
  const size_t B_BO_SIZE =
      (kernel_y_shape_[0] * kernel_y_shape_[1] * b_dtype_size_ * tile_info_.N +
       size_interleaved_qdq);
  const size_t A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_ * tile_info_.M);
  const size_t C_BO_SIZE = (kernel_z_shape_[0] * kernel_z_shape_[1] *
                            c_dtype_size_ * tile_info_.M * tile_info_.N);

  RYZENAI_LOG_TRACE("GEMM: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE) +
                    " a dtype size: " + std::to_string(a_dtype_size_) +
                    " c dtype size: " + std::to_string(c_dtype_size_));
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
  RYZENAI_LOG_TRACE("Matmul initialize_const_params ... DONE");
}
/*
 * execute matrix multiplication c = a * w
 *
 * perform matmul c = a * w. w is stored in the object with initilize_weights
 * method.
 *
 * @param a pointer to activation matrix
 * @param a_shape tuple containing the shape of the activation matrix
 * @param c pointer to store the result of matmul
 *
 * @return none
 */
template <typename InT, typename WtT, typename OutT>
void matmul_tile<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                          std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("Matmul tile execute ...");

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
    throw std::runtime_error("Matmul : Invalid shape received for input");
  }

  if (output.at(0).shape.size() == 4) {
    c_shape_[0] = output.at(0).shape.at(0) * output.at(0).shape.at(1) *
                  output.at(0).shape.at(2);
    c_shape_[1] = output.at(0).shape.at(3);
  } else if (output.at(0).shape.size() == 3) {
    c_shape_[0] = output.at(0).shape.at(0) * output.at(0).shape.at(1);
    c_shape_[1] = output.at(0).shape.at(2);
  } else if (output.at(0).shape.size() == 2) {
    c_shape_[0] = output.at(0).shape.at(0);
    c_shape_[1] = output.at(0).shape.at(1);
  } else {
    throw std::runtime_error("Matmul : Invalid shape received for output");
  }

  if (c_shape_[0] != a_shape_[0]) {
    throw std::runtime_error(
        "Matmul : Input and output matrix row dimentions don't match.");
  }

  RYZENAI_LOG_TRACE("GEMM: a_shape_[0]:" + std::to_string(a_shape_[0]) +
                    " a_shape_[1]:" + std::to_string(a_shape_[1]) +
                    " c_shape_[1]:" + std::to_string(c_shape_[1]));

  auto aie_out = (OutT *)output.at(0).data;
  auto a = (InT *)input.at(0).data;

  auto [M, K, N, tile_info] =
      map_padded_shape(a_shape_[0], a_shape_[1], c_shape_[1]);
  kernel_x_rows = M;
  if (tile_info.M > 1 || tile_info.N > 1) {
    std::string txn_key =
        get_instr_key(txn_fname_prefix_, M * tile_info.M, K, N * tile_info.N);
    {
      std::lock_guard lock(instr_reg_mutex_);
      if (!xrt_ctx_->get_registry().instr_in_registry(txn_key)) {
        RYZENAI_LOG_TRACE(
            "Tiling required, generating tiled transaction binary ...");
        std::vector<Tensor> input_param;
        input_param.push_back(input.at(0));
        input_param.push_back(const_params_.at(0));
        auto txn_bin_vec = get_transaction_bin(input_param, output);
        auto instr = std::make_pair("gemm_" + txn_key, true);
        xrt_ctx_->get_registry().insert_fused_instr_to_instruction_map(
            instr, txn_bin_vec);
      }
    }
  }

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
  auto instr_bo_key = "gemm_" + txn_fname_prefix_ + "_" +
                      std::to_string(kernel_x_rows * tile_info.M) + "_" +
                      std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_y_shape_[1] * tile_info.N);
  auto param_bo_key = "gemm_" + param_fname_prefix_ + "_" +
                      std::to_string(kernel_x_rows) + "_" +
                      std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_y_shape_[1]) + "_param";
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  const xrt::bo &param_bo =
      xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();

  xrt::run run;
  // launch the GEMM kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  // kernel call for GEMM that supports transaction binary flow
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  run = kernel_(2, instr_bo, instr_bo_words,
                c_bo_.address() + DDR_AIE_ADDR_OFFSET,
                a_bo_.address() + DDR_AIE_ADDR_OFFSET,
                b_bo_.address() + DDR_AIE_ADDR_OFFSET,
                param_bo.address() + DDR_AIE_ADDR_OFFSET, 0);
  run.wait2();
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  num_run_aie_++;
  // sync output activation to host memory
  auto c_sync_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  OutT *c_bo_map = c_bo_.map<OutT *>();
  auto c_sync_stop = GET_ELAPSED_TIME_NS();
  c_sync_time_ += static_cast<int64_t>(c_sync_stop - c_sync_start);
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);
  if (0) {
    std::map<std::string, std::any> attr;
    attr["input_shape"] =
        std::vector<int>{1, (int)a_shape_[0], (int)a_shape_[1]};
    attr["orig_output_shape"] =
        std::vector<int>{1, (int)c_shape_[0], (int)c_shape_[1]};
    format_output(output.at(0), c_bo_map, M * N * c_dtype_size_ * tile_info_.N,
                  0, attr);
  } else {
    if (N == c_shape_[1]) {
      memcpy((void *)aie_out, (void *)c_bo_map,
             (c_shape_[0] * c_shape_[1] * c_dtype_size_));
    } else {
      for (int64_t m = 0; m < tile_info.M; m++) {
        for (int64_t n = 0; n < tile_info.N; n++) {
          int64_t output_shape[2];
          output_shape[0] = std::min(c_shape_[0] - kernel_z_shape_[0] * m,
                                     kernel_z_shape_[0]);
          output_shape[1] = std::min(c_shape_[1] - kernel_z_shape_[1] * n,
                                     kernel_z_shape_[1]);
          auto dest_idx =
              m * c_shape_[1] * kernel_z_shape_[0] + n * kernel_z_shape_[1];
          auto src_idx =
              (m * tile_info.N + n) * kernel_z_shape_[0] * kernel_z_shape_[1];
          for (int64_t i = 0; i < output_shape[0]; i++) {
            memcpy((void *)&aie_out[dest_idx + i * c_shape_[1]],
                   (void *)&c_bo_map[src_idx + i * kernel_z_shape_[1]],
                   output_shape[1] * c_dtype_size_);
          }
        }
      }
    }
  }

  auto exec_end = GET_ELAPSED_TIME_NS();
  RYZENAI_LOG_INFO(
      std::to_string(matmul_id_) + " " + std::to_string(a_shape_[0]) + " " +
      std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) + " " +
      std::to_string(kernel_x_rows) + " " + std::to_string(kernel_x_shape_[1]) +
      " " + std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
  RYZENAI_LOG_TRACE("Matmul execute ... DONE");
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
void matmul_tile<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> matmul_tile<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  RYZENAI_LOG_TRACE("Generating transaction bin for matmul tile ... ");
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko, No, tile_info] = map_padded_shape(M, K, N);
  std::string txn_key = "gemm_" + get_instr_key(txn_fname_prefix_, Mo, Ko, No);
  RYZENAI_LOG_TRACE("txn_key: " + txn_key);
  Transaction &txn = Transaction::getInstance();
  std::vector<uint8_t> data = txn.get_txn_bvec(txn_key);

  if (tile_info.M > 1 || tile_info.N > 1) {
    RYZENAI_LOG_TRACE(("Dynamic shape required for matmul, num m tiles: " +
                       std::to_string(tile_info.M)));
    if (tile_info.N > 1) {
      RYZENAI_LOG_TRACE(
          "[Warning] Tiling along n dimension, fusion not supported");
    }
    // basically running get buf req of the kernel
    int Ksubv;
    if (design_param_.find("4x4") != std::string::npos) { // mzdk5 4x4 design
      SUBV_T key = {(int)M, (int)K, (int)N};
      if (tile_info.M > 1 || tile_info.N > 1) {
        RYZENAI_LOG_TRACE("Input is tiled, tile M " +
                          std::to_string(tile_info.M) + " tile N " +
                          std::to_string(tile_info.N));
        DD_THROW_IF(
            (key.at(0) % tile_info.M != 0 || key.at(1) % tile_info.N != 0),
            "Tiling parameter is not valid. ");
        key[0] = key[0] / (int)tile_info.M;
        key[1] = key[1] / (int)tile_info.N;
      }
      auto subv_mode = search_subv_mode(key);
      SUBV_T subv = get_subv(subv_mode);
      Ksubv = subv[1];
    } else {
      Ksubv = matmul_matrix::Ksubv;
      if (Ko % Ksubv_mzdk5 == 0) {
        Ksubv = matmul_matrix::Ksubv_mzdk5;
      }
    }
    size_t size_interleaved_qdq = Ko * No / Ksubv * sizeof(int64_t);
    size_interleaved_qdq += matmul_matrix::QDQparam_size * sizeof(int32_t);
    size_t const_params_bo_size =
        (Ko * No * b_dtype_size_) + size_interleaved_qdq;
    size_t input_bo_size = (Mo * Ko * a_dtype_size_);
    size_t output_bo_size = (Mo * No * c_dtype_size_);
    size_t super_kernel_size = get_super_kernel_params(input, output).size();
    std::vector<OpArgMap> arg_map{
        {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_bo_size},
        {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0, const_params_bo_size},
        {OpArgMap::OpArgType::OUTPUT, 0, 4, 0, output_bo_size},
        {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
         super_kernel_size}}; // buffer req of the kernel
    std::vector<int64_t> tiles = {tile_info.M, tile_info.K, tile_info.N};
    data = matmul_tile_transaction_bin(data, arg_map, tiles);
    RYZENAI_LOG_TRACE(
        OpsFusion::dd_format("Fused Dynamically Shaped : {}", txn_key));
  }
  return data;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> matmul_tile<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko, No, tile_info] = map_padded_shape(M, K, N);
  // TODO: Add check to validate tensor shapes
  std::string param_key =
      "gemm_" + get_instr_key(param_fname_prefix_, Mo, Ko, No) + "_param";
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(param_key);
}

/* Returns buffer requirement for the tiled shape */
template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> matmul_tile<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  // input --> [input, weights, output]
  // Check if IO buffers have batch.
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko, No, tile_info] = map_padded_shape(M, K, N);
  int Ksubv;
  if (design_param_.find("4x4") != std::string::npos) { // mzdk5 4x4 design
    SUBV_T key = {(int)M, (int)K, (int)N};
    if (tile_info.M > 1 || tile_info.N > 1) {
      RYZENAI_LOG_TRACE("Input is tiled, tile M " +
                        std::to_string(tile_info.M) + " tile N " +
                        std::to_string(tile_info.N));
      DD_THROW_IF(
          (key.at(0) % tile_info.M != 0 || key.at(1) % tile_info.N != 0),
          "Tiling parameter is not valid. ");
      key[0] = key[0] / (int)tile_info.M;
      key[1] = key[1] / (int)tile_info.N;
    }
    auto subv_mode = search_subv_mode(key);
    SUBV_T subv = get_subv(subv_mode);
    Ksubv = subv[1];
  } else {
    Ksubv = matmul_matrix::Ksubv;
    if (Ko % Ksubv_mzdk5 == 0) {
      Ksubv = matmul_matrix::Ksubv_mzdk5;
    }
  }

  // qdqc
  size_t size_interleaved_qdq = Ko * No / Ksubv * sizeof(int64_t);
  size_interleaved_qdq += matmul_matrix::QDQparam_size * sizeof(int32_t);
  size_t const_params_bo_size =
      ((Ko * No * b_dtype_size_) + size_interleaved_qdq) * tile_info.N;
  // ignoring tiling in K dimension because that would be 1
  size_t input_bo_size = (Mo * Ko * a_dtype_size_ * tile_info.M);
  size_t output_bo_size = (Mo * No * c_dtype_size_ * tile_info.M * tile_info.N);
  size_t super_kernel_size = get_super_kernel_params(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, 4, 0, output_bo_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size}};
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("Matmul Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
void matmul_tile<InT, WtT, OutT>::format_output(
    const Tensor &out_tensor, void *hw_out_ptr, size_t sz, size_t tensor_idx,
    const std::map<std::string, std::any> &attr) {
  // format_output(
  //     const Tensor &out_tensor, void *hw_out_ptr, size_t sz, int tensor_idx,
  //     const std::map<std::string, std::any> &attr) {
  size_t M, K, N;
  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));
    if (input_shape_vector.size() == 2) {
      M = input_shape_vector[0];
      K = input_shape_vector[1];
    } else if (input_shape_vector.size() == 3) {
      M = input_shape_vector[0] * input_shape_vector[1];
      K = input_shape_vector[2];
    } else if (input_shape_vector.size() == 4) {
      M = input_shape_vector[0] * input_shape_vector[1] * input_shape_vector[2];
      K = input_shape_vector[3];
    } else {
      std::cout << "Input shape attribute does not have the expected number of "
                   "elements.Number of passed : design_param_vector.size(), "
                   "Expected:3"
                << std::endl;
    }
    RYZENAI_LOG_TRACE("Matmul: input_shape: " + std::to_string(M) + ", " +
                      std::to_string(K));
  } else {
    std::cout << "Input Shape attribute not found or not of correct type."
              << std::endl;
  }

  if (attr.count("output_shape") &&
      attr.at("output_shape").type() == typeid(std::vector<int>)) {
    const auto &orig_output_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("output_shape"));
    if (orig_output_shape_vector.size() == 2) {
      N = orig_output_shape_vector[1];
    } else if (orig_output_shape_vector.size() == 3) {
      N = orig_output_shape_vector[2];
    } else if (orig_output_shape_vector.size() == 4) {
      N = orig_output_shape_vector[3];
    } else {
      std::cout
          << "output shape attribute does not have the expected number of "
             "elements.Number of passed : design_param_vector.size(), "
             "Expected:3"
          << std::endl;
    }
    RYZENAI_LOG_TRACE("Matmul: output_shape: " + std::to_string(M) + ", " +
                      std::to_string(N));
  } else {
    N = out_tensor.shape.at(2);
  }
  // get the mapped shape
  auto [Mo, Ko, No, tile_info] = map_padded_shape(M, K, N);
  // K, N is the dst.shape
  auto aie_out = (void *)out_tensor.data;

  if (sz != Mo * No * c_dtype_size_ * tile_info.N) {
    throw std::runtime_error("Matmul : The size of hw_out is not correct.");
  }

  if (N == No) {
    RYZENAI_LOG_TRACE("Triggering Matmul Output Memcpy");
    memcpy((void *)aie_out, (void *)hw_out_ptr, (M * No * c_dtype_size_));
  } else {
    RYZENAI_LOG_TRACE("Triggering Matmul Output Strided Memcpy");
    for (int64_t m = 0; m < tile_info.M; m++) {
      for (int64_t n = 0; n < tile_info.N; n++) {
        int64_t output_shape[2];
        output_shape[0] = std::min((int64_t)(K - Ko * m), (int64_t)Ko);
        output_shape[1] = std::min((int64_t)(N - No * n), (int64_t)No);
        auto dest_idx = m * N * Ko + n * No;
        auto src_idx = (m * tile_info.N + n) * Ko * No;

        for (int64_t i = 0; i < output_shape[0]; i++) {
          memcpy((void *)(static_cast<uint8_t *>(aie_out) +
                          (dest_idx + i * N) * c_dtype_size_),
                 (void *)(static_cast<uint8_t *>(hw_out_ptr) +
                          (src_idx + i * No) * c_dtype_size_),
                 output_shape[1] * c_dtype_size_);
        }
      }
    }
  }
}

template <typename InT, typename WtT, typename OutT>
std::once_flag matmul_tile<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t matmul_tile<InT, WtT, OutT>::matmul_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag matmul_tile<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
std::mutex matmul_tile<InT, WtT, OutT>::instr_reg_mutex_;

template class matmul_tile<uint8_t, uint8_t, uint8_t>;
template class matmul_tile<uint16_t, uint8_t, uint16_t>;

} // namespace ryzenai
