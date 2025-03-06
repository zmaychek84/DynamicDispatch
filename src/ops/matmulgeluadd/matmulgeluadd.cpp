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

#include <txn_container.hpp>
#include <utils/instruction_registry.hpp>
#include <xrt_context/xrt_context.hpp>

#include "utils/ctrl_pkt_utils.hpp"

#include <ops/matmulgeluadd/matmulgeluadd.hpp>
#include <ops/op_interface.hpp>
#include <ops/ops_common/coeffs.hpp>
#include <ops/ops_common/ctrlpkt.hpp>
#include <ops/ops_common/op_util.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include "xaiengine.h"

// Headers for BFP matrix formatting
#include "ops/ops_common/gelu_lut_bf16_512.h"
#include "ops/ops_common/matmul_matrix.hpp"
using namespace matmul_matrix;
namespace ryzenai {

static std::tuple<size_t, size_t, size_t>
extract_MKN(const std::vector<Tensor> &inputs) {
  // inputs[0] --> input
  // inputs[1] --> wts

  size_t M;
  if (inputs.at(0).shape.size() == 2) {
    M = inputs.at(0).shape.at(0);
  } else if (inputs.at(0).shape.size() == 3) { // has batch_dim
    M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(1);
  } else {
    throw std::runtime_error("Input Shape is not supported");
  }
  size_t K = inputs.at(1).shape.at(0);
  size_t N = inputs.at(1).shape.at(1);

  return std::make_tuple(M, K, N);
}

template <typename InT, typename WtT, typename OutT>
std::tuple<size_t, size_t>
matmulgeluadd<InT, WtT, OutT>::map_padded_shape(size_t M, size_t N) const {
  auto iter = raw_shapes_.find(txn_fname_prefix_);
  const std::vector<matrix_shapes> &supported_shapes = iter->second;
  size_t Mo = M;
  size_t No = N;
  size_t fidx = 0;
  bool f_found = false;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);
    if (M == mat.M && N == mat.K) {
      fidx = i;
      f_found = true;
      break;
    }
  }
  if (f_found) {
    auto iter = default_shapes_.find(txn_fname_prefix_);
    const std::vector<matrix_shapes> &actual_shapes = iter->second;
    auto mat = actual_shapes.at(fidx);
    Mo = mat.M;
    No = mat.K;
  } else {
    throw std::runtime_error("Cannot find the shape");
  }
  return std::make_tuple(Mo, No);
}

template <typename InT, typename WtT, typename OutT>
void matmulgeluadd<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
std::string matmulgeluadd<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                                         size_t m, size_t k,
                                                         size_t n) const {
  auto instr_key = prefix + "_" + std::to_string(m) + "_" + std::to_string(k) +
                   "_" + std::to_string(n);
  return instr_key;
}

template <typename InT, typename WtT, typename OutT>
void matmulgeluadd<InT, WtT, OutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  // GEMM
  // txn_fname_prefix_ = "gemmgelu_" + txnbin_a_header.at(a_dtype_) +
  //                    txnbin_b_header.at(b_dtype_) +
  //                    txnbin_acc_header.at(c_dtype_);
  // param_fname_prefix_ = "gemmgelu_" + txnbin_a_header.at(a_dtype_) +
  //                      txnbin_b_header.at(b_dtype_) +
  //                      txnbin_acc_header.at(c_dtype_);
  std::vector<matrix_shapes> supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);

    auto key =
        "gemmgelu_" + get_instr_key(txn_fname_prefix_, mat.M, mat.K, mat.N);
    auto param_key = "gemmgelu_" +
                     get_instr_key(param_fname_prefix_, mat.M, mat.K, mat.N) +
                     "_param";

    instructions.push_back(std::make_pair(key, false));
    layer_params.push_back(std::make_pair(param_key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
  xrt_ctx_->get_registry().add_layer_params(layer_params);
}

template <typename InT, typename WtT, typename OutT>
matmulgeluadd<InT, WtT, OutT>::matmulgeluadd(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {

  is_generic_pass_in_onnx = OpsFusion::check_generic_fusion(attr);

  txnbin_a_header = {{"uint16", "a16"}, {"uint8", "a8"}};

  txnbin_b_header = {{"int8", "w8"}, {"uint8", "w8"}};

  txnbin_acc_header = {
      {"int32", "acc32"}, {"uint16", "acc16"}, {"uint8", "acc8"}};

  // default shape is the padded shaped used in AIE for BO allocation
  default_shapes_["gemmgelu_a8w8acc8"] = std::vector<matrix_shapes>();
  default_shapes_["gemmgelu_a16w8acc16"] = std::vector<matrix_shapes>();
  default_shapes_["gemmgelu_4x4_a16w8acc16"] = std::vector<matrix_shapes>();

  default_shapes_["gemmgelu_a8w8acc8"].emplace_back(512, 768, 3072);
  default_shapes_["gemmgelu_a8w8acc8"].emplace_back(256, 768, 3072);

  default_shapes_["gemmgelu_a16w8acc16"].emplace_back(128, 768, 3072);
  default_shapes_["gemmgelu_a16w8acc16"].emplace_back(128, 768, 3072);
  default_shapes_["gemmgelu_a16w8acc16"].emplace_back(512, 768, 3072);

  default_shapes_["gemmgelu_a16w8acc16"].emplace_back(64, 128, 128);
  default_shapes_["gemmgelu_a16w8acc16"].emplace_back(64, 1024, 4096);
  default_shapes_["gemmgelu_a16w8acc16"].emplace_back(256, 512, 2048);
  default_shapes_["gemmgelu_a16w8acc16"].emplace_back(832, 256, 1024);
  default_shapes_["gemmgelu_a16w8acc16"].emplace_back(3136, 128, 512);
  default_shapes_["gemmgelu_a16w8acc16"].emplace_back(128, 1024, 4096);
  default_shapes_["gemmgelu_4x4_a16w8acc16"].emplace_back(64, 768, 3072);

  // raw shape is the actual shape from ONNX
  raw_shapes_["gemmgelu_a8w8acc8"] = std::vector<matrix_shapes>();
  raw_shapes_["gemmgelu_a16w8acc16"] = std::vector<matrix_shapes>();
  raw_shapes_["gemmgelu_4x4_a16w8acc16"] = std::vector<matrix_shapes>();

  raw_shapes_["gemmgelu_a8w8acc8"].emplace_back(512, 768, 3072);
  raw_shapes_["gemmgelu_a8w8acc8"].emplace_back(256, 768, 3072);

  raw_shapes_["gemmgelu_a16w8acc16"].emplace_back(77, 768, 3072);
  raw_shapes_["gemmgelu_a16w8acc16"].emplace_back(128, 768, 3072);
  raw_shapes_["gemmgelu_a16w8acc16"].emplace_back(512, 768, 3072);

  raw_shapes_["gemmgelu_a16w8acc16"].emplace_back(49, 128, 128);
  raw_shapes_["gemmgelu_a16w8acc16"].emplace_back(49, 1024, 4096);
  raw_shapes_["gemmgelu_a16w8acc16"].emplace_back(196, 512, 2048);
  raw_shapes_["gemmgelu_a16w8acc16"].emplace_back(784, 256, 1024);
  raw_shapes_["gemmgelu_a16w8acc16"].emplace_back(3136, 128, 512);
  raw_shapes_["gemmgelu_a16w8acc16"].emplace_back(77, 1024, 4096);
  raw_shapes_["gemmgelu_4x4_a16w8acc16"].emplace_back(64, 768, 3072);

  DPU_DIR = OpInterface::get_dd_base_dir() + "//transaction//" + "stx";

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  matmulgeluadd_id_ = matmulgeluadd_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dd_base_dir() + ryzenai::mdsqr_A8W8_QDQ_XCLBIN_PATH;

  if (a_dtype_ == "uint16") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mxpzi_A16W8_QDQ_XCLBIN_PATH;
  }

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

  txn_fname_prefix_ = "gemmgelu_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
  param_fname_prefix_ = "gemmgelu_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    txn_fname_prefix_ = "gemmgelu_4x4_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);

    param_fname_prefix_ = "gemmgelu_4x4_" + txnbin_a_header.at(a_dtype_) +
                          txnbin_b_header.at(b_dtype_) +
                          txnbin_acc_header.at(c_dtype_);
  }

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }
  KERNEL_M_MAX = 512;

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
        "matmulgeluadd_id M K N kernel_m kernel_k kernel_n Execute"
        "time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[OP] ID: " + std::to_string(matmulgeluadd_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype_ + ", " +
                    b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void matmulgeluadd<InT, WtT, OutT>::set_params(
    const std::string &model_name, std::vector<size_t> input_shape) {
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
  } else if (model_name == "mdsqrv1.1") {
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                   ryzenai::mdsqrv1_1_A8W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "mxganv1.2") {
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                   ryzenai::mxganv1_2_A16W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "4x4PSW1.0") {
    is_ctrl_pkt_ = 1;
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::PSW1_0_A16W8_QDQ_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }

  auto [M, K] = map_padded_shape(input_shape.at(0), input_shape.at(1));
  KERNEL_M_MAX = M;

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void matmulgeluadd<InT, WtT, OutT>::set_kernel_shapes() {
  // Use largest M dimension as the default
  //    NOTE: smaller M's can be selected in run_aie if needed
  RYZENAI_LOG_TRACE("GEMM_GELU: w_shape0:" + std::to_string(w_shape_[0]) +
                    " w_shape1:" + std::to_string(w_shape_[1]));
  kernel_x_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;

  kernel_x_shape_[1] = w_shape_[0];
  kernel_y_shape_[0] = w_shape_[0];
  kernel_y_shape_[1] = w_shape_[1];
  kernel_z_shape_[1] = w_shape_[1];
}

template <typename InT, typename WtT, typename OutT>
void matmulgeluadd<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("Matmulgelu initialize_const_params(ptr) ...");

  const int w_idx = 0, qdq_idx = 1, qdq_param_idx = 2, gelu_qdq_param_idx = 3;
  // The first data is Weight
  auto weights = (WtT *)const_params.at(w_idx).data;
  std::vector<size_t> shape = const_params.at(w_idx).shape;
  w_shape_[0] = shape[0];
  w_shape_[1] = shape[1];
  set_kernel_shapes();

  gelu_coeffs = std::vector<int32_t>(16, 0);
  int64_t *qdq;
  int32_t *qdq_params;
  int32_t *gelu_qdq_params;

  if (is_generic_pass_in_onnx) {
    // Extract
    //"a_s", "a_z", "w_s", "w_z", "q1_s", "q1_z",  "b_s", "b_z", "q2_s", "q2_z",
    //"q3_s", "q3_z"
    float a_s = OpsFusion::get_tensor_as_float_vec(const_params.at(2))[0];
    uint16_t a_z = OpsFusion::get_tensor_as_uint16_t_vec(const_params.at(3))[0];

    float w_s = OpsFusion::get_tensor_as_float_vec(const_params.at(4))[0];
    uint16_t w_z = OpsFusion::get_tensor_as_uint16_t_vec(const_params.at(5))[0];

    float b_s = OpsFusion::get_tensor_as_float_vec(const_params.at(6))[0];
    uint16_t b_z = OpsFusion::get_tensor_as_uint16_t_vec(const_params.at(7))[0];

    float q1_s = OpsFusion::get_tensor_as_float_vec(const_params.at(8))[0];
    uint16_t q1_z =
        OpsFusion::get_tensor_as_uint16_t_vec(const_params.at(9))[0];

    float q2_s = OpsFusion::get_tensor_as_float_vec(const_params.at(10))[0];
    uint16_t q2_z =
        OpsFusion::get_tensor_as_uint16_t_vec(const_params.at(11))[0];

    float q3_s = OpsFusion::get_tensor_as_float_vec(const_params.at(12))[0];
    uint16_t q3_z =
        OpsFusion::get_tensor_as_uint16_t_vec(const_params.at(13))[0];

    auto w_data = OpsFusion::fold2D<uint8_t>(const_params.at(0));

    auto b_data = OpsFusion::get_tensor_as_uint16_t_vec(const_params.at(1));

    const auto &out_dtype =
        std::any_cast<const std::vector<std::string> &>(attr.at("out_dtypes"));

    if (out_dtype[0] == "uint8") {
      _qdq_params =
          OpsFusion::coeffs::calculate_matmuladd_qdq_params_uint8_uint8(
              w_data, b_data, a_s, a_z, w_s, w_z, b_s, b_z, q2_s, q2_z);
    } else if (out_dtype[0] == "uint16") {
      _qdq_params =
          OpsFusion::coeffs::calculate_matmuladd_qdq_params_uint16_uint8(
              w_data, b_data, a_s, a_z, w_s, w_z, b_s, b_z, q2_s, q2_z);
    } else {
      RYZENAI_LOG_TRACE("Unknown Data Type");
    }

    qdq = _qdq_params.c0_coeffs.data();
    qdq_params = _qdq_params.qdq_params.data();

    auto [c0_sc_a, c0_zp_a, c0_sc_b, c0_zp_b] =
        OpsFusion::coeffs::calc_eltwise_coeff(q2_s, q2_z, (float)1.0 / q3_s,
                                              q3_z);

    // GELU CALC
    gelu_coeffs[0] = c0_zp_a;
    gelu_coeffs[1] = c0_sc_a;
    gelu_coeffs[2] = c0_zp_b;
    gelu_coeffs[3] = c0_sc_b;
    gelu_coeffs[4] = gelu_coeffs[4] = (out_dtype[0] == "uint8") ? 0 : 1;

    gelu_qdq_params = gelu_coeffs.data();
  } else {
    DD_THROW_IF(
        (const_params.size() != 4) || (const_params.at(0).shape.size() != 2) ||
            (const_params.at(1).shape.size() != 1),
        OpsFusion::dd_format("Unsupported const spec for Matmulgelu\n") +
            OpsFusion::dd_format("(Details : #const params == 2 ({}), Const "
                                 "param1 dim == 2 ({}), "
                                 "Const param2 dim == 1 ({})",
                                 const_params.size(),
                                 const_params.at(0).shape.size(),
                                 const_params.at(1).shape.size()));

    qdq = (int64_t *)const_params.at(qdq_idx).data;
    qdq_params = (int32_t *)const_params.at(qdq_param_idx).data;
    gelu_qdq_params = (int32_t *)const_params.at(gelu_qdq_param_idx).data;
  }

  int const size_lutab = sizeof(lnr_lutab);
  int const size_lutcd = sizeof(lnr_lutcd);

  if (a_dtype_ == "int8" || a_dtype_ == "uint8") {
    qdq_params[qdq_Mv_idx] = matmul_matrix::Msubv;
  } else {
    qdq_params[qdq_Mv_idx] = matmul_matrix::Msubv_16;
  }
  qdq_params[qdq_Nv_idx] = matmul_matrix::Nsubv;

  size_t write_offset = 0;
  std::vector<WtT> buf(w_shape_[0] * w_shape_[1]);
  if ((design_param_.find("4x4") != std::string::npos)) { // PSW 1.0 4x4 design
    qdq_params[qdq_Mv_idx] = matmul_matrix::Msubv_PSW;
    qdq_params[qdq_Nv_idx] = matmul_matrix::Nsubv_PSW_GeMM_GeLU;

    matmul_matrix::WgtMatrix<WtT> W(
        (int)w_shape_[0], (int)w_shape_[1], matmul_matrix::Ksubv_PSW,
        matmul_matrix::Nsubv_PSW_GeMM_GeLU, buf.data());
    for (int r = 0; r < w_shape_[0]; ++r) {
      for (int c = 0; c < w_shape_[1]; ++c) {
        W.at(r, c) = weights[(r * w_shape_[1]) + c];
      }
    }
  } else {
    matmul_matrix::WgtMatrix<WtT> W((int)w_shape_[0], (int)w_shape_[1], Ksubv,
                                    Nsubv, buf.data());
    for (int r = 0; r < w_shape_[0]; ++r) {
      for (int c = 0; c < w_shape_[1]; ++c) {
        W.at(r, c) = weights[(r * w_shape_[1]) + c];
      }
    }
  }

  /* This section of the code interleaves bias with weights Nsubv of bias
     with every K x N */
  if ((design_param_.find("4x4") != std::string::npos)) { // PSW 1.0 4x4 design
    auto total_size =
        matmul_matrix::Ksubv_PSW * matmul_matrix::Nsubv_PSW_GeMM_GeLU;
    auto qdq_size = matmul_matrix::Nsubv_PSW_GeMM_GeLU * sizeof(int64_t);
    auto qdq_params_size = matmul_matrix::QDQparam_size * sizeof(int32_t);

    for (int N_shard = 0;
         N_shard < (w_shape_[1]) / (matmul_matrix::Nsubv_PSW_GeMM_GeLU);
         N_shard++) {
      for (int K_shard = 0;
           K_shard < (w_shape_[0]) / (matmul_matrix::Ksubv_PSW); K_shard++) {
        io.write(write_offset,
                 (void *)&buf[(N_shard * w_shape_[0] *
                               matmul_matrix::Nsubv_PSW_GeMM_GeLU) +
                              (K_shard * total_size)],
                 (total_size));
        write_offset += total_size;
        io.write(write_offset,
                 (void *)&qdq[N_shard * matmul_matrix::Nsubv_PSW_GeMM_GeLU],
                 qdq_size);
        write_offset += qdq_size;
      }
    }
    io.write(write_offset, (void *)qdq_params, qdq_params_size);
    write_offset += qdq_params_size;

    io.write(write_offset, (void *)gelu_qdq_params, qdq_params_size);
    write_offset += qdq_params_size;
  } else {
    auto total_size = matmul_matrix::Ksubv * matmul_matrix::Nsubv;
    auto qdq_size = matmul_matrix::Nsubv * sizeof(int64_t);
    auto qdq_params_size = matmul_matrix::QDQparam_size * sizeof(int32_t);

    for (int N_shard = 0; N_shard < (w_shape_[1]) / (matmul_matrix::Nsubv);
         N_shard++) {
      for (int K_shard = 0; K_shard < (w_shape_[0]) / (matmul_matrix::Ksubv);
           K_shard++) {
        io.write(write_offset,
                 (void *)&buf[(N_shard * w_shape_[0] * matmul_matrix::Nsubv) +
                              (K_shard * total_size)],
                 (total_size));
        write_offset += total_size;
        io.write(write_offset, (void *)&qdq[N_shard * matmul_matrix::Nsubv],
                 qdq_size);
        write_offset += qdq_size;
      }
    }
    io.write(write_offset, (void *)qdq_params, qdq_params_size);
    write_offset += qdq_params_size;

    io.write(write_offset, (void *)gelu_qdq_params, qdq_params_size);
    write_offset += qdq_params_size;
  }

  io.write(write_offset, (void *)lnr_lutab, size_lutab);
  io.write(write_offset + size_lutab, (void *)lnr_lutcd, size_lutcd);

  RYZENAI_LOG_TRACE("Matmulgelu initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
std::vector<uint8_t> matmulgeluadd<InT, WtT, OutT>::get_ctrl_pkts(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko] = map_padded_shape(M, K);
  // TODO: Add check to validate tensor shapes
  std::string ctrl_key =
      "gemmgelu_" + get_instr_key(param_fname_prefix_, Mo, Ko, N) + "_ctrl";
  try {
    Transaction &txn = Transaction::getInstance();
    return txn.get_txn_bvec(ctrl_key);
  } catch (...) {
    return {};
  }
}

template <typename InT, typename WtT, typename OutT>
std::vector<CtrlPktPatchInfo>
matmulgeluadd<InT, WtT, OutT>::get_ctrl_pkt_patch_info(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko] = map_padded_shape(M, K);
  // TODO: Add check to validate tensor shapes
  try {
    auto ctrl_pkt_meta = "gemmgelu_" +
                         get_instr_key(param_fname_prefix_, Mo, Ko, N) +
                         "_ctrl_meta";
    Transaction &txn = Transaction::getInstance();
    return json_str_to_ctrlpkt_patch_info(txn.get_txn_bvec(ctrl_pkt_meta));
  } catch (...) {
    // throw std::runtime_error("elwadd : Can not file the ctrl_meta.json
    // file");
    return {};
  }
}

// For MATMULGELU: weight + bias + lutab + lutcd
template <typename InT, typename WtT, typename OutT>
void matmulgeluadd<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  // Check the number of inputs
  if (const_params.size() != 4) {
    throw std::runtime_error("MATMULGELU expect to have four constant.");
  }
  is_generic_pass_in_onnx = OpsFusion::check_generic_fusion(attr);
  const int w_idx = 0, qdq_idx = 1;
  // The first data is Weight
  // auto weight = (int8_t*)const_params.at(w_idx).data;
  std::vector<size_t> shape = const_params.at(w_idx).shape;
  size_t size_weight = shape[0] * shape[1] * b_dtype_size_;
  w_shape_[0] = shape[0];
  w_shape_[1] = shape[1];

  shape = const_params.at(qdq_idx).shape;
  size_t size_interleaved_qdq;
  if ((design_param_.find("4x4") != std::string::npos)) { // PSW 1.0 4x4 design
    size_interleaved_qdq =
        w_shape_[0] * w_shape_[1] / matmul_matrix::Ksubv_PSW * sizeof(int64_t);
  } else {
    size_interleaved_qdq =
        w_shape_[0] * w_shape_[1] / matmul_matrix::Ksubv * sizeof(int64_t);
  }
  size_interleaved_qdq += 2 * matmul_matrix::QDQparam_size * sizeof(int32_t);

  const size_t size_lutab = sizeof(lnr_lutab);
  const size_t size_lutcd = sizeof(lnr_lutcd);

  // Init the BO size
  set_kernel_shapes();

  // Create input/output BOs
  const size_t A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_);
  const size_t B_BO_SIZE =
      size_weight + size_interleaved_qdq + size_lutab + size_lutcd;
  // (kernel_y_shape_[0] * kernel_y_shape_[1] * b_dtype_size_);
  const size_t C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_);
  RYZENAI_LOG_TRACE("GEMM_GELU: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE:" + std::to_string(C_BO_SIZE));
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));

  // std::cout << "a bo size: " << A_BO_SIZE << std::endl;
  // std::cout << "b bo size: " << B_BO_SIZE << std::endl;
  // std::cout << "c bo size: " << C_BO_SIZE << std::endl;

  // copy b_bo
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  auto b_copy_start = GET_ELAPSED_TIME_NS();
  auto b_format_start = GET_ELAPSED_TIME_NS();
  WtT *b_bo_map = b_bo_.map<WtT *>();
  auto bo_const = BoConst(b_bo_map);
  initialize_const_params(bo_const, const_params);
  auto b_format_stop = GET_ELAPSED_TIME_NS();
  auto b_copy_stop = GET_ELAPSED_TIME_NS();
  b_format_time_ += static_cast<int64_t>(b_format_stop - b_format_start);
  b_copy_time_ = static_cast<int64_t>(b_copy_stop - b_copy_start);

  // sync b_bo
  auto b_sync_start = GET_ELAPSED_TIME_NS();
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto b_sync_stop = GET_ELAPSED_TIME_NS();
  b_sync_time_ = static_cast<int64_t>(b_sync_stop - b_sync_start);

  if (is_ctrl_pkt_) {
    // Based on the mapped_shape to get the meta json file
    auto Mo = (size_t)kernel_x_shape_[0];
    auto Ko = (size_t)w_shape_[0];
    auto No = (size_t)w_shape_[1];

    std::vector<uint8_t> json_data;
    try {
      auto json_key = "gemmgelu_" +
                      get_instr_key(param_fname_prefix_, Mo, Ko, No) +
                      "_ctrl_meta";
      Transaction &txn = Transaction::getInstance();
      json_data = txn.get_txn_bvec(json_key);
    } catch (...) {
      is_ctrl_pkt_ = 0;
    }

    if (is_ctrl_pkt_) {
      std::cout << "ctrlpkt patching" << std::endl;
      RYZENAI_LOG_TRACE("elwadd patch ctrlpkt ... START");
      // get param_bo address
      auto param_bo_key = "gemmgelu_" +
                          get_instr_key(param_fname_prefix_, Mo, Ko, No) +
                          "_param";
      const xrt::bo &param_bo =
          xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;

      // Get ctrl pkt patch info from json
      std::vector<CtrlPktPatchInfo> ctrlpkt_info;
      ctrlpkt_info = json_str_to_ctrlpkt_patch_info(json_data);

      // Get the ctrl pkt
      auto ctrl_bo_key = "gemmgelu_" +
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
      RYZENAI_LOG_TRACE("matmulgeluadd patch ctrlpkt ... DONE");
    }
  }
}

// matmulgelu
template <typename InT, typename WtT, typename OutT>
void matmulgeluadd<InT, WtT, OutT>::execute(const std::vector<Tensor> &input,
                                            std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 1) {
    throw std::runtime_error("MATMULGELU expect to have one input.");
  }
  const int a_idx = 0;
  // The first data is a
  InT *a = (InT *)input.at(a_idx).data;

  a_copy_time_ = 0;
  a_sync_time_ = 0;
  b_copy_time_ = 0;
  b_format_time_ = 0;
  b_sync_time_ = 0;
  c_copy_time_ = 0;
  c_sync_time_ = 0;
  run_aie_time_ = 0;
  cpu_acc_time_ = 0;

  auto exec_start = GET_ELAPSED_TIME_NS();

  a_shape_[0] = input.at(a_idx).shape.at(0);
  a_shape_[1] = input.at(a_idx).shape.at(1);

  c_shape_[0] = a_shape_[0];
  c_shape_[1] = w_shape_[1];

  auto [M, K] = map_padded_shape(a_shape_[0], a_shape_[1]);
  kernel_x_rows = M;

  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  size_t a_size = a_shape_[0] * a_shape_[1] * sizeof(InT);
  memcpy((void *)a_bo_map, (void *)a, a_size);
  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  // prepare inst_bo and param_bo
  auto instr_bo_key = "gemmgelu_" + txn_fname_prefix_ + "_" +
                      std::to_string(kernel_x_rows) + "_" +
                      std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_y_shape_[1]);
  auto param_bo_key = "gemmgelu_" + param_fname_prefix_ + "_" +
                      std::to_string(kernel_x_rows) + "_" +
                      std::to_string(kernel_x_shape_[1]) + "_" +
                      std::to_string(kernel_y_shape_[1]) + "_param";

  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  const xrt::bo &param_bo =
      xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();

  // launch the kernel

  auto run_aie_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  ryzenai::dynamic_dispatch::execute_kernel(
      kernel_, 2, instr_bo, instr_bo_words, c_bo_, a_bo_, b_bo_, param_bo,
      ctrl_bo_, true, is_ctrl_pkt_);
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
  memcpy((void *)aie_out, (void *)c_bo_map,
         c_shape_[0] * c_shape_[1] * sizeof(OutT));
  auto c_copy_stop = GET_ELAPSED_TIME_NS();
  c_copy_time_ = static_cast<int64_t>(c_copy_stop - c_copy_start);
  auto exec_end = GET_ELAPSED_TIME_NS();

  RYZENAI_LOG_INFO(
      std::to_string(matmulgeluadd_id_) + " " + std::to_string(a_shape_[0]) +
      " " + std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) +
      " " + std::to_string(kernel_x_rows) + " " +
      std::to_string(kernel_x_shape_[1]) + " " +
      std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> matmulgeluadd<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko] = map_padded_shape(M, K);
  std::string txn_key =
      "gemmgelu_" + get_instr_key(txn_fname_prefix_, Mo, Ko, N);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t>
matmulgeluadd<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko] = map_padded_shape(M, K);
  // TODO: Add check to validate tensor shapes
  std::string param_key =
      "gemmgelu_" + get_instr_key(param_fname_prefix_, Mo, Ko, N) + "_param";
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(param_key);
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> matmulgeluadd<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  // input --> [input, weights, bias, output]

  // TODO following check
  //  if (input.size() != (is_generic_pass_in_onnx ? 16 : 6)) {
  //    throw std::runtime_error(
  //        "MATMULGELUADD : Incorrect number of tensors received");
  //  }
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko] = map_padded_shape(M, K);

  size_t output_idx = is_generic_pass_in_onnx ? 15 : 5;
  int Ksubv;
  if (design_param_.find("4x4") != std::string::npos) { // PSW 1.0 4x4 design
    Ksubv = matmul_matrix::Ksubv_PSW;
  } else { // 4x2 design
    Ksubv = matmul_matrix::Ksubv;
  }

  size_t size_interleaved_qdq = Ko * N / matmul_matrix::Ksubv * sizeof(int64_t);
  size_interleaved_qdq += 2 * matmul_matrix::QDQparam_size * sizeof(int32_t);

  size_t const size_lutab = sizeof(lnr_lutab);
  size_t const size_lutcd = sizeof(lnr_lutcd);

  size_t B_BO_SIZE =
      (Ko * N * sizeof(WtT) + size_interleaved_qdq * sizeof(InT) + size_lutab +
       size_lutcd);
  size_t A_BO_SIZE = (Mo * Ko * sizeof(InT));
  size_t C_BO_SIZE = (Mo * N * sizeof(OutT));
  size_t super_kernel_size = get_super_kernel_params(input, output).size();
  size_t ctrl_pkt_size = get_ctrl_pkts(input, output).size();

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, A_BO_SIZE},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0, B_BO_SIZE},
      {OpArgMap::OpArgType::OUTPUT, 0, output_idx, 0, C_BO_SIZE},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size},
      {OpArgMap::OpArgType::CTRL_PKT_BIN, 4, 0, 0, ctrl_pkt_size}};
  return arg_map;
}

template <typename InT, typename WtT, typename OutT>
std::once_flag matmulgeluadd<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t matmulgeluadd<InT, WtT, OutT>::matmulgeluadd_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag matmulgeluadd<InT, WtT, OutT>::instr_reg_flag_;

template class matmulgeluadd<uint8_t, uint8_t, uint8_t>;
template class matmulgeluadd<uint16_t, uint8_t, uint16_t>;

} // namespace ryzenai
