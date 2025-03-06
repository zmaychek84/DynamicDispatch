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
#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
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

#include <ops/mhagprb/mhagprb.hpp>
#include <ops/op_interface.hpp>
#include <ops/ops_common/coeffs.hpp>
#include <ops/ops_common/op_util.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>

// AIE Driver header
#include "xaiengine.h"

#include "ops/ops_common/mhagprb_matrix.hpp"
#include "ops/ops_common/sigmoid_lut_512.h"

namespace ryzenai {

int64_t tensor_to_int64(Tensor t, size_t idx = 0) {
  if ("uint8" == t.dtype) {
    return static_cast<int64_t>(((uint8_t *)t.data)[idx]);
  } else if ("uint16" == t.dtype) {
    return static_cast<int64_t>(((uint16_t *)t.data)[idx]);
  } else if ("uint32" == t.dtype) {
    return static_cast<int64_t>(((uint32_t *)t.data)[idx]);
  } else {
    throw std::runtime_error("Unsupported tensor dtype '" + t.dtype +
                             "' at line " + std::to_string(__LINE__));
  }
}

float tensor_to_float(Tensor t, size_t idx = 0) {
  if ("float" == t.dtype || "float32" == t.dtype) {
    return static_cast<float>(((float *)t.data)[idx]);
  } else {
    throw std::runtime_error("Unsupported tensor dtype '" + t.dtype +
                             "' at line " + std::to_string(__LINE__));
  }
}

template <typename T> std::vector<std::vector<T>> tensor_to_vector2d(Tensor t) {
  const T *t_data = static_cast<const T *>(t.data);
  std::vector<std::vector<T>> vec(t.shape[0], std::vector<T>(t.shape[1], 0));

  for (int i = 0; i < t.shape[0]; i++) {
    for (int j = 0; j < t.shape[1]; j++) {
      vec[i][j] = t_data[i * t.shape[1] + j];
    }
  }

  return vec;
}

template <typename T> std::vector<T> tensor_to_vector1d(Tensor t) {
  // Cast the tensor data to a pointer of type T
  const T *t_data = static_cast<const T *>(t.data);

  // flattened size of tensor
  uint64_t s = 1;
  for (auto e : t.shape) {
    s *= e;
  }

  // Use the range constructor to initialize the vector
  std::vector<T> vec(t_data, t_data + s);

  return vec;
}

std::tuple<int64_t, int32_t, int64_t, int32_t, int32_t, int32_t, int32_t>
qdq_act_matmul_uint8_uint8_cstm(float a_dq_xscale, int64_t a_dq_xzero_pt,
                                int64_t weights_in_ch, float w_dq_xscale,
                                int64_t w_dq_xzero_pt, float a_q_yscale,
                                int64_t a_q_yzero_pt) {
  // Ensure the zero points are of type int64_t
  int64_t a_dq_xzero_pt_i64 = static_cast<int64_t>(a_dq_xzero_pt);
  int64_t w_dq_xzero_pt_i64 = static_cast<int64_t>(w_dq_xzero_pt);
  int64_t a_q_yzero_pt_i64 = static_cast<int64_t>(a_q_yzero_pt);

  // Calculate the c2 coefficient
  float c2_coeff = (a_dq_xscale * w_dq_xscale) / a_q_yscale;
  int64_t c2_coeff_prime;
  int32_t shft_c2;

  std::tie(c2_coeff_prime, shft_c2) =
      OpsFusion::coeffs::find_closest_shifted_int32(c2_coeff, 8388607);

  c2_coeff_prime = static_cast<int64_t>(c2_coeff_prime);

  // Calculate the weight coefficient scale
  int64_t weight_coeff_scale = -c2_coeff_prime * a_dq_xzero_pt_i64;
  int32_t weight_coeff_scale_shift = 0;

  if (std::abs(weight_coeff_scale) > 2147483647) { // Max int32 number
    weight_coeff_scale_shift = static_cast<int32_t>(
        std::ceil(std::log2(std::abs(weight_coeff_scale))) - 31);
  } else {
    weight_coeff_scale_shift = 0;
  }

  weight_coeff_scale =
      static_cast<int32_t>(weight_coeff_scale >> weight_coeff_scale_shift);

  // Calculate c1 coefficient
  int64_t c1_coeff = a_q_yzero_pt_i64 << shft_c2;

  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset =
      static_cast<int32_t>(-a_dq_xzero_pt_i64 * num_weights_unrolled);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_dq_xzero_pt_i64;
  c1_coeff += c3_coeff_scale * static_cast<int64_t>(c3_coeff_offset);

  // Calculate the shift for c3 coefficient scale
  int32_t c3_coeff_scale_shift = 0;
  if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
    c3_coeff_scale_shift = static_cast<int32_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  } else {
    c3_coeff_scale_shift = 0;
  }

  c3_coeff_scale = static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);

  int32_t matmul_shift = 0;

  return std::make_tuple(c1_coeff,                                   // C0
                         static_cast<int32_t>(c3_coeff_scale),       // C1
                         static_cast<int64_t>(c2_coeff_prime),       // C2
                         static_cast<int32_t>(weight_coeff_scale),   // C3
                         static_cast<int32_t>(c3_coeff_scale_shift), // shift_qb
                         static_cast<int32_t>(shft_c2), // shift_out
                         matmul_shift                   // matmul_shift
  );
}

std::vector<int32_t>
mha_qdq_params_fill(const std::tuple<int64_t, int32_t, int64_t, int32_t,
                                     int32_t, int32_t, int32_t> &coeff_qkt,
                    const std::tuple<int64_t, int32_t, int64_t, int32_t,
                                     int32_t, int32_t, int32_t> &coeff_smv,
                    const std::tuple<uint16_t, int> &sm_qdq_before,
                    const std::tuple<uint16_t, int> &sm_qdq_after,
                    int32_t is_qkt_smv_int16) {
  std::vector<int32_t> qdq_params(96, 0);

  constexpr int32_t qry_subv_rows = 32;
  //   constexpr int32_t qry_subv_cols = 96;
  constexpr int32_t key_subv_rows = 64;
  //   constexpr int32_t key_subv_rows_int16 = 16;
  //   constexpr int32_t key_subv_cols = 96;
  //   constexpr int32_t val_subv_rows = 64;
  constexpr int32_t val_subv_cols = 64;
  //   constexpr int32_t out_subv_rows = 32;
  //   constexpr int32_t out_subv_cols = 64;

  // QKT
  reinterpret_cast<int64_t *>(qdq_params.data())[0] = std::get<0>(coeff_qkt);
  qdq_params[(16 * 0) + 2] = std::get<1>(coeff_qkt);
  qdq_params[(16 * 0) + 3] = static_cast<int32_t>(std::get<2>(coeff_qkt));
  qdq_params[(16 * 0) + 4] = std::get<3>(coeff_qkt);
  qdq_params[(16 * 0) + 5] = qry_subv_rows;
  qdq_params[(16 * 0) + 6] = key_subv_rows;
  qdq_params[(16 * 0) + 7] = std::get<4>(coeff_qkt);
  qdq_params[(16 * 0) + 8] = std::get<5>(coeff_qkt);
  qdq_params[(16 * 0) + 9] = std::get<6>(coeff_qkt);
  qdq_params[(16 * 0) + 10] = is_qkt_smv_int16;

  // SM *V
  reinterpret_cast<int64_t *>(qdq_params.data())[8] = std::get<0>(coeff_smv);
  qdq_params[(16 * 1) + 2] = std::get<1>(coeff_smv);
  qdq_params[(16 * 1) + 3] = (int32_t)std::get<2>(coeff_smv);
  qdq_params[(16 * 1) + 4] = std::get<3>(coeff_smv);
  qdq_params[(16 * 1) + 5] = qry_subv_rows;
  qdq_params[(16 * 1) + 6] = val_subv_cols;
  qdq_params[(16 * 1) + 7] = std::get<4>(coeff_smv);
  qdq_params[(16 * 1) + 8] = std::get<5>(coeff_smv);
  qdq_params[(16 * 1) + 9] = std::get<6>(coeff_smv);
  qdq_params[(16 * 1) + 10] = is_qkt_smv_int16;

  // DQ before SM
  qdq_params[(16 * 2) + 0] = std::get<1>(sm_qdq_before);
  qdq_params[(16 * 2) + 1] = std::get<0>(sm_qdq_before);

  // Q after SM
  qdq_params[(16 * 3) + 0] = std::get<1>(sm_qdq_after);
  qdq_params[(16 * 3) + 1] = std::get<0>(sm_qdq_after);
  qdq_params[(16 * 3) + 2] = is_qkt_smv_int16;

  return qdq_params;
}

static std::uint32_t convert_float_to_qint(float in_f) {
  std::uint32_t ret{0};
  std::memcpy(&ret, &in_f, sizeof(in_f));
  ret &= 0x7fffffff;
  return ret;
}

std::vector<int64_t> grpb_qgprb_vec64_fill(std::vector<int64_t> bias,
                                           int64_t qk_qdq_c0,
                                           int64_t smv_qdq_c0) {
  std::vector<int64_t> gprb_vec64(11, 0);

  for (int i = 0; i < 8; i++) {
    gprb_vec64[i] = bias[i];
  }

  gprb_vec64[9] = qk_qdq_c0;
  gprb_vec64[10] = smv_qdq_c0;

  return gprb_vec64;
}

std::vector<int32_t>
gprb_vec32_fill(const std::vector<int64_t> &coeff_grpb, float act_scale,
                int32_t act_zero_point, float wgt_scale, int32_t wgt_zero_point,
                const std::vector<uint16_t> &model_a, float model_a_scale,
                int32_t model_a_zp, uint16_t model_b, float model_b_scale,
                int32_t model_b_zp, uint16_t model_c, float model_c_scale,
                int32_t model_c_zp, int32_t is_grpb_int16) {

  std::vector<int32_t> gprb_vec32(32, 0);

  // const int qdq_c0_idx = 0;
  const int qdq_c1_idx = 2;
  const int qdq_c2_idx = 3;
  const int qdq_c3_idx = 4;
  const int qdq_Mv_idx = 5;
  const int qdq_Nv_idx = 6;
  const int qdq_SQb_idx = 7;
  const int qdq_Sout_idx = 8;
  const int qdq_Stdm_idx = 9;

  const int gprb_act_scale_idx = 10;
  const int gprb_act_zero_idx = 11;
  const int gprb_wgt_scale_idx = 12;
  const int gprb_wgt_zero_idx = 13;
  const int gprb_model_a_idx = 14;
  const int gprb_model_b_idx = 26;
  const int gprb_model_c_idx = 27;
  const int gprb_isint16_idx = 28;

  const int num_heads = 12;

  gprb_vec32[qdq_c1_idx] = (int32_t)coeff_grpb[0];
  gprb_vec32[qdq_c2_idx] = (int32_t)coeff_grpb[1];
  gprb_vec32[qdq_c3_idx] = 0;
  gprb_vec32[qdq_Mv_idx] = 32;
  gprb_vec32[qdq_Nv_idx] = 8;
  gprb_vec32[qdq_SQb_idx] = (int32_t)coeff_grpb[2];
  gprb_vec32[qdq_Sout_idx] = (int32_t)coeff_grpb[3];
  gprb_vec32[qdq_Stdm_idx] = (int32_t)coeff_grpb[4];

  gprb_vec32[gprb_act_scale_idx] =
      static_cast<int32_t>(OpsFusion::coeffs::float_to_bfloat16(act_scale));
  gprb_vec32[gprb_act_zero_idx] = act_zero_point;
  gprb_vec32[gprb_wgt_scale_idx] =
      static_cast<int32_t>(OpsFusion::coeffs::float_to_bfloat16(wgt_scale));
  gprb_vec32[gprb_wgt_zero_idx] = wgt_zero_point;
  gprb_vec32[gprb_isint16_idx] = is_grpb_int16;

  std::vector<float> model_a_bf(num_heads);
  for (size_t i = 0; i < num_heads; ++i) {
    model_a_bf[i] =
        OpsFusion::coeffs::dq<int32_t>(model_a[i], model_a_scale, model_a_zp);
  }

  for (int h = 0; h < num_heads; ++h) {
    gprb_vec32[gprb_model_a_idx + h] = static_cast<int32_t>(
        OpsFusion::coeffs::float_to_bfloat16(model_a_bf[h]));
  }

  gprb_vec32[gprb_model_b_idx] =
      static_cast<int32_t>(OpsFusion::coeffs::float_to_bfloat16(
          OpsFusion::coeffs::dq<int32_t>(model_b, model_b_scale, model_b_zp)));
  gprb_vec32[gprb_model_c_idx] =
      static_cast<int32_t>(OpsFusion::coeffs::float_to_bfloat16(
          OpsFusion::coeffs::dq<int32_t>(model_c, model_c_scale, model_c_zp)));

  return gprb_vec32;
}

static std::array<size_t, 2> extract_shape(const Tensor &tensor) {
  std::array<size_t, 2> res;
  if (tensor.shape.size() == 4) {
    res = {tensor.shape.at(2), tensor.shape.at(3)};
  } else if (tensor.shape.size() == 3) {
    res = {tensor.shape.at(0) * tensor.shape.at(1), tensor.shape.at(2)};
  } else if (tensor.shape.size() == 2) {
    res = {tensor.shape.at(0), tensor.shape.at(1)};
  } else if (tensor.shape.size() == 1) {
    res = {tensor.shape.at(0)};
  } else {
    throw std::runtime_error("MHA : Invalid shape received for Matrix");
  }
  return res;
}

template <typename T>
static void pad_bias(T *dst, const std::vector<size_t> &dst_shape, const T *src,
                     const std::vector<size_t> &src_shape) {
  DD_ASSERT(src_shape.size() == 3,
            "Padding bias is supported for only 3D tensor");

  for (size_t c = 0; c < src_shape.at(0); ++c) {
    for (size_t h = 0; h < src_shape.at(1); ++h) {
      auto src_ptr =
          src + c * src_shape.at(1) * src_shape.at(2) + h * src_shape.at(2);
      auto dst_ptr =
          dst + c * dst_shape.at(1) * dst_shape.at(2) + h * dst_shape.at(2);
      memcpy(dst_ptr, src_ptr, src_shape.at(2) * sizeof(T));
    }
  }
}

template <typename InT, typename WtT, typename OutT>
std::tuple<size_t, size_t, size_t>
mhagrpb<InT, WtT, OutT>::map_padded_shape(size_t M, size_t K, size_t N) const {
  const std::vector<matrix_shapes> &supported_shapes =
      default_shapes_.at(txn_fname_prefix_);

  size_t Mo = std::numeric_limits<size_t>::max();
  size_t Ko = std::numeric_limits<size_t>::max();
  size_t No = N;
  bool found_candidate = false;

  for (const auto &shape : supported_shapes) {
    const size_t dim0 = shape.M;
    const size_t dim1 = shape.K;
    const size_t dim2 = shape.N;

    if ((M <= dim0) && (Mo > dim0) && (K <= dim1) && (Ko > dim1) &&
        (N == dim2)) {
      Mo = dim0;
      Ko = dim1;
      found_candidate = true;
    }
  }

  DD_THROW_IF(
      !found_candidate,
      OpsFusion::dd_format(
          "Could not find kernel implementation for given dims {}x{}x{}", M, K,
          N));
  return std::make_tuple(Mo, Ko, No);
}

template <typename InT, typename WtT, typename OutT>
std::once_flag mhagrpb<InT, WtT, OutT>::instr_reg_flag_;

template <typename InT, typename WtT, typename OutT>
void mhagrpb<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
std::string mhagrpb<InT, WtT, OutT>::get_instr_key(std::string prefix, size_t m,
                                                   size_t k, size_t n) const {
  return "mhagrpb_" + prefix + "_" + std::to_string(m) + "_" +
         std::to_string(k) + "_" + std::to_string(n);
}

template <typename InT, typename WtT, typename OutT>
std::once_flag mhagrpb<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t mhagrpb<InT, WtT, OutT>::mhagrpb_count = 0;

template <typename InT, typename WtT, typename OutT>
void mhagrpb<InT, WtT, OutT>::setup_instr_registry() {

  std::vector<std::pair<std::string, bool>> instructions;
  std::vector<std::pair<std::string, bool>> layer_params;
  // mhagrpb
  const std::vector<matrix_shapes> &supported_shapes =
      default_shapes_.find(txn_fname_prefix_)->second;
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto mat = supported_shapes.at(i);

    auto key = get_instr_key(txn_fname_prefix_, mat.M, mat.K, mat.N);
    auto param_key =
        get_instr_key(param_fname_prefix_, mat.M, mat.K, mat.N) + "_param";

    instructions.push_back(std::make_pair(key, false));
    layer_params.push_back(std::make_pair(param_key, false));
  }

  xrt_ctx_->get_registry().add_instructions(instructions);
  xrt_ctx_->get_registry().add_layer_params(layer_params);
}

template <typename InT, typename WtT, typename OutT>
mhagrpb<InT, WtT, OutT>::mhagrpb(const std::string &a_dtype,
                                 const std::string &b_dtype,
                                 const std::string &c_dtype, bool load_xrt,
                                 const std::map<std::string, std::any> attr) {

  is_generic_pass_in_onnx = OpsFusion::check_generic_fusion(attr);

  txnbin_a_header = {{"uint16", "a16"}, {"uint8", "a8"}};

  txnbin_b_header = {{"uint16", "w16"}, {"uint8", "w8"}};

  txnbin_acc_header = {{"uint16", "acc16"}, {"uint8", "acc8"}};

  default_shapes_["mhagrpb_a8w8acc8"] = std::vector<matrix_shapes>();
  default_shapes_["mhagrpb_a16w8acc16"] = std::vector<matrix_shapes>();
  default_shapes_["mhagrpb_a16w16acc16"] = std::vector<matrix_shapes>();
  default_shapes_["mhagrpb_a8w8acc8"].emplace_back(512, 512, 768);
  default_shapes_["mhagrpb_a8w8acc8"].emplace_back(256, 256,
                                                   768); // For mdsqrv1.1
  default_shapes_["mhagrpb_a16w16acc16"].emplace_back(128, 128, 768);
  default_shapes_["mhagrpb_a16w8acc16"].emplace_back(512, 512, 768);
  default_shapes_["mhagrpb_a16w16acc16"].emplace_back(512, 512,
                                                      768); // For mxganv1.2

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  mhagrpb_id_ = mhagrpb_count++;

  /*select xclbin based on the input/output types*/
  std::string XCLBIN_FNAME =
      OpInterface::get_dd_base_dir() + ryzenai::mdsqr_A8W8_QDQ_XCLBIN_PATH;

  if (a_dtype_ == "uint16") {
    XCLBIN_FNAME =
        OpInterface::get_dd_base_dir() + ryzenai::mxpzi_A16W8_QDQ_XCLBIN_PATH;
  }

  txn_fname_prefix_ = "mhagrpb_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);

  param_fname_prefix_ = "mhagrpb_" + txnbin_a_header.at(a_dtype_) +
                        txnbin_b_header.at(b_dtype_) +
                        txnbin_acc_header.at(c_dtype_);

  KERNEL_M_MAX = 512;

  if (a_dtype_ == "uint16") {
    KERNEL_M_MAX = 128;
  }

  if (load_xrt == true) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);

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

  std::call_once(logger_flag_, []() {
    std::string header =
        "ipu_wrapper_id M K N kernel_m kernel_k kernel_n Execute"
        "time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[MHA] ID: " + std::to_string(mhagrpb_id_) +
                    ", (a_dtype, b_dtype, c_dtype): (" + a_dtype + ", " +
                    b_dtype + ", " + c_dtype + ")");
}

template <typename InT, typename WtT, typename OutT>
void mhagrpb<InT, WtT, OutT>::set_params(const std::string &model_name,
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
  } else if (model_name == "mdsqrv1.1") {
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                   ryzenai::mdsqrv1_1_A8W8_QDQ_XCLBIN_PATH;
  } else if (model_name == "mxganv1.2") {
    XCLBIN_FNAME = OpInterface::get_dd_base_dir() +
                   ryzenai::mxganv1_2_A16W8_QDQ_XCLBIN_PATH;
  } else {
    throw std::invalid_argument("model_name is not supported");
  }

  KERNEL_M_MAX = input_shape.at(0);

  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });

  RYZENAI_LOG_TRACE("[MHA] ID: " + std::to_string(mhagrpb_id_) +
                    ", XCLBIN: " + XCLBIN_FNAME);
}

template <typename InT, typename WtT, typename OutT>
void mhagrpb<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("MHAGRPB initialize_const_params(ptr) ...");

  is_generic_pass_in_onnx = OpsFusion::check_generic_fusion(attr);

  uint8_t *weights;
  int64_t *gprb_vec64;
  int32_t *gprb_vec32;
  WtT *bias;
  int32_t *qdq_param;

  std::vector<int64_t> gprb_vec64_vec;
  std::vector<int32_t> gprb_vec32_vec;
  std::vector<uint8_t> vec3d;
  std::vector<int32_t> qdq_params_vec;

  std::vector<size_t> weight_shape;
  std::vector<size_t> bias_shape;

  size_t Q, K, V, pad_Q, pad_K, pad_V, padded_size_bias, size_bias;

  std::vector<size_t> padded_bias_shape;

  std::vector<uint8_t> padded_bias;

  if (is_generic_pass_in_onnx) {

    float query_sc = tensor_to_float(const_params.at(1));
    int64_t query_zp = tensor_to_int64(const_params.at(2));

    float key_sc = tensor_to_float(const_params.at(3));
    int64_t key_zp = tensor_to_int64(const_params.at(4));

    float qkt_sc = tensor_to_float(const_params.at(5));
    int64_t qkt_zp = tensor_to_int64(const_params.at(6));

    float sm_sc = tensor_to_float(const_params.at(7));
    int64_t sm_zp = tensor_to_int64(const_params.at(8));

    float v_sc = tensor_to_float(const_params.at(9));
    int64_t v_zp = tensor_to_int64(const_params.at(10));

    float vsm_sc = tensor_to_float(const_params.at(11));
    int64_t vsm_zp = tensor_to_int64(const_params.at(12));

    std::vector<std::vector<uint8_t>> grpb_w =
        tensor_to_vector2d<uint8_t>(const_params.at(13));
    float grpb_w_sc = tensor_to_float(const_params.at(14));
    int64_t grpb_w_zp = tensor_to_int64(const_params.at(15));

    auto temp_grpb_b = tensor_to_vector1d<uint8_t>(const_params.at(16));
    std::vector<uint16_t> grpb_b(temp_grpb_b.size());
    std::transform(temp_grpb_b.begin(), temp_grpb_b.end(), grpb_b.begin(),
                   [](uint8_t value) { return static_cast<uint16_t>(value); });

    float grpb_b_sc = tensor_to_float(const_params.at(17));
    int64_t grpb_b_zp = tensor_to_int64(const_params.at(18));

    float grpb_sc = tensor_to_float(const_params.at(19));
    int64_t grpb_zp = tensor_to_int64(const_params.at(20));

    int64_t div_w = tensor_to_int64(const_params.at(21));
    float div_w_sc = tensor_to_float(const_params.at(22));
    int64_t div_w_zp = tensor_to_int64(const_params.at(23));

    auto temp_mul_1_w = tensor_to_vector1d<uint8_t>(const_params.at(24));
    std::vector<uint16_t> mul_1_w(temp_mul_1_w.size());
    std::transform(temp_mul_1_w.begin(), temp_mul_1_w.end(), mul_1_w.begin(),
                   [](uint8_t value) { return static_cast<uint16_t>(value); });

    float mul_1_w_sc = tensor_to_float(const_params.at(25));
    int64_t mul_1_w_zp = tensor_to_int64(const_params.at(26));

    auto mul_3_w = tensor_to_vector1d<uint8_t>(const_params.at(27));
    float mul_3_w_sc = tensor_to_float(const_params.at(28));
    int64_t mul_3_w_zp = tensor_to_int64(const_params.at(29));

    int64_t add_w = tensor_to_int64(const_params.at(30));
    float add_sc = tensor_to_float(const_params.at(31));
    int64_t add_zp = tensor_to_int64(const_params.at(32));

    int64_t sub_w = tensor_to_int64(const_params.at(33));
    float sub_sc = tensor_to_float(const_params.at(34));
    int64_t sub_zp = tensor_to_int64(const_params.at(35));

    int is_qkt_smv_int16 = 0;
    int is_grpb_int16 = 0;

    auto coeff_qkt = qdq_act_matmul_uint8_uint8_cstm(
        query_sc, query_zp, 96, key_sc, key_zp, qkt_sc, qkt_zp);

    auto coeff_smv = qdq_act_matmul_uint8_uint8_cstm(sm_sc, sm_zp, 512, v_sc,
                                                     v_zp, vsm_sc, vsm_zp);

    qdq_params_vec = mha_qdq_params_fill( // in32_t * 96
        coeff_qkt, coeff_smv,
        std::make_tuple(OpsFusion::coeffs::float_to_bfloat16(
                            qkt_sc / ((div_w - div_w_zp) * div_w_sc)),
                        (int)qkt_zp),
        std::make_tuple(OpsFusion::coeffs::float_to_bfloat16(1.0f / sm_sc),
                        (int)sm_zp),
        is_qkt_smv_int16);

    // Extracting index 0 of mul_3_w
    uint64_t mul3_w_size0 = 1;
    for (auto e : const_params.at(27).shape) {
      mul3_w_size0 *= e;
    }
    mul3_w_size0 /= const_params.at(27).shape[0];

    vec3d = std::vector<uint8_t>(mul3_w_size0);
    for (uint64_t i = 0; i < mul3_w_size0; i++) {
      vec3d[i] = mul_3_w[i];
    }

    auto [c0_gate_linear, c1_gate_linear, c2_gate_linear, shift_qb_gate_linear,
          shift_out_gate_linear, matmul_shift_gate_linear] =
        OpsFusion::coeffs::compute_qdq_coeff_matmul_bias(
            query_sc, (uint8_t)query_zp, grpb_w, grpb_w_sc, (uint8_t)grpb_w_zp,
            grpb_b, grpb_b_sc, (uint8_t)grpb_b_zp, grpb_sc, (uint8_t)grpb_zp);

    gprb_vec64_vec = grpb_qgprb_vec64_fill( // int64_t * 11
        c0_gate_linear, std::get<0>(coeff_qkt), std::get<0>(coeff_smv));

    gprb_vec32_vec = gprb_vec32_fill( // int32_t * 32
        {
            c1_gate_linear,
            c2_gate_linear,
            shift_qb_gate_linear,
            shift_out_gate_linear,
            matmul_shift_gate_linear,
        },
        grpb_sc, (int32_t)grpb_zp, mul_3_w_sc, (int32_t)mul_3_w_zp, mul_1_w,
        mul_1_w_sc, (int32_t)mul_1_w_zp, (uint16_t)sub_w, sub_sc,
        (int32_t)sub_zp, (uint16_t)add_w, add_sc, (int32_t)add_zp,
        is_grpb_int16);

    weights = (uint8_t *)const_params.at(0).data;
    gprb_vec64 = gprb_vec64_vec.data();
    gprb_vec32 = gprb_vec32_vec.data();
    bias = (WtT *)vec3d.data();
    qdq_param = qdq_params_vec.data();

    bias_shape = {const_params.at(27).shape[1], const_params.at(27).shape[2],
                  const_params.at(27).shape[3]};

    weight_shape = const_params.at(0).shape;

    Q = bias_shape.at(1);
    K = bias_shape.at(2);
    V = weight_shape.at(0) * weight_shape.at(1);

    std::tie(pad_Q, pad_K, pad_V) = map_padded_shape(Q, K, V);

    padded_bias_shape = {bias_shape.at(0), static_cast<size_t>(pad_Q),
                         static_cast<size_t>(pad_K)};

    padded_size_bias =
        std::accumulate(padded_bias_shape.begin(), padded_bias_shape.end(),
                        size_t{1}, std::multiplies{}) *
        b_dtype_size_;

    padded_bias = std::vector<uint8_t>(padded_size_bias, 0);

    pad_bias((WtT *)padded_bias.data(), padded_bias_shape, (WtT *)bias,
             bias_shape);

    size_bias = padded_size_bias;

  } else {

    DD_THROW_IF(
        (const_params.size() != 5) || (const_params.at(0).shape.size() != 2),
        OpsFusion::dd_format(
            "Unsupported const spec for MHAGRPB\n"
            "(Details : #const params == 2 ({}), Const param1 dim == 2 ({})",
            const_params.size(), const_params.at(0).shape.size()));

    const int weight_idx = 0, gprb_vec64_idx = 1, gprb_vec32_idx = 2,
              bias_idx = 3, qdq_idx = 4;

    weights = (uint8_t *)const_params.at(weight_idx).data;
    weight_shape = const_params.at(weight_idx).shape;

    gprb_vec64 = (int64_t *)const_params.at(gprb_vec64_idx).data;

    gprb_vec32 = (int32_t *)const_params.at(gprb_vec32_idx).data;

    bias = (WtT *)const_params.at(bias_idx).data;
    bias_shape = const_params.at(bias_idx).shape;

    Q = bias_shape.at(1);
    K = bias_shape.at(2);
    V = weight_shape.at(0) * weight_shape.at(1);

    std::tie(pad_Q, pad_K, pad_V) = map_padded_shape(Q, K, V);

    padded_bias_shape = {bias_shape.at(0), static_cast<size_t>(pad_Q),
                         static_cast<size_t>(pad_K)};

    padded_size_bias =
        std::accumulate(padded_bias_shape.begin(), padded_bias_shape.end(),
                        size_t{1}, std::multiplies{}) *
        b_dtype_size_;

    padded_bias = std::vector<uint8_t>(padded_size_bias, 0);

    pad_bias((WtT *)padded_bias.data(), padded_bias_shape, (WtT *)bias,
             bias_shape);

    bias = (WtT *)padded_bias.data();
    bias_shape = padded_bias_shape;
    size_bias = padded_size_bias;

    qdq_param = (int32_t *)const_params.at(qdq_idx).data;
  }

  size_t H = bias_shape[0];
  size_t St = bias_shape[1];
  size_t S = bias_shape[2];

  int qry_subv_rows_act = qry_subv_rows;

  if constexpr (std::is_same_v<InT, uint16_t>) { // mxgan
    if (512 == S) {
      qry_subv_rows_act = qry_subv_rows_mxgan;
    }
  }

  size_t size_gprbparam = GPRB_buf_size;
  std::vector<uint8_t> dest_buffer(GPRB_buf_size);
  auto prm =
      (mhagprb_matrix::GprbParams<int64_t, InT, gprb_rows, gprb_cols, num_heads>
           *)dest_buffer.data();

  for (int i = 0; i < gprb_rows * gprb_cols; ++i) {
    prm->proj_mat[i] = weights[i];
  }

  for (int i = 0; i < gprb_cols; ++i) {
    prm->qdq_bias[i] = gprb_vec64[i];
  }

  prm->c0 = gprb_vec64[gprb_c0_scalar_idx];
  prm->c1 = gprb_vec32[qdq_c1_idx];
  prm->c2 = gprb_vec32[qdq_c2_idx];
  prm->c3 = gprb_vec32[qdq_c3_idx];
  prm->M = qry_subv_rows_act;
  prm->N = gprb_cols;
  prm->shift_Qb = gprb_vec32[qdq_SQb_idx];
  prm->shift_Qout = gprb_vec32[qdq_Sout_idx];
  prm->res = gprb_vec32[qdq_Stdm_idx];
  prm->act_scale.value = gprb_vec32[gprb_act_scale_idx];
  prm->act_zero_point = (InT)gprb_vec32[gprb_act_zero_idx];
  prm->wgt_scale.value = gprb_vec32[gprb_wgt_scale_idx];
  prm->wgt_zero_point = (InT)gprb_vec32[gprb_wgt_zero_idx];
  for (int h = 0; h < num_heads; ++h) {
    prm->model_a[h].value = gprb_vec32[gprb_model_a_idx + h];
  }
  prm->model_b.value = gprb_vec32[gprb_model_b_idx];
  prm->model_c.value = gprb_vec32[gprb_model_c_idx];
  prm->isint16 = gprb_vec32[gprb_isint16_idx];

  RYZENAI_LOG_TRACE("MHA: size_gprbparam:" + std::to_string(size_gprbparam) +
                    " size_bias:" + std::to_string(size_bias));
  io.write(0, dest_buffer.data(), size_gprbparam);
  size_t const lnr_lut_ab_size = sizeof(lnr_lutab);
  size_t const lnr_lut_cd_size = sizeof(lnr_lutcd);
  dest_buffer.resize(size_bias);
  void *b_bias = dest_buffer.data();

  if (S == (key_subv_rows_mxpzi * 8)) {
    mhagprb_matrix::ScaleTensor<WtT, qry_subv_rows, key_subv_rows_mxpzi> aie_S(
        (int)H, (int)St, (int)S, b_bias);
    for (int h = 0; h < H; ++h) {
      for (int i = 0; i < St; ++i) {
        for (int j = 0; j < S; ++j) {
          aie_S.at(h, i, j) = bias[h * St * S + i * S + j];
        }
      }
    }
  } else if (qry_subv_rows_act == qry_subv_rows_mxgan) { // mxgan
    mhagprb_matrix::ScaleTensor<WtT, qry_subv_rows_mxgan, key_subv_rows> aie_S(
        (int)H, (int)St, (int)S, b_bias);
    for (int h = 0; h < H; ++h) {
      for (int i = 0; i < St; ++i) {
        for (int j = 0; j < S; ++j) {
          aie_S.at(h, i, j) = bias[h * St * S + i * S + j];
        }
      }
    }
  } else if (S == (key_subv_rows_mdsqrv1_1 * 8)) {
    mhagprb_matrix::ScaleTensor<WtT, qry_subv_rows, key_subv_rows_mdsqrv1_1>
        aie_S((int)H, (int)St, (int)S, b_bias);
    for (int h = 0; h < H; ++h) {
      for (int i = 0; i < St; ++i) {
        for (int j = 0; j < S; ++j) {
          aie_S.at(h, i, j) = bias[h * St * S + i * S + j];
        }
      }
    }
  } else {
    mhagprb_matrix::ScaleTensor<WtT, qry_subv_rows, key_subv_rows> aie_S(
        (int)H, (int)St, (int)S, b_bias);
    for (int h = 0; h < H; ++h) {
      for (int i = 0; i < St; ++i) {
        for (int j = 0; j < S; ++j) {
          aie_S.at(h, i, j) = bias[h * St * S + i * S + j];
        }
      }
    }
  }
  io.write(size_gprbparam, dest_buffer.data(), size_bias);

  io.write(size_gprbparam + size_bias, (void *)lnr_lutab, lnr_lut_ab_size);
  io.write(size_gprbparam + size_bias + lnr_lut_ab_size, (void *)lnr_lutcd,
           lnr_lut_cd_size);
  int size_qdqparam = QDQparam_size * num_qdq_nodes * sizeof(int32_t);
  *(int64_t *)(&qdq_param[(16 * 0) + qdq_c0_idx]) =
      gprb_vec64[qk_qdq_c0_scalar_idx];
  *(int64_t *)(&qdq_param[(16 * 1) + qdq_c0_idx]) =
      gprb_vec64[smv_qdq_c0_scalar_idx];

  // SW convert scale to 1/scale and bfloat16 for Q

  qdq_param[(16 * 0) + qdq_Mv_idx] = qry_subv_rows_act;
  if (S == (key_subv_rows_mxpzi * 8)) {
    qdq_param[(16 * 0) + qdq_Nv_idx] = key_subv_rows_mxpzi;
  } else if (S == (key_subv_rows_mdsqrv1_1 * 8)) {
    qdq_param[(16 * 0) + qdq_Nv_idx] = key_subv_rows_mdsqrv1_1;
  } else {
    qdq_param[(16 * 0) + qdq_Nv_idx] = key_subv_rows;
  }
  qdq_param[(16 * 1) + qdq_Mv_idx] = qry_subv_rows_act;
  qdq_param[(16 * 1) + qdq_Nv_idx] = val_subv_cols;

  io.write(size_gprbparam + size_bias + lnr_lut_ab_size + lnr_lut_cd_size,
           qdq_param, size_qdqparam);

  RYZENAI_LOG_TRACE("MHAGRPB initialize_const_params(ptr) ... DONE");
}

// For MHA+GPRB: weight + bias
template <typename InT, typename WtT, typename OutT>
void mhagrpb<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  // Check the number of inputs
  DD_ASSERT((const_params.size() == 5),
            OpsFusion::dd_format("MHAGRPB expects two constants. Got {}",
                                 const_params.size()));
  const int weight_idx = 0, bias_idx = 3;

  std::vector<size_t> shape = const_params.at(weight_idx).shape;
  size_t size_weight = shape[0] * shape[1]; // always uint8

  shape = const_params.at(bias_idx).shape;
  size_t size_bias = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                     std::multiplies{}) *
                     b_dtype_size_;
  size_t H = shape[0];
  size_t size_lutab = sizeof(lnr_lutab);
  size_t size_lutcd = sizeof(lnr_lutcd);
  // this is the weights + gprb_vec + gprb_qdq_params
  size_t size_mhaparam = GPRB_buf_size;
  size_t size_qdqparam = QDQparam_size * num_qdq_nodes * sizeof(int32_t);
  size_t size_msk = KERNEL_M_MAX * 2; // msk uint16
  // Init the BO size
  kernel_x_shape_[0] = KERNEL_M_MAX; // Q, K, V, mask
  kernel_x_shape_[1] = 256 * H;      // Q+K+V = 3072;
  kernel_y_shape_[0] = H;            // for bias
  kernel_y_shape_[1] = KERNEL_M_MAX * KERNEL_M_MAX;
  kernel_z_shape_[0] = KERNEL_M_MAX;
  kernel_z_shape_[1] = out_subv_cols * H;
  w_shape_[0] = KERNEL_M_MAX;
  w_shape_[1] = 3072;

  // Create input/output BOs
  const size_t A_BO_SIZE =
      (kernel_x_shape_[0] * kernel_x_shape_[1] * a_dtype_size_) + size_msk;
  const size_t B_BO_SIZE =
      (kernel_y_shape_[0] * kernel_y_shape_[1] * b_dtype_size_ + size_mhaparam +
       size_lutab + size_lutcd + size_qdqparam);
  const size_t C_BO_SIZE =
      (kernel_z_shape_[0] * kernel_z_shape_[1] * c_dtype_size_);

  RYZENAI_LOG_TRACE("MHA: A_BO_SIZE:" + std::to_string(A_BO_SIZE) +
                    " B_BO_SIZE:" + std::to_string(B_BO_SIZE) +
                    " C_BO_SIZE size:" + std::to_string(C_BO_SIZE));
  RYZENAI_LOG_TRACE("MHA: size_weight:" + std::to_string(size_weight) +
                    " size_bias:" + std::to_string(size_bias) +
                    " size_lutab:" + std::to_string(size_lutab) +
                    " size_lutcd:" + std::to_string(size_lutcd));

  a_bo_ = xrt::bo(xrt_ctx_->get_device(), A_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), B_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));
  c_bo_ = xrt::bo(xrt_ctx_->get_device(), C_BO_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(8));

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
}

// Q+K+V+mask
template <typename InT, typename WtT, typename OutT>
void mhagrpb<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                      std::vector<Tensor> &output) {
  // Check the number of inputs
  if (input.size() != 4) {
    throw std::runtime_error("MHA IPU Wrapper expect to have three inputs.");
  }
  const int q_idx = 0, k_idx = 1, v_idx = 2, msk_idx = 3;
  // The first data is Query
  InT *a = (InT *)input.at(q_idx).data;
  // The second data is Key
  InT *key = (InT *)input.at(k_idx).data;
  // The third data is Val
  InT *val = (InT *)input.at(v_idx).data;
  // The forth data is mask
  uint16_t *msk = (uint16_t *)input.at(msk_idx).data;

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

  a_shape_[0] = input.at(q_idx).shape.at(0);
  a_shape_[1] = input.at(q_idx).shape.at(1);

  int64_t key_shape_[2], val_shape_[2], msk_shape_[2];
  key_shape_[0] = input.at(k_idx).shape.at(0);
  key_shape_[1] = input.at(k_idx).shape.at(1);
  val_shape_[0] = input.at(v_idx).shape.at(0);
  val_shape_[1] = input.at(v_idx).shape.at(1);
  msk_shape_[0] = input.at(msk_idx).shape.at(0);
  msk_shape_[1] = input.at(msk_idx).shape.at(1);
  assert(key_shape_[1] == val_shape_[0]);
  c_shape_[0] = a_shape_[0];
  c_shape_[1] = val_shape_[1];
  kernel_x_rows = a_shape_[0];

  // a_bo copy
  auto a_copy_start = GET_ELAPSED_TIME_NS();
  InT *a_bo_map = a_bo_.map<InT *>();
  size_t a_size = a_shape_[0] * a_shape_[1] * sizeof(InT);
  memcpy((void *)a_bo_map, (void *)a, a_size);
  size_t key_size = key_shape_[0] * key_shape_[1] * sizeof(InT);
  size_t val_size = val_shape_[0] * val_shape_[1] * sizeof(InT);
  size_t msk_size = msk_shape_[0] * msk_shape_[1] * sizeof(uint16_t);
  void *aie_key =
      static_cast<void *>((reinterpret_cast<int8_t *>(a_bo_map)) + a_size);
  void *aie_val =
      static_cast<void *>((reinterpret_cast<int8_t *>(aie_key)) + key_size);
  memcpy((void *)aie_key, (void *)key, key_size);
  memcpy((void *)aie_val, (void *)val, val_size);
  void *aie_msk =
      static_cast<void *>((reinterpret_cast<int8_t *>(aie_val)) + val_size);
  memcpy((void *)aie_msk, (void *)msk, msk_size);
  auto a_copy_stop = GET_ELAPSED_TIME_NS();

  // a_bo sync
  auto a_sync_start = GET_ELAPSED_TIME_NS();
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto a_sync_stop = GET_ELAPSED_TIME_NS();

  a_copy_time_ = static_cast<int64_t>(a_copy_stop - a_copy_start);
  a_sync_time_ = static_cast<int64_t>(a_sync_stop - a_sync_start);

  w_shape_[0] = key_shape_[1];
  w_shape_[1] = key_shape_[0] + val_shape_[1];

  // prepare inst_bo and param_bo
  auto instr_bo_key = get_instr_key(txn_fname_prefix_, a_shape_[0],
                                    key_shape_[0], val_shape_[1]);
  auto param_bo_key = get_instr_key(param_fname_prefix_, a_shape_[0],
                                    key_shape_[0], val_shape_[1]) +
                      "_param";
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  const xrt::bo &param_bo =
      xrt_ctx_->get_registry().get_param_bo(param_bo_key).second;
  uint32_t instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();

  // launch the kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  c_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, c_bo_, a_bo_, b_bo_,
                                            param_bo, 0, true, false);
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

  RYZENAI_LOG_TRACE("MHAGRPB execute() ... DONE");
  RYZENAI_LOG_INFO(
      std::to_string(mhagrpb_id_) + " " + std::to_string(a_shape_[0]) + " " +
      std::to_string(a_shape_[1]) + " " + std::to_string(w_shape_[1]) + " " +
      std::to_string(kernel_x_rows) + " " + std::to_string(kernel_x_shape_[1]) +
      " " + std::to_string(kernel_y_shape_[1]) + " " +
      std::to_string(exec_end - exec_start) + " " +
      std::to_string(num_run_aie_) + " " + std::to_string(run_aie_time_) + " " +
      std::to_string(a_copy_time_) + " " + std::to_string(a_sync_time_) + " " +
      std::to_string(c_copy_time_) + " " + std::to_string(c_sync_time_) + " " +
      std::to_string((double)run_aie_time_ / num_run_aie_) + "\n");
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> mhagrpb<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto Q_shape = extract_shape(input.at(0));
  auto K_shape = extract_shape(input.at(1));
  auto V_shape = extract_shape(input.at(2));

  auto [Q_pad, K_pad, V_pad] =
      map_padded_shape(Q_shape[0], K_shape[0], V_shape[1]);
  std::string txn_key = get_instr_key(txn_fname_prefix_, Q_pad, K_pad, V_pad);

  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> mhagrpb<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  auto Q_shape = extract_shape(input.at(0));
  auto K_shape = extract_shape(input.at(1));
  auto V_shape = extract_shape(input.at(2));
  auto [Q_pad, K_pad, V_pad] =
      map_padded_shape(Q_shape[0], K_shape[0], V_shape[1]);
  std::string param_key =
      get_instr_key(param_fname_prefix_, Q_pad, K_pad, V_pad) + "_param";

  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(param_key);
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> mhagrpb<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {

  // [Q, K, V, mask, wgt, bias, out]
  if (input.size() != int(is_generic_pass_in_onnx ? 41 : 10)) {
    throw std::runtime_error(
        "MHA : Incorrect number of tensors received, " +
        std::to_string(input.size()) + " v/s " +
        std::to_string(int(is_generic_pass_in_onnx ? 41 : 10)));
  }

  auto Q_shape = extract_shape(input.at(0));
  auto K_shape = extract_shape(input.at(1));
  auto V_shape = extract_shape(input.at(2));
  auto mask_shape = extract_shape(input.at(3));
  auto orig_bias_shape =
      extract_shape(input.at(is_generic_pass_in_onnx ? 31 : 7));
  auto out_shape = extract_shape(input.at(is_generic_pass_in_onnx ? 40 : 9));

  // Update shapes as per Kernel default shape
  auto [Q_pad, K_pad, V_pad] =
      map_padded_shape(Q_shape[0], K_shape[0], V_shape[1]);
  std::vector<size_t> bias_shape;
  if (is_generic_pass_in_onnx) {
    bias_shape = {input.at(31).shape.at(1), static_cast<size_t>(Q_pad),
                  static_cast<size_t>(K_pad)};
  } else {
    bias_shape = {input.at(7).shape.at(0), static_cast<size_t>(Q_pad),
                  static_cast<size_t>(K_pad)};
  }
  out_shape[0] = Q_pad;
  mask_shape.back() = Q_pad;

  int size_mhaparam = GPRB_buf_size; // this is the actual weight size
                                     // allocate in hw
  int size_qdqparam = QDQparam_size * num_qdq_nodes * sizeof(int32_t);
  size_t Q_size = (Q_pad * Q_shape[1] * sizeof(InT));
  size_t K_size = (K_pad * K_shape[1] * sizeof(InT));
  size_t V_size = (Q_pad * V_pad * sizeof(InT));
  size_t mask_size = (mask_shape[0] * mask_shape[1] * sizeof(uint16_t));
  size_t bias_size = (std::accumulate(bias_shape.begin(), bias_shape.end(),
                                      size_t{1}, std::multiplies{}) *
                      sizeof(WtT));
  size_t out_size = (out_shape[0] * out_shape[1] * sizeof(OutT));

  size_t super_kernel_size = get_super_kernel_params(input, output).size();
  int const lnr_lut_ab_size = sizeof(lnr_lutab);
  int const lnr_lut_cd_size = sizeof(lnr_lutcd);

  size_t out_index = is_generic_pass_in_onnx ? 40 : 9;

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, Q_size},
      {OpArgMap::OpArgType::INPUT, 1, 1, Q_size, K_size},
      {OpArgMap::OpArgType::INPUT, 1, 2, Q_size + K_size, V_size},
      {OpArgMap::OpArgType::INPUT, 1, 3, Q_size + K_size + V_size, mask_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 4, 0,
       bias_size + size_mhaparam + lnr_lut_ab_size + lnr_lut_cd_size +
           size_qdqparam},
      {OpArgMap::OpArgType::OUTPUT, 0, out_index, 0, out_size},
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
       super_kernel_size}};
  return arg_map;
}

// Taken from dd custom op
static uint16_t float_to_bfloat16_1(float x) {
  uint32_t i;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *tmp = (uint8_t *)&i;
  // copy float to uint32_t
  std::memcpy(tmp, src, sizeof(float));
  // round to nearest even
  uint32_t lsb = (i >> 16) & 0x1;
  uint32_t bias = 0x7fff + lsb;
  i += bias;
  // extract upper half of input
  uint16_t y = uint16_t(i >> 16);
  return y;
}

template <typename InT, typename WtT, typename OutT>
void mhagrpb<InT, WtT, OutT>::initialize_inputs(
    const std::vector<Tensor> &input,
    const std::map<std::string, std::any> &attr) {
  auto Q_shape = extract_shape(input.at(0));
  auto K_shape = extract_shape(input.at(1));
  auto V_shape = extract_shape(input.at(2));
  auto mask_shape = extract_shape(input.at(3));

  // Mask should be padded with ZeroPoint to kernel's default shape
  auto [Q_pad, K_pad, V_pad] =
      map_padded_shape(Q_shape[0], K_shape[0], V_shape[1]);
  if (mask_shape.back() != Q_pad) {
    auto &mask_tensor = input.at(3);
    const auto iq_params =
        std::any_cast<std::vector<float>>(MAP_AT(attr, "input_q_params"));
    float scale = ARRAY_AT(iq_params, 6); // TODO: set based on if generic
    float zp = ARRAY_AT(iq_params, 7);    // TODO: set based on if generic
    if (mask_tensor.dtype == "bfloat16") {
      float value = (float)((-zp) * (scale));
      auto res = float_to_bfloat16_1(value);
      auto *mask_ptr = static_cast<uint16_t *>(mask_tensor.data);
      for (size_t i = mask_shape.back(); i < Q_pad; ++i) {
        mask_ptr[i] = res;
      }
    }
  }
}

template class mhagrpb<uint8_t, uint8_t, uint8_t>;
template class mhagrpb<uint16_t, uint16_t, uint16_t>;
template class mhagrpb<uint16_t, uint8_t, uint16_t>;
} // namespace ryzenai
