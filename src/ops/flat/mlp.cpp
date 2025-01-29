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

#include "txn_container.hpp"
#include <ops/flat/mlp.hpp>
#include <ops/op_interface.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>

#include "ops/ops_common/mladf_matmul_matrix.hpp"
namespace ryzenai {
namespace flat {

template <typename InT, typename WtT, typename OutT>
void mlp<InT, WtT, OutT>::setup_instr_registry() {
  std::vector<std::pair<std::string, bool>> instructions;
  DD_ASSERT(default_shapes_.find(txn_fname_prefix_) != default_shapes_.end(),
            OpsFusion::dd_format("txn_fname_prefix_ {} not found",
                                 txn_fname_prefix_));
  auto supported_shapes = default_shapes_[txn_fname_prefix_];
  for (size_t i = 0; i < supported_shapes.size(); i++) {
    auto shape = supported_shapes.at(i);
    auto key = get_instr_key(txn_fname_prefix_, shape);
    instructions.push_back(std::make_pair(key, false));
  }
  xrt_ctx_->get_registry().add_instructions(instructions);
}

template <typename InT, typename WtT, typename OutT>
std::string
mlp<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                   const std::vector<size_t> &shape) const {
  std::string out_str = prefix;
  for (size_t i = 0; i < shape.size(); i++) {
    out_str += "_" + std::to_string(shape[i]);
  }
  return out_str;
}

template <typename InT, typename WtT, typename OutT>
mlp<InT, WtT, OutT>::mlp(const std::string &ifm_dtype,
                         const std::string &gate_wts_dtype,
                         const std::string &ofm_dtype, bool load_xrt,
                         const std::map<std::string, std::any> &attr) {
  a_dtype_ = ifm_dtype;
  b_dtype_ = gate_wts_dtype;
  c_dtype_ = ofm_dtype;
  txnbin_a_header = {{"bfloat16", "a16bf"}, {"bfp16ebs8", "a16bfp"}};
  txnbin_b_header = {{"bfloat16", "w16bf"}, {"int4", "w3"}, {"uint4", "w4"}};
  txnbin_acc_header = {{"bfloat16", "acc16bf"}};
  txn_fname_prefix_ = "flat_mlp_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
  // M, K, N
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{1, 3072, 11008});

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  mlp_id_++;

  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));
    if (input_shape_vector.size() == 3) {
      M_ = uint64_t(input_shape_vector[0]);
      K_ = uint64_t(input_shape_vector[1]);
      N_ = uint64_t(input_shape_vector[2]);
      DD_ASSERT(K_ % sv_k == 0,
                OpsFusion::dd_format(
                    "K_ {} must be multiples of sv_k=128 but is", K_));
      DD_ASSERT(N_ % sv_n == 0,
                OpsFusion::dd_format(
                    "N_ {} must be multiples of sv_k=128 but is", N_));
      ifm_size_ = M_ * K_ * a_dtype_size_;
      ofm_size_ = M_ * N_ * c_dtype_size_;
      cal_shuffled_wts_size(N_, K_);
    } else {
      throw std::runtime_error(
          "FlatMLP input_shape attr should be a vector of size 4");
    }
    RYZENAI_LOG_TRACE("FlatMLP: InputShape: " + std::to_string(M_) + ", " +
                      std::to_string(K_) + ", " + std::to_string(N_));
  } else {
    throw std::runtime_error(
        "FlatMLP input_shape attr not found or not of correct type");
  }

  XCLBIN_FNAME_ =
      OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\FlatMLP.xclbin";
  RYZENAI_LOG_TRACE(OpsFusion::dd_format("xclbin fname : {}", XCLBIN_FNAME_));
  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
    std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
  }

  std::call_once(logger_flag_, []() {
    std::string header = "ipu_wrapper_id M K N Execute"
                         "time(us) num_aie_runs run_aie_time(ns) "
                         "A_copy_time(ns) A_sync_time(ns) "
                         "C_copy_time(ns) C_sync_time(ns) "
                         "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });

  RYZENAI_LOG_TRACE("[FLAT MLP] ID: " + std::to_string(mlp_id_) + ", XCLBIN: " +
                    XCLBIN_FNAME_ + ", (a_dtype, b_dtype, c_dtype): (" +
                    a_dtype_ + ", " + b_dtype_ + ", " + c_dtype_ + ")");
}

template <typename InT, typename WtT, typename OutT>
void mlp<InT, WtT, OutT>::set_params(std::vector<uint64_t> &input_shape,
                                     bool create_bo) {
  M_ = input_shape[0];
  K_ = input_shape[1];
  N_ = input_shape[2];
  ifm_size_ = M_ * K_ * a_dtype_size_;
  ofm_size_ = M_ * N_ * c_dtype_size_;
  cal_shuffled_wts_size(N_, K_);
  if (create_bo) {
    const int gid = 0;
    a_bo_ = xrt::bo(xrt_ctx_->get_device(), ifm_size_, XRT_BO_FLAGS_HOST_ONLY,
                    xrt_ctx_->get_kernel().group_id(gid));
    b_bo_ = xrt::bo(xrt_ctx_->get_device(), wts_size_, XRT_BO_FLAGS_HOST_ONLY,
                    xrt_ctx_->get_kernel().group_id(gid));

    c_bo_ = xrt::bo(xrt_ctx_->get_device(), ofm_size_, XRT_BO_FLAGS_HOST_ONLY,
                    xrt_ctx_->get_kernel().group_id(gid));
  }
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void mlp<InT, WtT, OutT>::wts_shuffle(std::vector<uint8_t> &bo_map,
                                      uint8_t *gate_weights, uint8_t *gate_zp,
                                      float *gate_scales, float *gate_bias,
                                      uint8_t *up_weights, uint8_t *up_zp,
                                      float *up_scales, float *up_bias) {
  bo_map.resize(wts_size_, 0);
  // N = 8192   sv_n = 64     sv_n_num = 128
  // K = 3072   sv_k = 128    sv_k_num = 24
  for (uint64_t c = 0; c < sv_n_num_; c++) {
    for (uint64_t r = 0; r < sv_k_num_ + 1; r++) {
      for (uint64_t zigzag_i = 0; zigzag_i < 2; zigzag_i++) {
        if (r == 0) {
          // first pack the bias (bf16)
          size_t bias_dst_base =
              ((sv_k_num_ + 1) * c * 2 + zigzag_i) * wts_vec_size_;
          size_t bias_src_base = (2 * c + zigzag_i) * (sv_n / 2);
          for (uint64_t bias_i = 0; bias_i < sv_n / 2; bias_i++) {
            uint16_t *uint16_ptr = reinterpret_cast<uint16_t *>(
                &bo_map[bias_dst_base + bias_i * 2]);
            *uint16_ptr = float_to_bfloat16(gate_bias[bias_src_base + bias_i]);
          }
          for (uint64_t bias_i = 0; bias_i < sv_n / 2; bias_i++) {
            uint16_t *uint16_ptr = reinterpret_cast<uint16_t *>(
                &bo_map[bias_dst_base + sv_n + bias_i * 2]);
            *uint16_ptr = float_to_bfloat16(up_bias[bias_src_base + bias_i]);
          }
          for (uint64_t bias_i = 2 * sv_n; bias_i < wts_vec_size_; bias_i++) {
            bo_map[bias_dst_base + bias_i] = 0;
          }
        } else {
          // format quantized weights (int4/uint4)
          int64_t src_r = r - 1;
          size_t quant_sv_dst_base =
              ((sv_k_num_ + 1) * 2 * c + 2 * r + zigzag_i) * wts_vec_size_;
          size_t src_sv_N_base = (2 * c + zigzag_i) * sv_n / 2;
          size_t src_sv_K_base = src_r * sv_k;
          for (uint64_t quant_iter_n = 0; quant_iter_n < sv_n / 8;
               quant_iter_n++) {
            for (uint64_t quant_iter_k = 0; quant_iter_k < sv_k / 8;
                 quant_iter_k++) {
              // format n8k8 and split
              size_t n8k8_dst_base = quant_sv_dst_base +
                                     8 * sv_k * quant_iter_n / 2 +
                                     32 * quant_iter_k;
              if (quant_iter_n < sv_n / 8 / 2) {
                // for gate quant
                size_t gate_src_n8k8_N_base = src_sv_N_base + quant_iter_n * 8;
                size_t gate_src_n8k8_K_base = src_sv_K_base + quant_iter_k * 8;

                int x_to_pack = -1;
                int y_to_pack = -1;
                for (int n8k8_k_index = 0; n8k8_k_index < 8; n8k8_k_index++) {
                  for (int n8k8_n_index = 0; n8k8_n_index < 8; n8k8_n_index++) {
                    size_t gate_src_n8k8_N =
                        gate_src_n8k8_N_base + n8k8_n_index;
                    size_t gate_src_n8k8_K =
                        gate_src_n8k8_K_base + n8k8_k_index;
                    int8_t value = static_cast<int8_t>(
                        gate_weights[gate_src_n8k8_N * K_ / 2 +
                                     gate_src_n8k8_K / 2]);
                    if (n8k8_k_index & 1) {
                      if (n8k8_n_index & 1) {
                        if (b_dtype_ == "int4") {
                          y_to_pack =
                              static_cast<int>(((value & 0xF0) >> 4) - 8);
                        } else {
                          y_to_pack = static_cast<int>((value & 0xF0) >> 4);
                        }
                      } else {
                        if (b_dtype_ == "int4") {
                          x_to_pack =
                              static_cast<int>(((value & 0xF0) >> 4) - 8);
                        } else {
                          x_to_pack = static_cast<int>((value & 0xF0) >> 4);
                        }
                      }
                    } else {
                      if (n8k8_n_index & 1) {
                        if (b_dtype_ == "int4") {
                          y_to_pack = static_cast<int>((value & 0xF) - 8);
                        } else {
                          y_to_pack = static_cast<int>(value & 0xF);
                        }
                      } else {
                        if (b_dtype_ == "int4") {
                          x_to_pack = static_cast<int>((value & 0x0F) - 8);
                        } else {
                          x_to_pack = static_cast<int>(value & 0x0F);
                        }
                      }
                    }
                    if (n8k8_n_index & 1) {
                      if (b_dtype_ == "int4") {
                        bo_map[n8k8_dst_base + n8k8_k_index * 8 / 2 +
                               n8k8_n_index / 2] =
                            static_cast<uint8_t>(
                                pack_v2int4(x_to_pack, y_to_pack));
                      } else {
                        bo_map[n8k8_dst_base + n8k8_k_index * 8 / 2 +
                               n8k8_n_index / 2] =
                            static_cast<uint8_t>(
                                pack_v2uint4(x_to_pack, y_to_pack));
                      }
                    }
                  }
                }
              } else {
                // for up quant
                size_t up_quant_iter_n_index = quant_iter_n - 4;
                size_t up_src_n8k8_N_base =
                    src_sv_N_base + up_quant_iter_n_index * 8;
                size_t up_src_n8k8_K_base = src_sv_K_base + quant_iter_k * 8;
                int x_to_pack = -1;
                int y_to_pack = -1;
                for (int n8k8_k_index = 0; n8k8_k_index < 8; n8k8_k_index++) {
                  for (int n8k8_n_index = 0; n8k8_n_index < 8; n8k8_n_index++) {
                    size_t up_src_n8k8_N = up_src_n8k8_N_base + n8k8_n_index;
                    size_t up_src_n8k8_K = up_src_n8k8_K_base + n8k8_k_index;
                    int8_t value = static_cast<int8_t>(
                        up_weights[up_src_n8k8_N * K_ / 2 + up_src_n8k8_K / 2]);
                    if (n8k8_k_index & 1) {
                      if (n8k8_n_index & 1) {
                        if (b_dtype_ == "int4") {
                          y_to_pack =
                              static_cast<int>(((value & 0xF0) >> 4) - 8);
                        } else {
                          y_to_pack = static_cast<int>((value & 0xF0) >> 4);
                        }
                      } else {
                        if (b_dtype_ == "int4") {
                          x_to_pack =
                              static_cast<int>(((value & 0xF0) >> 4) - 8);
                        } else {
                          x_to_pack = static_cast<int>((value & 0xF0) >> 4);
                        }
                      }
                    } else {
                      if (n8k8_n_index & 1) {
                        if (b_dtype_ == "int4") {
                          y_to_pack = static_cast<int>((value & 0xF) - 8);
                        } else {
                          y_to_pack = static_cast<int>(value & 0xF);
                        }
                      } else {
                        if (b_dtype_ == "int4") {
                          x_to_pack = static_cast<int>((value & 0x0F) - 8);
                        } else {
                          x_to_pack = static_cast<int>(value & 0x0F);
                        }
                      }
                    }
                    if (n8k8_n_index & 1) {
                      if (b_dtype_ == "int4") {
                        bo_map[n8k8_dst_base + n8k8_k_index * 8 / 2 +
                               n8k8_n_index / 2] =
                            static_cast<uint8_t>(
                                pack_v2int4(x_to_pack, y_to_pack));
                      } else {
                        bo_map[n8k8_dst_base + n8k8_k_index * 8 / 2 +
                               n8k8_n_index / 2] =
                            static_cast<uint8_t>(
                                pack_v2uint4(x_to_pack, y_to_pack));
                      }
                    }
                  }
                }
              }
            }
          }

          size_t zp_sv_dst_base = quant_sv_dst_base + quants_sv_size_;

          // pack zp uint4
          // zp shape {N * sv_k_num * uint4}
          size_t zp_src_base = (2 * c + zigzag_i) * (sv_n / 2 / 2) * sv_k_num_;
          for (uint64_t zp_i = 0; zp_i < sv_n / 2; zp_i += 2) {
            int x = 0;
            int y = 0;
            uint8_t x_value =
                gate_zp[zp_src_base + zp_i * sv_k_num_ / 2 + src_r / 2];
            uint8_t y_value =
                gate_zp[zp_src_base + (zp_i + 1) * sv_k_num_ / 2 + src_r / 2];
            if (src_r & 1) {
              x = static_cast<int>((x_value & 0xF0) >> 4);
              y = static_cast<int>((y_value & 0xF0) >> 4);
            } else {
              x = static_cast<int>(x_value & 0xF);
              y = static_cast<int>(y_value & 0xF);
            }

            if (b_dtype_ == "int4") {
              bo_map[zp_sv_dst_base + zp_i / 2] =
                  static_cast<uint8_t>(pack_v2int4(x, y));
            } else {
              bo_map[zp_sv_dst_base + zp_i / 2] = pack_v2uint4(x, y);
            }
          }
          for (uint64_t zp_i = 0; zp_i < sv_n / 2; zp_i += 2) {
            int x = 0;
            int y = 0;
            uint8_t x_value =
                up_zp[zp_src_base + zp_i * sv_k_num_ / 2 + src_r / 2];
            uint8_t y_value =
                up_zp[zp_src_base + (zp_i + 1) * sv_k_num_ / 2 + src_r / 2];
            if (src_r & 1) {
              x = static_cast<int>((x_value & 0xF0) >> 4);
              y = static_cast<int>((y_value & 0xF0) >> 4);
            } else {
              x = static_cast<int>(x_value & 0xF);
              y = static_cast<int>(y_value & 0xF);
            }
            if (b_dtype_ == "int4") {
              bo_map[zp_sv_dst_base + sv_n / 2 / 2 + zp_i / 2] =
                  pack_v2int4(x, y);
            } else {
              bo_map[zp_sv_dst_base + sv_n / 2 / 2 + zp_i / 2] =
                  pack_v2uint4(x, y);
            }
          }
          for (uint64_t zp_i = sv_n / 2; zp_i < zp_sv_size_; zp_i++) {
            bo_map[zp_sv_dst_base + zp_i] = 0;
          }

          size_t scale_sv_dst_base = zp_sv_dst_base + zp_sv_size_;
          // pack scale bf16
          // scale shape(N * K / sv_k) bf16 -> {N * sv_k_num * bf16}
          size_t scale_sv_src_base =
              (2 * c + zigzag_i) * (sv_n / 2) * sv_k_num_;
          for (uint64_t scale_i = 0; scale_i < sv_n / 2; scale_i++) {
            uint16_t *uint16_ptr = reinterpret_cast<uint16_t *>(
                &bo_map[scale_sv_dst_base + scale_i * 2]);
            *uint16_ptr = float_to_bfloat16(
                gate_scales[scale_sv_src_base + scale_i * sv_k_num_ + src_r]);
          }
          for (uint64_t scale_i = 0; scale_i < sv_n / 2; scale_i++) {
            uint16_t *uint16_ptr = reinterpret_cast<uint16_t *>(
                &bo_map[scale_sv_dst_base + sv_n + scale_i * 2]);
            *uint16_ptr = float_to_bfloat16(
                up_scales[scale_sv_src_base + scale_i * sv_k_num_ + src_r]);
          }
        }
      } // zigzag
    }
  }
}

template <typename InT, typename WtT, typename OutT>
void mlp<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  DD_ASSERT(const_params.size() == 8,
            OpsFusion::dd_format("FlatMLP expects one constant. Got {}",
                                 const_params.size()));
  uint8_t *gate_weights = (uint8_t *)const_params.at(0).data;
  uint8_t *gate_zp = (uint8_t *)const_params.at(2).data;
  float *gate_scales = (float *)const_params.at(1).data;
  float *gate_bias = (float *)const_params.at(3).data;

  uint8_t *up_weights = (uint8_t *)const_params.at(4).data;
  uint8_t *up_zp = (uint8_t *)const_params.at(6).data;
  float *up_scales = (float *)const_params.at(5).data;
  float *up_bias = (float *)const_params.at(7).data;

  std::vector<uint8_t> bo_map;
  wts_shuffle(bo_map, gate_weights, gate_zp, gate_scales, gate_bias, up_weights,
              up_zp, up_scales, up_bias);
  io.write(0, bo_map.data(), wts_size_);
}

template <typename InT, typename WtT, typename OutT>
void mlp<InT, WtT, OutT>::cal_shuffled_wts_size(int64_t N, int64_t K) {
  N_ = N;
  K_ = K;
  sv_k_num_ = (K_ - 1) / sv_k + 1;
  sv_n_num_ = (N_ - 1) / sv_n + 1;
  total_sv_k_num_ = sv_k_num_ + 1; // add the bias row
  wts_size_ = total_sv_k_num_ * sv_n_num_ * wts_vec_size_ * 2;

  wts_bo_size_ = N_ * K_ / 2;
  scale_bo_size_ = N_ * K_ / sv_k * sizeof(float);
  zp_bo_size_ = N_ * K_ / sv_k / 2;
  bias_bo_size_ = N_ * sizeof(float);
}

template <typename InT, typename WtT, typename OutT>
void mlp<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("flat mlp initialize_const_params ...");

  DD_ASSERT(const_params.size() == 8,
            OpsFusion::dd_format("FlatMLP expects one constant. Got {}",
                                 const_params.size()));
  cal_shuffled_wts_size(N_, K_);
  const int gid = 0;
  a_bo_ = xrt::bo(xrt_ctx_->get_device(), ifm_size_, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(gid));
  b_bo_ = xrt::bo(xrt_ctx_->get_device(), wts_size_, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(gid));

  c_bo_ = xrt::bo(xrt_ctx_->get_device(), ofm_size_, XRT_BO_FLAGS_HOST_ONLY,
                  xrt_ctx_->get_kernel().group_id(gid));

  // init const mask
  WtT *b_bo_map = b_bo_.map<WtT *>();
  auto bo_const = BoConst(b_bo_map);
  initialize_const_params(bo_const, const_params, attr);
  b_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

template <typename InT, typename WtT, typename OutT>
void mlp<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                  std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("mlp execute ...");
  DD_ASSERT(input.size() == 1,
            OpsFusion::dd_format("Flat MLP input tensor expects 1. Got {}",
                                 input.size()));
  DD_ASSERT(output.size() == 1,
            OpsFusion::dd_format("Flat MLP output tensor expects 1. Got {}",
                                 output.size()));
  // inputs
  a_bo_.write(input.at(0).data);
  a_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // prepare inst_bo
  std::vector<size_t> param_shape = {size_t(M_), size_t(K_), size_t(N_)};
  auto instr_bo_key = get_instr_key(txn_fname_prefix_, param_shape);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  size_t instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  xrt::run run;
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  // param order to be confirmed
  run = kernel_(2, instr_bo, instr_bo_words,
                a_bo_.address() + DDR_AIE_ADDR_OFFSET,
                b_bo_.address() + DDR_AIE_ADDR_OFFSET,
                c_bo_.address() + DDR_AIE_ADDR_OFFSET, 0);
  run.wait2();
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);
  num_run_aie_++;

  c_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  c_bo_.read(output.at(0).data);
}

template <typename InT, typename WtT, typename OutT>
std::vector<xrt::bo> mlp<InT, WtT, OutT>::get_inputs() {
  return {a_bo_, b_bo_};
}

template <typename InT, typename WtT, typename OutT>
std::vector<xrt::bo> mlp<InT, WtT, OutT>::get_outputs() {
  return {c_bo_};
}

template <typename InT, typename WtT, typename OutT>
bool mlp<InT, WtT, OutT>::create_bo(void *usr_ptr, size_t size,
                                    int operand_index) {
  std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(usr_ptr);
  constexpr std::uint32_t MASK = ((1 << 12) - 1);
  if ((addr & MASK) != 0) {
    return false;
  }
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
    return false;
  }
  return true;
}

template <typename InT, typename WtT, typename OutT>
void mlp<InT, WtT, OutT>::execute(std::vector<xrt::bo> &input,
                                  std::vector<xrt::bo> &output, bool wait) {
  std::string txn_key = txn_fname_prefix_ + "_" + std::to_string(M_) + "_" +
                        std::to_string(K_) + "_" + std::to_string(N_);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(txn_key);
  auto instr_bo_words = uint32_t(instr_bo.size() / sizeof(int));
  auto kernel_ = xrt_ctx_->get_kernel();

  ryzenai::dynamic_dispatch::execute_kernel(kernel_, 2, instr_bo,
                                            instr_bo_words, input[0], input[1],
                                            output[0], 0, 0, wait, false);
}

template <typename InT, typename WtT, typename OutT>
void mlp<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> mlp<InT, WtT, OutT>::get_transaction_bin() const {
  std::string txn_key = txn_fname_prefix_ + "_" + std::to_string(M_) + "_" +
                        std::to_string(K_) + "_" + std::to_string(N_);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> mlp<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return this->get_transaction_bin();
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> mlp<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, ifm_size_},
      {OpArgMap::OpArgType::CONST_INPUT, 1, 1, 0, wts_size_},
      {OpArgMap::OpArgType::OUTPUT, 2, 9, 0, ofm_size_}};
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("flat mlp argmap : {}", cvt_to_string(arg_map)));

  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
std::once_flag mlp<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t mlp<InT, WtT, OutT>::mlp_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag mlp<InT, WtT, OutT>::instr_reg_flag_;

template class mlp<std::uint16_t, std::uint8_t, std::uint16_t>;
} // namespace flat
} // namespace ryzenai
