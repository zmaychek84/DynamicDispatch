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
#include <ops/op_interface.hpp>
#include <ops/sd/mha.hpp>
#include <txn_container.hpp>
#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>

namespace ryzenai {
namespace sd {

template <typename InT, typename WtT, typename OutT>
void mha<InT, WtT, OutT>::setup_instr_registry() {
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
mha<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                   const std::vector<size_t> &shape) const {
  std::string out_str = prefix;
  for (size_t i = 0; i < shape.size(); i++) {
    out_str += "_" + std::to_string(shape[i]);
  }
  return out_str;
}

template <typename InT, typename WtT, typename OutT>
mha<InT, WtT, OutT>::mha(const std::string &a_dtype, const std::string &b_dtype,
                         const std::string &c_dtype, bool load_xrt,
                         const std::map<std::string, std::any> &attr) {
  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  txnbin_a_header = {{"bfloat16", "a16bf"}, {"bfp16ebs8", "a16bfp"}};
  txnbin_b_header = {{"bfloat16", "w16bf"}, {"bfp16ebs8", "w16bfp"}};
  txnbin_acc_header = {{"bfloat16", "acc16bf"}};
  txn_fname_prefix_ = "sd_mha_" + txnbin_a_header.at(a_dtype_) +
                      txnbin_b_header.at(b_dtype_) +
                      txnbin_acc_header.at(c_dtype_);
  // B, M, K, N
  // UNET layer1
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1024, 640, 1024});
  // UNET layer2
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1024, 640, 77});
  // UNET layer3
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 256, 1280, 256});
  // UNET layer4
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 256, 1280, 77});
  // UNET layer5
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 4096, 320, 4096});
  // UNET layer6
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 4096, 320, 77});
  // UNET layer7
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 64, 1280, 64});
  // UNET layer8
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 64, 1280, 77});
  // VAE layer1
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{1, 4096, 512, 4096});
  // SD3 512 mmdit aka SD 3.0 MHA_mmdit layer 1
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 1178, 1536, 1178});
  // SD 3.0 MHA_mmdit layer 2
  default_shapes_[txn_fname_prefix_].emplace_back(
      std::vector<size_t>{2, 4250, 1536, 4250});

  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

  mha_id_++;

  if (attr.count("input_shape") &&
      attr.at("input_shape").type() == typeid(std::vector<int>)) {
    const auto &input_shape_vector =
        std::any_cast<const std::vector<int> &>(attr.at("input_shape"));
    if (input_shape_vector.size() == 4) {
      B_ = input_shape_vector[0];
      M_ = input_shape_vector[1];
      K_ = input_shape_vector[2];
      N_ = input_shape_vector[3];
    } else {
      throw std::runtime_error(
          "SDMHA input_shape attr should be a vector of size 4");
    }
    RYZENAI_LOG_TRACE("SDMHA: InputShape: " + std::to_string(B_) + ", " +
                      std::to_string(M_) + ", " + std::to_string(K_) + ", " +
                      std::to_string(N_));
  } else {
    throw std::runtime_error(
        "SDMHA input_shape attr not found or not of correct type");
  }

  if (attr.count("num_heads") &&
      attr.at("num_heads").type() == typeid(std::vector<int>)) {
    const auto &heads_vector =
        std::any_cast<const std::vector<int> &>(attr.at("num_heads"));
    DD_ASSERT(heads_vector.size() == 1,
              OpsFusion::dd_format("SDMHA expects head size == 1. Got {}",
                                   heads_vector.size()));
    H_ = heads_vector[0];
  } else {
    throw std::runtime_error(
        "SDMHA num_heads attr not found or not of correct type");
  }

  q_size_ = B_ * M_ * K_ * sizeof(InT);
  k_size_ = B_ * K_ * N_ * sizeof(InT);
  // for VAE Decoder(both sd1.5 and sd3) batch = 1
  if (B_ == 1) {
    v_size_ = k_size_;
    out_size_ = q_size_;
    // scratch = bmm1 out size + softmax out size(same as bmm1 out)
    scratch_size_ = B_ * H_ * M_ * N_ * sizeof(InT) * 2;
    mask_size_ = M_ * sizeof(WtT);
    if (M_ == 16384) {
      // sd3.0 vae 1024 use a different xclbin
      XCLBIN_FNAME_ =
          OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\SD3MHA_VAE.xclbin";
    } else {
      // sd1.5 vae mha
      XCLBIN_FNAME_ =
          OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\SDMHA_VAE.xclbin";
    }
  } else {
    // sd1.5 unet or sd3 mmdit batch = 2
    if (K_ == 1536) {
      // mmdit mha, v need to pad, align N_ to 256
      auto N_256_aligned = Utils::align_to_next(N_, 256);
      v_size_ = B_ * K_ * N_256_aligned * sizeof(InT);
      out_size_ = v_size_;
      // mask align to 256
      mask_size_ = N_256_aligned * sizeof(WtT);
      // scratch also requires M to align to 256
      auto M_256_aligned = Utils::align_to_next(M_, 256);
      scratch_size_ = B_ * H_ * M_256_aligned * N_256_aligned * sizeof(InT);
    } else {
      // sd1.5 unet mha
      v_size_ = k_size_;
      out_size_ = q_size_;
      // align N_ to 4 for scratch size calculation
      auto N_4_aligned = Utils::align_to_next(N_, 4);
      // scratch = bmm1 out size(half of sd1.5 vae)
      scratch_size_ = B_ * H_ * M_ * N_4_aligned * sizeof(InT);
      // mask need a minimum size of 128 * size of WtT
      auto padded_mask_size = N_ < min_mask_size_ ? min_mask_size_ : N_;
      mask_size_ = padded_mask_size * sizeof(WtT);
    }
    XCLBIN_FNAME_ =
        OpInterface::get_dd_base_dir() + "\\xclbin\\stx\\SDMHA.xclbin";
  }

  // std::cerr << "SDMHA: B: " << B_ << ", M: " << M_ << ", K: " << K_
  //           << ", N: " << N_ << ", H: " << H_ << std::endl;
  // std::cerr << "SDMHA: q_size: " << q_size_ << ", k_size: " << k_size_
  //           << ", v_size: " << v_size_ << ", out_size: " << out_size_
  //           << ", scratch_size: " << scratch_size_
  //           << ", mask_size: " << mask_size_ << std::endl;
  // std::cerr << "XCLBIN: " << XCLBIN_FNAME_ << std::endl;

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

  RYZENAI_LOG_TRACE("[SD mha] ID: " + std::to_string(mha_id_) + ", XCLBIN: " +
                    XCLBIN_FNAME_ + ", (a_dtype, b_dtype, c_dtype): (" +
                    a_dtype + ", " + b_dtype + ", " + c_dtype + ")");
}

template <typename InT, typename WtT, typename OutT>
void mha<InT, WtT, OutT>::set_params() {
  // shape already set in constructor
  xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME_);
  std::call_once(instr_reg_flag_, [this]() { setup_instr_registry(); });
}

template <typename InT, typename WtT, typename OutT>
void mha<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  io.write(0, const_params.at(0).data, mask_size_);
}

template <typename InT, typename WtT, typename OutT>
void mha<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("SD mha initialize_const_params ...");
  DD_ASSERT((const_params.size() == 1),
            OpsFusion::dd_format("SDMHA expects one constant. Got {}",
                                 const_params.size()));
  // confirm group_ids
  const int gid = 0;
  qkv_bo_ =
      xrt::bo(xrt_ctx_->get_device(), q_size_ + k_size_ + v_size_,
              XRT_BO_FLAGS_HOST_ONLY, xrt_ctx_->get_kernel().group_id(gid));
  const_mask_bo_ =
      xrt::bo(xrt_ctx_->get_device(), mask_size_, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(gid));
  scratch_bo_ =
      xrt::bo(xrt_ctx_->get_device(), scratch_size_, XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(gid));
  out_bo_ = xrt::bo(xrt_ctx_->get_device(), out_size_, XRT_BO_FLAGS_HOST_ONLY,
                    xrt_ctx_->get_kernel().group_id(gid));

  // init const mask
  WtT *const_bo_map = const_mask_bo_.map<WtT *>();
  auto bo_const = BoConst(const_bo_map);
  initialize_const_params(bo_const, const_params, attr);
  const_mask_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

template <typename InT, typename WtT, typename OutT>
void mha<InT, WtT, OutT>::execute(std::vector<Tensor> &input,
                                  std::vector<Tensor> &output) {
  RYZENAI_LOG_TRACE("mha execute ...");
  DD_ASSERT(input.size() == 1,
            OpsFusion::dd_format("sd mha input tensor expects 1. Got {}",
                                 input.size()));
  DD_ASSERT(output.size() == 1,
            OpsFusion::dd_format("sd mha output tensor expects 1. Got {}",
                                 output.size()));
  // inputs
  qkv_bo_.write(input.at(0).data);
  qkv_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // prepare inst_bo
  std::vector<size_t> param_shape = {B_, M_, K_, N_};
  auto instr_bo_key = get_instr_key(txn_fname_prefix_, param_shape);
  auto instr_bo = xrt_ctx_->get_registry().get_instr_bo(instr_bo_key);
  size_t instr_bo_words = instr_bo.size() / sizeof(int);

  auto kernel_ = xrt_ctx_->get_kernel();
  // launch the kernel
  auto run_aie_start = GET_ELAPSED_TIME_NS();
  // param order to be confirmed

  ryzenai::dynamic_dispatch::execute_kernel(
      kernel_, 2, instr_bo, instr_bo_words, qkv_bo_, scratch_bo_,
      const_mask_bo_, out_bo_, 0, true, false);
  auto run_aie_stop = GET_ELAPSED_TIME_NS();
  run_aie_time_ += static_cast<int64_t>(run_aie_stop - run_aie_start);
  num_run_aie_++;

  out_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  out_bo_.read(output.at(0).data);
}

template <typename InT, typename WtT, typename OutT>
void mha<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> mha<InT, WtT, OutT>::get_transaction_bin() const {
  std::string txn_key = txn_fname_prefix_ + "_" + std::to_string(B_) + "_" +
                        std::to_string(M_) + "_" + std::to_string(K_) + "_" +
                        std::to_string(N_);
  Transaction &txn = Transaction::getInstance();
  return txn.get_txn_bvec(txn_key);
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> mha<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  return this->get_transaction_bin();
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> mha<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  // mask has the same size as scratch
  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, q_size_},
      {OpArgMap::OpArgType::INPUT, 0, 1, q_size_, k_size_},
      {OpArgMap::OpArgType::INPUT, 0, 2, q_size_ + k_size_, v_size_},
      {OpArgMap::OpArgType::SCRATCH_PAD, 1, 0, 0, scratch_size_},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 3, 0, mask_size_},
      {OpArgMap::OpArgType::OUTPUT, 3, 4, 0, out_size_}};
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("SD mha argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
std::once_flag mha<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t mha<InT, WtT, OutT>::mha_count = 0;

template <typename InT, typename WtT, typename OutT>
std::once_flag mha<InT, WtT, OutT>::instr_reg_flag_;

template class mha<std::uint16_t, std::uint16_t, std::uint16_t>;
} // namespace sd
} // namespace ryzenai
