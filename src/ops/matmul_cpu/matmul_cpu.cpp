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

#include <ops/matmul_cpu/matmul_cpu.hpp>
#include <ops/op_interface.hpp>
#include <ops/ops_common/ctrlpkt.hpp>
#include <txn_container.hpp>
#include <utils/instruction_registry.hpp>

#include <utils/logging.hpp>
#include <utils/meta_utils.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>
#include <xrt_context/xrt_context.hpp>

// AIE Driver header
#include "ops/ops_common/matmul_matrix.hpp"
#include "utils/ctrl_pkt_utils.hpp"
#include "xaiengine.h"

using namespace matmul_matrix;

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

template <typename InT, typename WtT, typename OutT>
std::tuple<size_t, size_t, size_t>
matmul_cpu<InT, WtT, OutT>::map_padded_shape(size_t M, size_t K,
                                             size_t N) const {
  size_t Mo = M;
  size_t Ko = K;
  size_t No = N;
  return std::make_tuple(Mo, Ko, No);
}

/* Utility function to set the kernel shape based on the weights dimensions
 * Pick kernel shape using weight matrix size
 * Select OPT shapes when a_type is int8
 * Select Llamav2 shapes when a_type is int16
 * Need to fix this to pick shapes independent of the datatype*/
template <typename InT, typename WtT, typename OutT>
void matmul_cpu<InT, WtT, OutT>::set_kernel_shapes() {}

template <typename InT, typename WtT, typename OutT>
std::string matmul_cpu<InT, WtT, OutT>::get_instr_key(std::string prefix,
                                                      size_t m, size_t k,
                                                      size_t n) const {
  return "";
}

/*
 * matmul_cpu class constructor
 */
template <typename InT, typename WtT, typename OutT>
matmul_cpu<InT, WtT, OutT>::matmul_cpu(
    const std::string &a_dtype, const std::string &b_dtype,
    const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {

  a_dtype_ = a_dtype;
  b_dtype_ = b_dtype;
  c_dtype_ = c_dtype;
  a_dtype_size_ = sizeof(InT);
  b_dtype_size_ = sizeof(WtT);
  c_dtype_size_ = sizeof(OutT);

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

  if (design_param_.find("4x4") != std::string::npos) { // 4x4 design
    if (attr.count("input_shape") &&
        attr.at("input_shape").type() == typeid(std::vector<int>)) {
      const auto &input_shape_vector =
          std::any_cast<const std::vector<int> &>(attr.at("input_shape"));

      if (input_shape_vector.size() == 4) {
        inputShape_[0] = input_shape_vector[0];
        inputShape_[1] = input_shape_vector[1];
        inputShape_[2] = input_shape_vector[2];
        inputShape_[3] = input_shape_vector[3];
        RYZENAI_LOG_TRACE(
            "matmul_cpu: InputShape: " + std::to_string(input_shape_vector[0]) +
            ", " + std::to_string(input_shape_vector[1]) + ", " +
            std::to_string(input_shape_vector[2]) + ", " +
            std::to_string(input_shape_vector[3]));
      } else if (input_shape_vector.size() ==
                 3) { // PSW case. input_shape_vector[0] is 1. Batch matmul_cpu
                      // we have separate op
        inputShape_[0] = input_shape_vector[1];
        inputShape_[1] = input_shape_vector[2];
        inputShape_[2] = 0;
        inputShape_[3] = 0;
        RYZENAI_LOG_TRACE(
            "matmul_cpu: InputShape: " + std::to_string(input_shape_vector[0]) +
            ", " + std::to_string(input_shape_vector[1]) + ", " +
            std::to_string(input_shape_vector[2]));
      } else {
        std::cout
            << "Input Shape attribute does not have the expected number of "
               "elements.Number of passed : input_shape_vector.size(), "
               "Expected:4"
            << std::endl;
      }
    } else {
      std::cout << "Input Shape attribute not found or not of correct type."
                << std::endl;
    }
  }

  std::call_once(logger_flag_, []() {
    std::string header =
        "matmul_cpu_id M K N kernel_m kernel_k kernel_n Execute"
        "time(us) num_aie_runs run_aie_time(ns) "
        "A_copy_time(ns) A_sync_time(ns) "
        "C_copy_time(ns) C_sync_time(ns) "
        "Avg_time_per_aie_run(ns)\n";
    RYZENAI_LOG_INFO(header);
  });
}

template <typename InT, typename WtT, typename OutT>
void matmul_cpu<InT, WtT, OutT>::set_params(const std::string &model_name,
                                            std::vector<size_t> input_shape) {}

/*
 * copy weight matrix into XRT BOs with padding and tiling
 *
 * this method copies the weight matrix into XRT BOs. This is re-used for
 * every matmul_cpu performed for this object with different activations. weight
 * matrix is padded, tiled and reformatted while copying to XRT BOs. padding
 * is done to align with kernel_y_shape each tile of the weight matrix is of
 * shape kernel_y_shape this method also reformats the matrix b/weight matrix
 * as required by AIE/IPU matmul_cpu implementation
 *
 * @param weights pointer to the weight matrix
 * @param w_shape tuple containing the shape of the weight matrix
 *
 * @return none
 */

template <typename InT, typename WtT, typename OutT>
void matmul_cpu<InT, WtT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("matmul_cpu initialize_const_params(ptr) ...");
  w_shape_[0] = const_params.at(0).shape.at(0);
  w_shape_[1] = const_params.at(0).shape.at(1);

  auto K_raw = w_shape_[0];
  auto N_raw = w_shape_[1];

  auto weights = (WtT *)const_params.at(0).data;
  auto qdq = (int64_t *)const_params.at(1).data;
  auto qdq_params = (int32_t *)const_params.at(2).data;

  size_t weights_size = w_shape_[0] * w_shape_[1] * sizeof(WtT);
  auto qdq_size = w_shape_[1] * sizeof(int64_t);
  auto qdq_params_size = matmul_matrix::QDQparam_size * sizeof(int32_t);

  size_t write_offset = 0;
  io.write(write_offset, (void *)weights, weights_size);
  write_offset += weights_size;
  io.write(write_offset, (void *)qdq, qdq_size);
  write_offset += qdq_size;
  io.write(write_offset, (void *)qdq_params, qdq_params_size);
  write_offset += qdq_params_size;

  RYZENAI_LOG_TRACE("matmul_cpu initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename WtT, typename OutT>
void matmul_cpu<InT, WtT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {}
/*
 * execute matrix multiplication c = a * w
 *
 * perform matmul_cpu c = a * w. w is stored in the object with
 * initilize_weights method.
 *
 * @param a pointer to activation matrix
 * @param a_shape tuple containing the shape of the activation matrix
 * @param c pointer to store the result of matmul_cpu
 *
 * @return none
 */

template <typename InT, typename WtT, typename OutT>
void matmul_cpu<InT, WtT, OutT>::execute_cpu(std::vector<Tensor> &input,
                                             void *consts,
                                             std::vector<Tensor> &output) {
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

  auto [M, K, N] = map_padded_shape(a_shape_[0], a_shape_[1], c_shape_[1]);
  std::vector<WtT> b(K * N, 1);
  std::vector<InT> a(M * K);
  std::memcpy(a.data(), input.at(0).data, M * K * sizeof(InT));
  std::vector<OutT> out(M * N, 0);
  b.resize(N * K);
  std::memcpy(b.data(), consts, K * N * sizeof(WtT));
  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < K; ++k) {
      for (size_t j = 0; j < N; ++j) {
        out[i * N + j] += a[i * K + k] * b[k * N + j];
      }
    }
  }
  std::memcpy(output.at(0).data, out.data(), M * N * sizeof(OutT));
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
void matmul_cpu<InT, WtT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> matmul_cpu<InT, WtT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  std::vector<uint8_t> txn_vec(sizeof(XAie_TxnHeader), 0);
  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn_vec.data();
  Hdr->TxnSize = uint32_t(sizeof(XAie_TxnHeader));
  Hdr->NumOps = 0;
  return txn_vec;
}

template <typename InT, typename WtT, typename OutT>
const std::vector<uint8_t> matmul_cpu<InT, WtT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  std::vector<uint8_t> vec;
  return vec;
}

template <typename InT, typename WtT, typename OutT>
std::vector<OpArgMap> matmul_cpu<InT, WtT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  // input --> [input, weights, output]
  // Check if IO buffers have batch.
  auto [M, K, N] = extract_MKN(input);
  auto [Mo, Ko, No] = map_padded_shape(M, K, N);
  // qdqc
  size_t size_interleaved_qdq = Ko * No * sizeof(int64_t);
  size_interleaved_qdq += matmul_matrix::QDQparam_size * sizeof(int32_t);

  size_t const_params_bo_size =
      (Ko * No * b_dtype_size_) + size_interleaved_qdq;
  size_t input_bo_size = (Mo * Ko * a_dtype_size_);
  size_t output_bo_size = (Mo * No * c_dtype_size_);

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, 4, 0, output_bo_size}};
  RYZENAI_LOG_TRACE(
      OpsFusion::dd_format("matmul_cpu Argmap : {}", cvt_to_string(arg_map)));
  return arg_map;
};

template <typename InT, typename WtT, typename OutT>
void matmul_cpu<InT, WtT, OutT>::format_output(
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
    RYZENAI_LOG_TRACE("matmul_cpu: input_shape: " + std::to_string(M) + ", " +
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
      N = orig_output_shape_vector[2] * orig_output_shape_vector[0];
    } else if (orig_output_shape_vector.size() == 4) {
      N = orig_output_shape_vector[3];
    } else {
      std::cout
          << "output shape attribute does not have the expected number of "
             "elements.Number of passed : design_param_vector.size(), "
             "Expected:3"
          << std::endl;
    }
    RYZENAI_LOG_TRACE("matmul_cpu: output_shape: " + std::to_string(M) + ", " +
                      std::to_string(N));
  } else {
    N = out_tensor.shape.at(2);
  }
  // get the mapped shape
  auto [Mo, Ko, No] = map_padded_shape(M, K, N);
  // K, N is the dst.shape
  auto aie_out = (void *)out_tensor.data;

  if (sz != Mo * No * c_dtype_size_) {
    throw std::runtime_error("matmul_cpu : The size of hw_out is not correct.");
  }

  if (N == No) {
    RYZENAI_LOG_TRACE("Triggering matmul_cpu Output Memcpy");
    memcpy((void *)aie_out, (void *)hw_out_ptr, (M * No * c_dtype_size_));
  } else {
    RYZENAI_LOG_TRACE("Triggering matmul_cpu Output Strided Memcpy");
    for (int i = 0; i < M; i++) {
      memcpy(
          (void *)(static_cast<uint8_t *>(aie_out) + i * N * c_dtype_size_),
          (void *)(static_cast<uint8_t *>(hw_out_ptr) + i * No * c_dtype_size_),
          (N * c_dtype_size_));
    }
  }
}

template <typename InT, typename WtT, typename OutT>
std::once_flag matmul_cpu<InT, WtT, OutT>::logger_flag_;

template <typename InT, typename WtT, typename OutT>
uint64_t matmul_cpu<InT, WtT, OutT>::matmul_cpu_count = 0;

template class matmul_cpu<uint8_t, uint8_t, uint8_t>;
template class matmul_cpu<uint16_t, uint8_t, uint16_t>;

} // namespace ryzenai
