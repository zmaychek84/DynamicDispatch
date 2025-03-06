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
#include <fstream>
#include <iostream>
#include <map>
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

#include "ops/ops_common/help_file.hpp"
#include "ops/ops_common/matmul_matrix.hpp"
#include "utils/ctrl_pkt_utils.hpp"
#include <ops/ops_common/ctrlpkt.hpp>
#include <ops/quantizelinear_cpu/quantizelinear_cpu.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>
// AIE Driver header
#include <xaiengine.h>
using namespace matmul_matrix;

namespace ryzenai {
static std::tuple<size_t, size_t>
extract_MK(const std::vector<Tensor> &inputs) {
  size_t M = 0;
  size_t K = 0;
  if (inputs.at(0).shape.size() == 2) {
    M = inputs.at(0).shape.at(0);
    K = inputs.at(0).shape.at(1);
  } else if (inputs.at(0).shape.size() == 3) {
    M = (inputs.at(0).shape.at(1) * inputs.at(0).shape.at(0));
    K = inputs.at(0).shape.at(2);
  } else if (inputs.at(0).shape.size() == 4) {
    if (inputs.at(0).shape.at(1) == inputs.at(0).shape.at(2)) { // NHWC
      M = inputs.at(0).shape.at(0) * inputs.at(0).shape.at(1) *
          inputs.at(0).shape.at(2);
      K = inputs.at(0).shape.at(3);
    } else { // NCHW
      M = inputs.at(0).shape.at(2) * inputs.at(0).shape.at(3);
      K = inputs.at(0).shape.at(1);
    }
  }
  return std::make_tuple(M, K);
}

template <typename InT, typename OutT>
std::once_flag quantizelinear_cpu<InT, OutT>::logger_flag_;

template <typename InT, typename OutT>
uint64_t quantizelinear_cpu<InT, OutT>::quantizelinear_cpu_count = 0;

template <typename InT, typename OutT>
void quantizelinear_cpu<InT, OutT>::debug(bool enable) {
  debug_ = enable;
}

template <typename InT, typename OutT>
std::string quantizelinear_cpu<InT, OutT>::get_instr_key(std::string prefix,
                                                         size_t m,
                                                         size_t k) const {
  return "";
}

template <typename InT, typename OutT>
const std::vector<uint8_t> quantizelinear_cpu<InT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  std::vector<uint8_t> txn_vec(sizeof(XAie_TxnHeader), 0);
  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn_vec.data();
  Hdr->TxnSize = uint32_t(sizeof(
      XAie_TxnHeader)); // transactions header size without any instructions
  Hdr->NumOps = 0;
  return txn_vec;
}

template <typename InT, typename OutT>
const std::vector<uint8_t>
quantizelinear_cpu<InT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  std::vector<uint8_t> vec;
  return vec;
}

template <typename InT, typename OutT>
std::vector<OpArgMap> quantizelinear_cpu<InT, OutT>::get_buffer_reqs(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  // input --> [input, gamma, beta, output]
  // Check if IO buffers have batch.
  auto [M, N] = extract_MK(input);

  size_t const_params_bo_size = sizeof(InT) + sizeof(OutT);
  size_t input_bo_size = (M * N * sizeof(InT));
  size_t output_bo_size = (M * N * sizeof(OutT));

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_bo_size},
      {OpArgMap::OpArgType::CONST_INPUT, 2, 1, 0, const_params_bo_size},
      {OpArgMap::OpArgType::OUTPUT, 0, 3, 0, output_bo_size}};
  return arg_map;
};

template <typename InT, typename OutT>
quantizelinear_cpu<InT, OutT>::quantizelinear_cpu(
    const std::string &a_dtype, const std::string &c_dtype, bool load_xrt,
    const std::map<std::string, std::any> &attr) {
  a_dtype_ = a_dtype;
  c_dtype_ = c_dtype;

  a_dtype_size_ = sizeof(InT);
  c_dtype_size_ = sizeof(OutT);

  quantizelinear_cpu_id_ = quantizelinear_cpu_count++;

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
}

template <typename InT, typename OutT>
void quantizelinear_cpu<InT, OutT>::initialize_const_params(
    ConstBufferIO &io, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("quantizelinear_cpu initialize_const_params(ptr) ...");

  DD_THROW_IF(
      (const_params.size() != 2),
      OpsFusion::dd_format("Unsupported const spec for quantizelinear_cpu\n") +
          OpsFusion::dd_format("(Details : #const params == 1 ({})",
                               const_params.size()));

  auto const1 = (int32_t *)const_params.at(0).data;
  auto const2 = (int32_t *)const_params.at(1).data;
  io.write(0, (void *)const1, sizeof(InT));
  io.write(sizeof(InT), (void *)const2, sizeof(OutT));

  RYZENAI_LOG_TRACE("quantizelinear_cpu initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename OutT>
void quantizelinear_cpu<InT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  RYZENAI_LOG_TRACE("quantize_cpu initialize_const_params(ptr) ...");

  RYZENAI_LOG_TRACE("quantize_cpu initialize_const_params(ptr) ... DONE");
}

template <typename InT, typename OutT>
void quantizelinear_cpu<InT, OutT>::execute_cpu(std::vector<Tensor> &input,
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
    throw std::runtime_error(
        "QuantizeLinear : Invalid shape received for input");
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
    throw std::runtime_error(
        "QuantizeLinear : Invalid shape received for output");
  }

  size_t a_elem = a_shape_[0] * a_shape_[1];
  size_t c_elem = c_shape_[0] * c_shape_[1];

  if (c_elem != a_elem) {
    throw std::runtime_error(
        "QuantizeLinear : Input and output tensor sizes don't match.");
  }

  InT *inPtr = (InT *)input.at(0).data;
  OutT *outPtr = (OutT *)output.at(0).data;

  InT sc_val = *((InT *)consts);
  OutT zp_val = *((OutT *)consts + 2);

  cpu_runner_ops::QuantizeLinear((float *)inPtr, outPtr, a_elem, sc_val,
                                 zp_val);
}

template <typename InT, typename OutT>
void quantizelinear_cpu<InT, OutT>::execute(std::vector<Tensor> &input,
                                            std::vector<Tensor> &output) {}

template class quantizelinear_cpu<float, uint8_t>;
template class quantizelinear_cpu<float, int8_t>;
template class quantizelinear_cpu<float, uint16_t>;
template class quantizelinear_cpu<float, int16_t>;
} // namespace ryzenai
