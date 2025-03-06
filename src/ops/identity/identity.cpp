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

// #include "ops/ops_common/silu_lut_bf16_512.h"
#include "utils/ctrl_pkt_utils.hpp"
#include <ops/ops_common/ctrlpkt.hpp>
#include <utils/logging.hpp>
#include <utils/utils.hpp>
// AIE Driver header
#include <xaiengine.h>

#include <ops/identity/identity.hpp>

namespace ryzenai {

const std::vector<uint8_t> identity::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) const {
  std::vector<uint8_t> txn_vec(sizeof(XAie_TxnHeader), 0);
  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn_vec.data();
  Hdr->TxnSize = uint32_t(sizeof(
      XAie_TxnHeader)); // transactions header size without any instructions
  Hdr->NumOps = 0;
  return txn_vec;
}

std::vector<OpArgMap>
identity::get_buffer_reqs(std::vector<Tensor> &input,
                          std::vector<Tensor> &output,
                          const std::map<std::string, std::any> &attr) const {

  size_t num_dims_input = input.at(0).shape.size();
  size_t num_dims_output = output.at(0).shape.size();
  auto test = input.at(0);
  if (num_dims_input < 1 || num_dims_output < 1) {
    throw std::runtime_error(
        "Input and output tensors must have at least 1 dimension");
  }

  size_t input_elmnts =
      std::accumulate(input.at(0).shape.begin(), input.at(0).shape.end(),
                      size_t(1), std::multiplies<size_t>());
  size_t output_elmnts =
      std::accumulate(output.at(0).shape.begin(), output.at(0).shape.end(),
                      size_t(1), std::multiplies<size_t>());
  size_t input_size = input_elmnts * Utils::get_size_of_type(input.at(0).dtype);
  size_t output_size =
      output_elmnts * Utils::get_size_of_type(output.at(0).dtype);
  if (input_size != output_size) {
    throw std::runtime_error(
        "Input and output buffer size should be same identity op");
  }

  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, input_size},
      {OpArgMap::OpArgType::OUTPUT, 1, 1, 0, output_size},
  };
  return arg_map;
}

identity::identity() {}

} // namespace ryzenai
